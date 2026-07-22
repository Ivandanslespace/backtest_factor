import hashlib
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import func
from BacktestEngine import PtfBuilder
from factor_config import make_signal_config, make_signal_dimensions, signal_options


def _screen_minimal():
    rows = []
    dates = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])
    for security_index, isin in enumerate(['A', 'B', 'C']):
        for date_index, date in enumerate(dates):
            rows.append({
                'ISIN': isin,
                'Date': date,
                ' Benchmark ICB Supersector ': 'Industrials',
                'Exchange Country Region': 'Europe',
                'Revenue 5Y CAGR': security_index + date_index + 1.0,
                'Net Debt to Ebit': 5.0 - security_index + date_index,
            })
    return pd.DataFrame(rows)


def _fake_backtest(screen, returns, metric, list_noire_path, metadata=None, **options):
    components = (metadata or {}).get('components', [])
    return {
        'metadata': {**(metadata or {}), 'metric': metric},
        'composition': pd.DataFrame(components),
        'raw_variables': list(dict.fromkeys(
            component['raw_variable'] for component in components
        )),
        'raw_variable_weights': func.summarize_component_weights(components),
        'figure': None,
        'robust_score': 0.0,
        'top_bench_ratio': 0.0,
        'top_worst_ratio': 0.0,
    }


def _deterministic_backtest_data():
    """Construit un petit marché stable destiné aux tests d'équivalence."""
    benchmark = 'STOXX EUROPE 600'
    dates = pd.date_range('2023-12-31', periods=8, freq='ME')
    rows = []
    sedols = []
    for sector in range(1, 20):
        for company in range(4):
            isin = f'ISIN{sector:02d}{company}'
            sedol = f'S{sector:02d}{company:02d}-R'
            sedols.append(sedol)
            for date_index, date in enumerate(dates):
                rows.append({
                    'Date': date,
                    'ISIN': isin,
                    'Company SEDOL': sedol,
                    ' Benchmark ICB Supersector ': float(sector),
                    f'Weight in {benchmark}': 1 / 76,
                    'Benchmark Market Value Millions in EUR ': float(
                        100 + sector * 10 + company
                    ),
                    'Signal': float(company * 10 + sector + date_index / 10),
                })
    screen = pd.DataFrame(rows)
    return_dates = pd.bdate_range('2024-01-01', '2024-09-10')
    observations = np.arange(len(return_dates))
    returns = pd.DataFrame({
        sedol: (
            ((index % 7) - 3) * 0.00005
            + np.sin(observations + index) * 0.0001
        )
        for index, sedol in enumerate(sedols)
    }, index=return_dates)
    return screen, returns, benchmark


def _frame_digest(frame):
    """Calcule une empreinte stricte et reproductible d'un tableau de résultat."""
    content = frame.to_csv(
        index=True, float_format='%.17g', date_format='%Y-%m-%dT%H:%M:%S',
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _assert_batches_exactly_equal(test_case, sequential, parallel):
    """Compare les sorties financières de deux lots dans leur ordre d'origine."""
    test_case.assertEqual(
        list(sequential['results']), list(parallel['results']),
    )
    frame_names = (
        'performance', 'top_holdings', 'worst_holdings',
        'ratios', 'period_metrics', 'composition',
    )
    scalar_names = (
        'robust_score', 'top_bench_ratio', 'top_worst_ratio',
        'active_max_drawdown', 'tracking_error_annualized',
        'min_rolling_3y_cagr',
    )
    for result_name in sequential['results']:
        expected = sequential['results'][result_name]
        actual = parallel['results'][result_name]
        for frame_name in frame_names:
            pd.testing.assert_frame_equal(
                expected[frame_name], actual[frame_name], check_exact=True,
            )
        pd.testing.assert_frame_equal(
            pd.DataFrame(expected['classic_metrics']),
            pd.DataFrame(actual['classic_metrics']),
            check_exact=True,
        )
        for scalar_name in scalar_names:
            np.testing.assert_equal(
                expected[scalar_name], actual[scalar_name],
            )


class TestChargementDonnees(unittest.TestCase):
    """Vérifie la projection des colonnes et la fenêtre historique chargée."""

    def test_start_date_conserve_uniquement_le_lookback_demande(self):
        benchmark = 'STOXX EUROPE 600'
        screen = pd.DataFrame({
            'Date': pd.to_datetime([
                '2022-12-31', '2023-06-30', '2023-12-31', '2024-01-31',
            ]),
            'ISIN': ['A', 'A', 'A', 'A'],
            'Company SEDOL': ['AAA-R'] * 4,
            ' Benchmark ICB Supersector ': [1.0] * 4,
            'Exchange Country Region': ['Europe'] * 4,
            f'Weight in {benchmark}': [1.0] * 4,
            'Benchmark Market Value Millions in EUR ': [100.0] * 4,
            'Signal': [1.0, 2.0, 3.0, 4.0],
            'Colonne inutile': [9.0] * 4,
        })
        returns = pd.DataFrame(
            {'AAA-R': [0.01, 0.02, 0.03], 'HORS-BENCH': [0.0, 0.0, 0.0]},
            index=pd.to_datetime(['2023-12-29', '2024-01-02', '2024-01-03']),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            screen_path = directory / 'screen.parquet'
            returns_path = directory / 'returns.parquet'
            screen.to_parquet(screen_path, index=False)
            returns.to_parquet(returns_path)
            loaded_screen, loaded_returns = func.load_backtest_data(
                screen_path,
                returns_path,
                variables=['Signal'],
                bench=benchmark,
                start_date='2024-01-01',
                lookback_periods=6,
            )

        self.assertEqual(loaded_screen['Date'].min(), pd.Timestamp('2023-12-31'))
        self.assertNotIn('Colonne inutile', loaded_screen.columns)
        self.assertEqual(list(loaded_returns.columns), ['AAA-R'])
        self.assertEqual(loaded_returns.index.min(), pd.Timestamp('2024-01-02'))
        self.assertIsInstance(
            loaded_screen['Company SEDOL'].dtype, pd.CategoricalDtype,
        )
        self.assertIsInstance(
            loaded_screen['Exchange Country Region'].dtype,
            pd.CategoricalDtype,
        )
        self.assertEqual(str(loaded_screen[' Benchmark ICB Supersector '].dtype), 'Int8')

    def test_lookback_negatif_est_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            screen_path = Path(directory) / 'screen.parquet'
            returns_path = Path(directory) / 'returns.parquet'
            pd.DataFrame({'Date': pd.to_datetime(['2024-01-31'])}).to_parquet(
                screen_path, index=False,
            )
            pd.DataFrame(index=pd.to_datetime(['2024-01-31'])).to_parquet(
                returns_path,
            )
            with self.assertRaises(ValueError):
                func.load_backtest_data(
                    screen_path, returns_path, start_date='2024-01-01',
                    lookback_periods=-1,
                )


class TestConfigurationSignaux(unittest.TestCase):
    """Vérifie les règles de direction et les configurations réellement utilisées."""

    def test_liste_unitaire_detecte_automatiquement_la_direction(self):
        screen = _screen_minimal()
        with patch.object(func, 'run_top_worst_backtest', side_effect=_fake_backtest):
            batch = func.test_unitary_signals(
                screen, pd.DataFrame(), ['Net Debt to Ebit'], None,
                dimensions=('level',),
            )

        component = batch['results']['Net Debt to Ebit | level']['metadata'][
            'components'
        ][0]
        self.assertFalse(component['higher_is_better'])
        self.assertEqual(set(batch), {'screen', 'results'})

    def test_unitaire_sans_dimensions_utilise_uniquement_une_periode(self):
        screen = _screen_minimal()
        with patch.object(func, 'run_top_worst_backtest', side_effect=_fake_backtest):
            batch = func.test_unitary_signals(
                screen, pd.DataFrame(), ['Revenue 5Y CAGR'], None,
            )

        self.assertEqual(list(batch['results']), [
            'Revenue 5Y CAGR | level',
            'Revenue 5Y CAGR | pct_1',
            'Revenue 5Y CAGR | diff_1',
            'Revenue 5Y CAGR | rank_diff_1',
        ])
        self.assertFalse(any(
            str(column).startswith('Unitary_')
            for column in batch['screen'].columns
        ))
        self.assertIn('Revenue 5Y CAGR__pct_1', batch['screen'].columns)
        self.assertIn('Revenue 5Y CAGR__diff_1', batch['screen'].columns)
        self.assertIn('Revenue 5Y CAGR__rank_diff_1', batch['screen'].columns)

    def test_plusieurs_horizons_partagent_un_seul_tri(self):
        screen = _screen_minimal()
        options = signal_options(pct_1=1.0, diff_3=1.0, rank_diff_1=1.0)
        original_sort_values = pd.DataFrame.sort_values
        with patch.object(
            pd.DataFrame,
            'sort_values',
            autospec=True,
            side_effect=original_sort_values,
        ) as sort_values:
            result = func.calculate_composite_score(
                screen, 'Score', {'Revenue 5Y CAGR': options},
            )

        self.assertEqual(sort_values.call_count, 1)
        for dimension in ('pct_1', 'diff_3', 'rank_diff_1'):
            self.assertIn(f'Revenue 5Y CAGR__{dimension}', result.columns)

    def test_generation_de_toutes_les_dimensions_unitaires(self):
        dimensions = make_signal_dimensions(periods=(1, 3, 6, 12))

        self.assertEqual(dimensions, (
            'level',
            'pct_1', 'pct_3', 'pct_6', 'pct_12',
            'diff_1', 'diff_3', 'diff_6', 'diff_12',
            'rank_diff_1', 'rank_diff_3', 'rank_diff_6', 'rank_diff_12',
        ))

        screen = _screen_minimal()
        with patch.object(func, 'run_top_worst_backtest', side_effect=_fake_backtest):
            batch = func.test_unitary_signals(
                screen, pd.DataFrame(), ['Revenue 5Y CAGR'], None,
                dimensions=dimensions,
            )
        self.assertEqual(len(batch['results']), 13)

    def test_anciens_poids_designent_une_periode(self):
        options = signal_options(pct=0.3, diff=0.4, rank_diff=0.5)

        self.assertEqual(options['weight_pct_1'], 0.3)
        self.assertEqual(options['weight_diff_1'], 0.4)
        self.assertEqual(options['weight_rank_diff_1'], 0.5)
        self.assertNotIn('weight_pct', options)

    def test_composition_exclut_les_variables_absentes(self):
        screen = _screen_minimal()
        config = {
            'Revenue 5Y CAGR': signal_options(level=1.0),
            'Variable absente': signal_options(level=1.0),
        }
        with patch.object(func, 'run_top_worst_backtest', side_effect=_fake_backtest):
            batch = func.test_composite_signals(
                screen, pd.DataFrame(), {'Mixte': config}, None,
            )

        result = batch['results']['Mixte']
        self.assertEqual(result['raw_variables'], ['Revenue 5Y CAGR'])
        self.assertNotIn('Variable absente', result['composition']['raw_variable'].tolist())

    def test_rank_diff_mesure_l_amelioration_du_rang_oriente(self):
        dates = pd.to_datetime(['2024-01-31', '2024-02-29'])
        values = {
            'A': ((1.0, 3.0), 'Industrials'),
            'B': ((2.0, 2.0), 'Industrials'),
            'C': ((3.0, 1.0), 'Industrials'),
            'D': ((1000.0, 0.0), 'Banks'),
        }
        rows = []
        for isin, (observations, industry) in values.items():
            for date, value in zip(dates, observations):
                rows.append({
                    'ISIN': isin,
                    'Date': date,
                    ' Benchmark ICB Supersector ': industry,
                    'Exchange Country Region': 'Europe',
                    'Net Debt to Ebit': value,
                })
        screen = pd.DataFrame(rows)
        config = {
            'Net Debt to Ebit': signal_options(
                higher_is_better=False, rank_diff=1.0,
            ),
        }

        scored = func.calculate_composite_score(screen, 'Score_Rank', config)
        latest = scored.loc[scored['Date'].eq(dates[-1])].set_index('ISIN')
        composition = func.describe_signal_config(config)

        self.assertLess(latest.loc['A', 'Net Debt to Ebit__rank_diff_1'], 0)
        self.assertGreater(latest.loc['C', 'Net Debt to Ebit__rank_diff_1'], 0)
        self.assertEqual(latest.loc['D', 'Net Debt to Ebit__rank_diff_1'], 0)
        self.assertLess(latest.loc['A', 'Score_Rank'], latest.loc['C', 'Score_Rank'])
        self.assertTrue(composition[0]['higher_is_better'])
        self.assertFalse(composition[0]['source_higher_is_better'])
        generated = make_signal_config(
            variables=['Net Debt to Ebit'], transformations=('rank_diff',),
        )
        self.assertEqual(generated['Net Debt to Ebit']['weight_rank_diff_1'], 1.0)

    def test_comparaisons_explicitent_trois_six_et_douze_periodes(self):
        dates = pd.date_range('2023-01-31', periods=13, freq='ME')
        rows = []
        for isin, observations in {
            'A': range(1, 14),
            'B': range(13, 0, -1),
        }.items():
            for date, value in zip(dates, observations):
                rows.append({
                    'ISIN': isin,
                    'Date': date,
                    ' Benchmark ICB Supersector ': 'Industrials',
                    'Exchange Country Region': 'Europe',
                    'Signal': float(value),
                })
        screen = pd.DataFrame(rows)
        config = {
            'Signal': signal_options(
                pct_3=1.0, pct_6=1.0, pct_12=1.0,
                diff_3=1.0, diff_6=1.0, diff_12=1.0,
                rank_diff_3=1.0, rank_diff_6=1.0, rank_diff_12=1.0,
            ),
        }

        scored = func.calculate_composite_score(screen, 'Score_Multi', config)
        latest = scored.loc[
            scored['Date'].eq(dates[-1]) & scored['ISIN'].eq('A')
        ].iloc[0]

        self.assertAlmostEqual(latest['Signal__diff_3'], 3.0)
        self.assertAlmostEqual(latest['Signal__diff_6'], 6.0)
        self.assertAlmostEqual(latest['Signal__diff_12'], 12.0)
        self.assertAlmostEqual(latest['Signal__pct_3'], 13 / 10 - 1)
        self.assertAlmostEqual(latest['Signal__pct_6'], 13 / 7 - 1)
        self.assertAlmostEqual(latest['Signal__pct_12'], 13 / 1 - 1)
        self.assertGreater(latest['Signal__rank_diff_12'], 0)


class TestLotsStandardises(unittest.TestCase):
    """Vérifie la structure commune et le parcours destiné aux exports."""

    def test_les_trois_tests_retournent_screen_et_results(self):
        screen = _screen_minimal()
        with patch.object(func, 'run_top_worst_backtest', side_effect=_fake_backtest):
            unitary = func.test_unitary_signals(
                screen, pd.DataFrame(), ['Revenue 5Y CAGR'], None,
                dimensions=('level',),
            )
            incremental = func.test_incremental_signals(
                unitary['screen'], pd.DataFrame(),
                {'Revenue 5Y CAGR': signal_options(level=1.0)},
                {'Net Debt to Ebit': signal_options(
                    higher_is_better=False, level=1.0,
                )},
                None,
            )
            composite = func.test_composite_signals(
                incremental['screen'], pd.DataFrame(),
                {'Mixte': {'Revenue 5Y CAGR': signal_options(level=1.0)}},
                None,
            )

        for batch in (unitary, incremental, composite):
            self.assertEqual(set(batch), {'screen', 'results'})

        paths = [
            path for path, _ in func._iter_backtest_results({
                'unitary': unitary,
                'incremental': incremental,
                'composite': composite,
            })
        ]
        self.assertEqual(paths, [
            'unitary / Revenue 5Y CAGR | level',
            'incremental / Baseline',
            'incremental / Net Debt to Ebit',
            'composite / Mixte',
        ])


class TestBenchmarkExplicite(unittest.TestCase):
    """Vérifie l'injection directe de la performance du benchmark."""

    def test_ptf_builder_copie_le_benchmark_fourni(self):
        benchmark = pd.Series(
            [1.0, 1.1], index=pd.to_datetime(['2024-01-01', '2024-01-02']),
        )
        builder = PtfBuilder(
            screen=pd.DataFrame(), returns=pd.DataFrame(), bench='Benchmark',
            percentile=0.13, metrics='Score', liste_noire=None,
            bench_perf=benchmark,
        )
        self.assertTrue(builder.perf_bench.equals(benchmark))
        self.assertIsNot(builder.perf_bench, benchmark)

    def test_un_benchmark_fourni_n_est_pas_recalcule(self):
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        benchmark = pd.Series([1.0, 1.01, 1.02], index=dates)
        top = PtfBuilder.__new__(PtfBuilder)
        worst = PtfBuilder.__new__(PtfBuilder)
        top.perf_ptf = pd.Series([1.0, 1.02, 1.04], index=dates)
        worst.perf_ptf = pd.Series([1.0, 0.99, 0.98], index=dates)
        top.perf_bench = benchmark
        worst.perf_bench = benchmark.copy()
        top.sec_list_historical = pd.DataFrame()
        worst.sec_list_historical = pd.DataFrame()
        top.backtest_get_bench_perf = Mock()
        top._calculate_robust_score = lambda *args: (0.0, 0.0, 0.0)
        top._calculate_classic_metrics = lambda *args, **kwargs: {}
        top._calculate_period_metrics = lambda *args, **kwargs: pd.DataFrame()

        result = PtfBuilder.calculate_top_vs_bottom_results(
            top, worst, period_breakpoints=[],
        )

        top.backtest_get_bench_perf.assert_not_called()
        self.assertTrue(result['performance']['Bench'].equals(benchmark))

    def test_run_options_sont_transmises_aux_builders(self):
        benchmark = pd.Series([1.0], index=pd.to_datetime(['2024-01-01']))
        top = Mock()
        worst = Mock()
        top.sec_list_historical = pd.DataFrame()
        worst.sec_list_historical = pd.DataFrame()
        top.calculate_top_vs_bottom_results.return_value = {
            'robust_score': 0.0,
            'top_bench_ratio': 0.0,
            'top_worst_ratio': 0.0,
            'performance': pd.DataFrame(),
            'ratios': pd.DataFrame(),
            'classic_metrics': {},
            'period_metrics': pd.DataFrame(),
        }
        with (
            patch.object(func, '_backtest_inputs', return_value=(
                pd.DataFrame(), pd.DataFrame(),
            )),
            patch.object(
                func, 'PtfBuilder', side_effect=[top, worst, top, worst],
            ) as constructor,
        ):
            result = func.run_top_worst_backtest(
                pd.DataFrame(), pd.DataFrame(), 'Score', None,
                bench='Benchmark', bench_perf=benchmark,
                start_date='2021-02-03', freq_rebal=3,
                fill_method='drift', period_breakpoints=[],
                show_plot=False, build_figure=False,
            )
            retained_result = func.run_top_worst_backtest(
                pd.DataFrame(), pd.DataFrame(), 'Score', None,
                bench='Benchmark', bench_perf=benchmark,
                start_date='2021-02-03', freq_rebal=3,
                fill_method='drift', period_breakpoints=[],
                show_plot=False, build_figure=False,
                retain_builders=True,
            )

        for call in constructor.call_args_list:
            self.assertIs(call.kwargs['bench_perf'], benchmark)
        for builder in (top, worst):
            self.assertEqual(builder.start_date, pd.Timestamp('2021-02-03'))
            self.assertEqual(builder.freq_rebal, 3)
            self.assertEqual(builder.fill_method, 'drift')
        self.assertTrue(result['metadata']['benchmark_performance_provided'])
        self.assertFalse(result['metadata']['builders_retained'])
        self.assertNotIn('top_builder', result)
        self.assertNotIn('worst_builder', result)
        self.assertTrue(retained_result['metadata']['builders_retained'])
        self.assertIs(retained_result['top_builder'], top)
        self.assertIs(retained_result['worst_builder'], worst)


class TestOptimisationsMemoire(unittest.TestCase):
    """Vérifie le partage des entrées et de la préparation mensuelle."""

    def test_builders_partagent_les_entrees_en_lecture_seule(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        builder = PtfBuilder(
            screen=screen, returns=returns, bench=benchmark,
            percentile=0.13, metrics='Signal', liste_noire=None,
        )

        self.assertIs(builder.screen, screen)
        self.assertIs(builder.returns, returns)

    def test_projection_ne_modifie_pas_le_screen_source(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        target = 'Benchmark Market Value Millions in EUR '
        source = target.rstrip()
        screen.rename(columns={target: source}, inplace=True)

        projected_screen, _ = func._backtest_inputs(
            screen, returns, metric='Signal', bench=benchmark,
        )

        self.assertNotIn(target, screen.columns)
        self.assertIn(target, projected_screen.columns)

    def test_secteur_manquant_conserve_le_rang_global_historique(self):
        builder = PtfBuilder.__new__(PtfBuilder)
        builder.score_neutral = 'ICB 19'
        frame = pd.DataFrame({
            'Score': [1.0, 2.0, 3.0],
            ' Benchmark ICB Supersector ': [1.0, 1.0, np.nan],
        })

        result = builder.neutralise_score_by_secteur(frame, ['Score'])

        self.assertEqual(result['Score'].tolist(), [0.0, 1.0, 1.0])

    def test_worst_reutilise_la_preparation_mensuelle_du_top(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        top = PtfBuilder(
            screen=screen, returns=returns, bench=benchmark,
            percentile=0.13, metrics='Signal', liste_noire=None, Top=True,
            esg_exclusion=0,
        )
        worst = PtfBuilder(
            screen=screen, returns=returns, bench=benchmark,
            percentile=0.13, metrics='Signal', liste_noire=None, Top=False,
            esg_exclusion=0,
        )

        with (
            warnings.catch_warnings(),
            patch('tqdm.tqdm', side_effect=lambda values, **kwargs: values),
            patch.object(worst, 'sec_list_spot', wraps=worst.sec_list_spot) as call,
        ):
            warnings.simplefilter('ignore')
            top.generic_histo_seclists_pair(
                worst, start_date=pd.Timestamp('2024-01-01'),
                freq_rebal=1, fill_method='copy',
            )

        call.assert_not_called()
        self.assertFalse(top.sec_list_historical.empty)
        self.assertFalse(worst.sec_list_historical.empty)

    def test_deux_signaux_reutilisent_la_meme_base_mensuelle(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        screen['Signal 2'] = screen['Signal']
        cache = {}
        original_polyfit = np.polyfit
        with (
            warnings.catch_warnings(),
            patch('tqdm.tqdm', new=lambda values, **kwargs: values),
            patch(
                'BacktestEngine.np.polyfit', side_effect=original_polyfit,
            ) as polyfit,
        ):
            warnings.simplefilter('ignore')
            benchmark_performance = func.calculate_benchmark_performance(
                screen, returns, bench=benchmark, start_date='2024-01-01',
            )
            first = func.run_top_worst_backtest(
                screen, returns, 'Signal', None,
                bench=benchmark, bench_perf=benchmark_performance,
                start_date='2024-01-01', period_breakpoints=[],
                show_plot=False, build_figure=False,
                monthly_base_cache=cache,
            )
            first_call_count = polyfit.call_count
            cached_frames = [
                value['df'] for key, value in cache.items()
                if key != '_source_id'
            ]
            second = func.run_top_worst_backtest(
                screen, returns, 'Signal 2', None,
                bench=benchmark, bench_perf=benchmark_performance,
                start_date='2024-01-01', period_breakpoints=[],
                show_plot=False, build_figure=False,
                monthly_base_cache=cache,
            )

        self.assertGreater(first_call_count, 0)
        self.assertEqual(polyfit.call_count, first_call_count)
        self.assertTrue(cached_frames)
        self.assertTrue(all(
            'Company SEDOL' not in frame.columns for frame in cached_frames
        ))
        self.assertTrue(all(
            not isinstance(frame.index, pd.CategoricalIndex)
            for frame in cached_frames
        ))
        pd.testing.assert_frame_equal(
            first['performance'], second['performance'], check_exact=True,
        )


class TestParallelisationSignaux(unittest.TestCase):
    """Vérifie l'équivalence stricte des lots séquentiels et parallèles."""

    @staticmethod
    def _inputs():
        screen, returns, benchmark = _deterministic_backtest_data()
        screen['Exchange Country Region'] = 'Europe'
        screen['Signal 2'] = screen['Signal']
        screen['Signal 3'] = screen['Signal']
        benchmark_performance = func.calculate_benchmark_performance(
            screen, returns, bench=benchmark, start_date='2024-01-01',
        )
        options = {
            'bench': benchmark,
            'bench_perf': benchmark_performance,
            'start_date': '2024-01-01',
            'period_breakpoints': [],
            'show_plot': False,
            'build_figure': False,
        }
        return screen, returns, options

    def test_signaux_unitaires_paralleles_restent_exacts(self):
        screen, returns, options = self._inputs()
        config = {
            variable: signal_options(level=1.0)
            for variable in ('Signal', 'Signal 2', 'Signal 3')
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sequential = func.test_unitary_signals(
                screen.copy(), returns, config, None,
                dimensions=('level',), n_jobs=1,
                monthly_base_cache={}, **options,
            )
            parallel = func.test_unitary_signals(
                screen.copy(), returns, config, None,
                dimensions=('level',), n_jobs=2,
                monthly_base_cache={}, **options,
            )

        _assert_batches_exactly_equal(self, sequential, parallel)
        self.assertFalse(any(
            str(column).startswith('Unitary_')
            for column in parallel['screen'].columns
        ))

    def test_composites_paralleles_restent_exacts_et_ordonnes(self):
        screen, returns, options = self._inputs()
        options = {**options, 'build_figure': True}
        composites = {
            name: {variable: signal_options(level=1.0)}
            for name, variable in (
                ('Composite A', 'Signal'),
                ('Composite B', 'Signal 2'),
                ('Composite C', 'Signal 3'),
            )
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sequential = func.test_composite_signals(
                screen.copy(), returns, composites, None,
                n_jobs=1, monthly_base_cache={}, **options,
            )
            parallel = func.test_composite_signals(
                screen.copy(), returns, composites, None,
                n_jobs=2, monthly_base_cache={}, **options,
            )

        _assert_batches_exactly_equal(self, sequential, parallel)
        self.assertEqual(
            list(parallel['results']), list(composites),
        )
        self.assertTrue(all(
            result['figure'] is not None
            for result in parallel['results'].values()
        ))

    def test_candidats_incrementaux_paralleles_restent_exacts(self):
        screen, returns, options = self._inputs()
        baseline = {'Signal': signal_options(level=1.0)}
        candidates = {
            variable: signal_options(level=1.0)
            for variable in ('Signal 2', 'Signal 3')
        }
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sequential = func.test_incremental_signals(
                screen.copy(), returns, baseline, candidates, None,
                n_jobs=1, monthly_base_cache={}, **options,
            )
            parallel = func.test_incremental_signals(
                screen.copy(), returns, baseline, candidates, None,
                n_jobs=2, monthly_base_cache={}, **options,
            )

        _assert_batches_exactly_equal(self, sequential, parallel)
        self.assertEqual(
            list(parallel['results']), ['Baseline', 'Signal 2', 'Signal 3'],
        )

    def test_n_jobs_invalide_est_refuse(self):
        screen = _screen_minimal()
        with self.assertRaisesRegex(ValueError, 'n_jobs'):
            func.test_composite_signal(
                screen, pd.DataFrame(), 'Score',
                {'Revenue 5Y CAGR': signal_options(level=1.0)},
                None, n_jobs=0,
            )

    def test_options_lourdes_sont_refusees_en_parallele(self):
        config = {
            variable: signal_options(level=1.0)
            for variable in ('Revenue 5Y CAGR', 'Net Debt to Ebit')
        }
        for option, value in (
            ('retain_builders', True),
            ('save_path', Path('resultat_unique.html')),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, option,
            ):
                func.test_unitary_signals(
                    _screen_minimal(), pd.DataFrame(), config, None,
                    dimensions=('level',), n_jobs=2,
                    **{option: value},
                )


class TestReconstructionPeriodes(unittest.TestCase):
    """Vérifie la réunion de la période totale et des sous-périodes."""

    def test_reconstruction_reunit_total_et_sous_periode(self):
        summary = pd.DataFrame([{
            'test_path': 'unitary / Signal | level',
            'test_name': 'Signal | level',
            'test_type': 'unitary',
            'metric': 'Unitary_level_Signal',
            'start_date': '2010-01-01',
            'observation_count': 1000,
            'top_annualized_return': 0.08,
            'worst_annualized_return': 0.02,
            'bench_annualized_return': 0.05,
            'robust_score': 0.1,
        }])
        periods = pd.DataFrame([{
            'test_path': 'unitary / Signal | level',
            'test_name': 'Signal | level',
            'test_type': 'unitary',
            'metric': 'Unitary_level_Signal',
            'period_id': 'depuis_2020',
            'period_label': 'Depuis 2020',
            'active_cagr': 0.03,
        }])

        combined = func._combine_total_and_period_metrics(summary, periods)

        self.assertEqual(combined['scope'].tolist(), ['total', 'subperiod'])
        self.assertEqual(combined['period_label'].tolist(), [
            'Période totale', 'Depuis 2020',
        ])


class TestEquivalenceMoteur(unittest.TestCase):
    """Fige les positions, performances et métriques du moteur de référence."""

    def test_positions_performances_et_metriques_restent_identiques(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        with (
            warnings.catch_warnings(),
            patch('tqdm.tqdm', new=lambda iterable, **kwargs: iterable),
        ):
            warnings.simplefilter('ignore')
            benchmark_performance = func.calculate_benchmark_performance(
                screen, returns, bench=benchmark, start_date='2024-01-01',
            )
            result = func.run_top_worst_backtest(
                screen, returns, 'Signal', None,
                bench=benchmark, bench_perf=benchmark_performance,
                start_date='2024-01-01', period_breakpoints=[],
                show_plot=False, build_figure=False,
            )

        expected = {
            'performance': '448c604bccd590c6d96cbf3a719bb108929170839c00fa9a80a23a569115995f',
            'top_holdings': '838808f9e98aec40e114e15ab41a28e9ae288814aee17b6428d24f6d8ae28758',
            'worst_holdings': 'bd8406e4bb86ff63e8eea684b66fb6deb5cafc9b6b3f71ba44b10116e02732f0',
            'period_metrics': 'e0fa7356c7f23fd8ac340259928799a6b477e71b5ba63e88912aa41c920251b7',
        }
        for name, digest in expected.items():
            self.assertEqual(_frame_digest(result[name]), digest, name)
        self.assertEqual(
            hashlib.sha256(repr(result['classic_metrics']).encode()).hexdigest(),
            'e48ca6a22d763d21a6a54c8cd3e8ebd5c52ab9974cdfed161ff08c9c34b9a476',
        )

    def test_isin_peut_rester_index_du_screen_parquet(self):
        screen, returns, benchmark = _deterministic_backtest_data()
        screen = screen.set_index('ISIN')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            performance = func.calculate_benchmark_performance(
                screen, returns, bench=benchmark, start_date='2024-01-01',
            )

        self.assertFalse(performance.empty)


if __name__ == '__main__':
    unittest.main()
