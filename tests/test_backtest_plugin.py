import unittest
from unittest.mock import Mock, patch

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
            patch.object(func, 'PtfBuilder', side_effect=[top, worst]) as constructor,
        ):
            result = func.run_top_worst_backtest(
                pd.DataFrame(), pd.DataFrame(), 'Score', None,
                bench='Benchmark', bench_perf=benchmark,
                start_date='2021-02-03', freq_rebal=3,
                fill_method='drift', period_breakpoints=[],
                show_plot=False, build_figure=False,
            )

        for call in constructor.call_args_list:
            self.assertIs(call.kwargs['bench_perf'], benchmark)
        for builder in (top, worst):
            self.assertEqual(builder.start_date, pd.Timestamp('2021-02-03'))
            self.assertEqual(builder.freq_rebal, 3)
            self.assertEqual(builder.fill_method, 'drift')
        self.assertTrue(result['metadata']['benchmark_performance_provided'])


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


if __name__ == '__main__':
    unittest.main()
