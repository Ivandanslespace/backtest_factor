"""Outils génériques pour construire et tester des signaux factoriels.

Le dictionnaire SIGNAL_CONFIG ci-dessous est uniquement un exemple de format.
Dans un notebook, définissez votre propre configuration dans une cellule puis
passez-la explicitement aux fonctions de ce module.
"""

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

try:
    from BacktestEngine import PtfBuilder, build_periods_from_breakpoints
except ImportError:
    from Codes.BacktestEngine import PtfBuilder, build_periods_from_breakpoints


GROUP_COLS = [' Benchmark ICB Supersector ', 'Date', 'Exchange Country Region']
DEFAULT_BENCHMARK = 'STOXX EUROPE 600'
DEFAULT_PERCENTILE = 0.13
DEFAULT_START_DATE = '2010-01-01'
CLASSIC_METRIC_NAMES = (
    'total_return', 'annualized_return', 'annualized_volatility',
    'sharpe_ratio', 'max_drawdown', 'sortino_ratio',
    'beta', 'tracking_error', 'information_ratio',
)
CLASSIC_METRIC_COLUMNS = tuple(
    f'{portfolio}_{metric}'
    for portfolio in ('top', 'worst', 'bench')
    for metric in CLASSIC_METRIC_NAMES
)
PERIOD_SUMMARY_METRICS = (
    'observation_count', 'top_cagr', 'active_cagr', 'top_worst_cagr',
    'top_sharpe_ratio', 'top_max_drawdown', 'top_sortino_ratio',
    'top_information_ratio',
)


# Ruptures recommandées pour un historique démarrant en 2010 :
# 2020 pour la pandémie, 2022 pour le régime inflationniste et 2024 pour la normalisation.
# Ajoutez 2009 si les données couvrent aussi la période précédant la crise financière.
RECOMMENDED_PERIOD_BREAKPOINTS = [2020, 2022, 2024]


_BENCHMARK_PERFORMANCE_CACHE = {}
_DATA_SOURCE_TOKEN = '_backtest_source_token'


# Exemple de configuration : les catégories métier ne déterminent pas le flux.
SIGNAL_CONFIG = {
    'Quality Avg Percentile': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': True, 'weight_pct': 1.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Revenue 5Y CAGR': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Sales Growth FY1 CIQ': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Ebitda 5Y CAGR': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'EBITDA Growth FY1 CIQ': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Ebit 5Y CAGR': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'EPS Growth FY1 CIQ': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'SP Est 5Y EPS Gr CIQ': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'CFO 5Y CAGR': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'FCF Conversion': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': True, 'weight_diff': 1.0,
    },
    'Gross Profit 5Y CAGR': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Const Earning 5Y CAGR': {
        'higher_is_better': True,
        'use_level': False, 'weight_level': 0.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': True, 'weight_diff': 1.0,
    },
    'Gross Margin': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Ebitda Margin': {
        'higher_is_better': True,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Cont Op Earning Margin': {
        'higher_is_better': True,
        'use_level': False, 'weight_level': 0.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': True, 'weight_diff': 1.0,
    },
    'R&D Expense CIQ': {
        'higher_is_better': True,
        'denominator': 'Sales',
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Capex CIQ': {
        'higher_is_better': True,
        'denominator': 'Sales',
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Sales FY1': {
        'higher_is_better': True,
        'denominator': 'Sales',
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
    'Net Debt to Ebit': {
        'higher_is_better': False,
        'use_level': False, 'weight_level': 0.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': True, 'weight_diff': 1.0,
    },
    'Net Debt to Tot Equity': {
        'higher_is_better': False,
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': True, 'weight_diff': 1.0,
    },
    'Interest expense CIQ': {
        'higher_is_better': False,
        'denominator': 'Ebitda',
        'use_level': True, 'weight_level': 1.0,
        'use_pct': False, 'weight_pct': 0.0,
        'use_diff': False, 'weight_diff': 0.0,
    },
}


def _config_variables(signal_config):
    """Retourne les variables et dénominateurs demandés par une configuration."""
    if signal_config is None:
        return []
    if isinstance(signal_config, dict):
        columns = list(signal_config)
        columns.extend(
            options.get('denominator')
            for options in signal_config.values()
            if isinstance(options, dict) and options.get('denominator')
        )
        return list(dict.fromkeys(columns))
    if isinstance(signal_config, (list, tuple, set, pd.Index)):
        return list(dict.fromkeys(signal_config))
    raise TypeError('La configuration doit être un dictionnaire ou une liste de variables.')


def required_screen_columns(variables=None, signal_config=None,
                            bench=DEFAULT_BENCHMARK):
    """Liste les seules colonnes de screen nécessaires à la préparation et au backtest."""
    columns = [
        'Date', 'ISIN', 'Company SEDOL',
        ' Benchmark ICB Supersector ', 'Exchange Country Region',
        f'Weight in {bench}', 'Benchmark Market Value Millions in EUR ',
    ]
    columns.extend([] if variables is None else list(variables))
    columns.extend(_config_variables(signal_config))
    return list(dict.fromkeys(columns))


def _parquet_columns(path):
    """Lit uniquement le schéma d'un parquet, sans charger ses données."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            'pyarrow est requis pour sélectionner les colonnes parquet avant lecture.'
        ) from error
    return parquet.ParquetFile(path).schema_arrow.names


def _set_source_token(data, token):
    """Attache une identité de source conservée par les copies pandas."""
    data.attrs[_DATA_SOURCE_TOKEN] = str(token)
    return data


def load_backtest_data(screen_path, returns_path, variables=None, signal_config=None,
                       bench=DEFAULT_BENCHMARK):
    """Charge les colonnes utiles du screen et les rendements des membres du benchmark."""
    screen_path = Path(screen_path)
    returns_path = Path(returns_path)
    available_screen_columns = set(_parquet_columns(screen_path))
    requested_columns = required_screen_columns(
        variables=variables, signal_config=signal_config, bench=bench,
    )
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    if market_cap_column not in available_screen_columns:
        requested_columns.remove(market_cap_column)
        requested_columns.append(market_cap_column.rstrip())
    missing_columns = [
        column for column in requested_columns if column not in available_screen_columns
    ]
    if missing_columns:
        raise KeyError(f'Colonnes absentes du screen : {missing_columns}')

    screen = pd.read_parquet(screen_path, columns=requested_columns)
    weight_column = f'Weight in {bench}'
    benchmark_sedols = (
        screen.loc[screen[weight_column].fillna(0).gt(0), 'Company SEDOL']
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    available_return_columns = set(_parquet_columns(returns_path))
    return_columns = [
        sedol for sedol in benchmark_sedols if sedol in available_return_columns
    ]
    if not return_columns:
        raise ValueError('Aucun SEDOL du benchmark n’est disponible dans les rendements.')
    returns = pd.read_parquet(returns_path, columns=return_columns)

    screen_stat = screen_path.stat()
    returns_stat = returns_path.stat()
    _set_source_token(
        screen,
        f'{screen_path.resolve()}:{screen_stat.st_size}:{screen_stat.st_mtime_ns}',
    )
    _set_source_token(
        returns,
        f'{returns_path.resolve()}:{returns_stat.st_size}:{returns_stat.st_mtime_ns}',
    )
    print(
        f'Données chargées : {len(screen.columns)} colonnes screen et '
        f'{len(returns.columns)} colonnes de rendements du benchmark.'
    )
    return screen, returns


def clear_benchmark_performance_cache():
    """Vide le cache après toute modification des données du benchmark."""
    _BENCHMARK_PERFORMANCE_CACHE.clear()


def _data_source_token(data):
    """Identifie une source en mémoire tout en gardant le jeton de ses copies."""
    token = data.attrs.get(_DATA_SOURCE_TOKEN)
    if token is None:
        token = f'memory:{uuid4().hex}'
        data.attrs[_DATA_SOURCE_TOKEN] = token
    return token


def _backtest_inputs(screen, returns, metric, bench):
    """Réduit les deux tables aux colonnes réellement consommées par le moteur."""
    if not isinstance(screen, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError('screen et returns doivent être des DataFrames pandas.')
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    source_market_cap = market_cap_column.rstrip()
    if market_cap_column not in screen.columns:
        if source_market_cap not in screen.columns:
            raise KeyError(f'Colonne requise absente : {market_cap_column}')
        screen[market_cap_column] = screen[source_market_cap]

    weight_column = f'Weight in {bench}'
    screen_columns = [
        'Date', 'Company SEDOL', ' Benchmark ICB Supersector ',
        weight_column, market_cap_column, metric,
    ]
    if 'ISIN' in screen.columns:
        screen_columns.insert(1, 'ISIN')
    elif screen.index.name != 'ISIN':
        raise KeyError('Colonne ou index ISIN absent du screen.')
    screen_columns = list(dict.fromkeys(screen_columns))
    missing_columns = [column for column in screen_columns if column not in screen.columns]
    if missing_columns:
        raise KeyError(f'Colonnes requises absentes pour le backtest : {missing_columns}')

    benchmark_sedols = set(
        screen.loc[screen[weight_column].fillna(0).gt(0), 'Company SEDOL']
        .dropna()
        .astype(str)
    )
    return_columns = [column for column in returns.columns if str(column) in benchmark_sedols]
    if not return_columns:
        raise ValueError('Aucun rendement ne correspond aux membres du benchmark.')

    slim_screen = screen.loc[:, screen_columns].copy()
    slim_returns = returns.loc[:, return_columns].copy()
    _set_source_token(slim_screen, _data_source_token(screen))
    _set_source_token(slim_returns, _data_source_token(returns))
    return slim_screen, slim_returns


def _benchmark_cache_key(screen, returns, bench, start_date):
    """Construit la clé des seules entrées capables de modifier le benchmark."""
    return (
        _data_source_token(screen), _data_source_token(returns),
        str(bench), str(pd.Timestamp(start_date).date()),
    )


def handle_missing_values(df, columns, group_cols=None):
    """Remplace les valeurs manquantes par la médiane du groupe disponible."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(df.groupby(group_cols)[col].transform('median'))
    return df


def neutralize_score(df, score_col, higher_is_better, group_cols=None):
    """Convertit une variable en rang centile de 0 à 10 dans chaque groupe."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    df[score_col] = (
        df.groupby(group_cols)[score_col]
        .rank(pct=True, ascending=higher_is_better) * 10
    )
    return df


def prepare_signals(screen, signal_config, group_cols=None, copy_data=False):
    """Prépare les variables brutes dans screen, ou dans une copie si demandé."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    prepared = screen.copy() if copy_data else screen
    resolved_config = {}
    prepared_cols = []

    for variable, options in signal_config.items():
        if variable not in prepared.columns:
            print(f'Avertissement : {variable} absent des données. Signal ignoré.')
            continue

        denominator = options.get('denominator')
        prepared_variable = variable
        if denominator:
            if denominator not in prepared.columns:
                print(f'Avertissement : dénominateur {denominator} absent pour {variable}. Signal ignoré.')
                continue
            prepared_variable = f'{variable}__over__{denominator}'
            prepared[prepared_variable] = prepared[variable] / prepared[denominator]

        resolved_config[prepared_variable] = copy.deepcopy(options)
        prepared_cols.append(prepared_variable)

    prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
    return handle_missing_values(prepared, prepared_cols, group_cols), resolved_config


def build_signal_component(screen, variable, options, group_cols=None, keep_derived_columns=True):
    """Construit les composantes niveau, variation relative et variation absolue."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    contribution = pd.Series(0.0, index=screen.index)

    components = (
        ('level', variable, options.get('use_level', False), options.get('weight_level', 0.0)),
        ('pct', f'{variable}__pct', options.get('use_pct', False), options.get('weight_pct', 0.0)),
        ('diff', f'{variable}__diff', options.get('use_diff', False), options.get('weight_diff', 0.0)),
    )

    for component, column, enabled, weight in components:
        if not enabled:
            continue
        if component in ('pct', 'diff'):
            isin_values = (
                screen['ISIN'].to_numpy()
                if 'ISIN' in screen.columns else screen.index.to_numpy()
            )
            ordered = pd.DataFrame({
                '_position': np.arange(len(screen)),
                '_isin': isin_values,
                '_date': pd.to_datetime(screen['Date']).to_numpy(),
                '_value': screen[variable].to_numpy(),
            }).sort_values(['_isin', '_date'])
            if component == 'pct':
                filled_values = ordered.groupby('_isin')['_value'].ffill()
                derived = filled_values.groupby(ordered['_isin']).pct_change(
                    fill_method=None
                )
            else:
                derived = ordered.groupby('_isin')['_value'].diff()
            screen[column] = pd.Series(
                derived.to_numpy(), index=ordered['_position']
            ).sort_index().to_numpy()

        screen[column] = screen[column].replace([np.inf, -np.inf], np.nan)
        screen = handle_missing_values(screen, [column], group_cols)
        score_col = f'{column}__score'
        screen[score_col] = screen[column]
        screen = neutralize_score(screen, score_col, options['higher_is_better'], group_cols)
        contribution = contribution.add(screen[score_col] * weight, fill_value=0.0)

        if component != 'level' and not keep_derived_columns:
            screen.drop(columns=[column], inplace=True)
        screen.drop(columns=[score_col], inplace=True)

    return screen, contribution


def calculate_composite_score(screen, score_col, signal_config, group_cols=None,
                              copy_data=False, keep_derived_columns=True):
    """Agrège les signaux et conserve par défaut les variables dérivées dans screen."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    prepared, resolved_config = prepare_signals(
        screen, signal_config, group_cols, copy_data=copy_data,
    )
    total_score = pd.Series(0.0, index=prepared.index)
    active_signals = 0

    for variable, options in resolved_config.items():
        if not any(options.get(f'use_{component}', False) for component in ('level', 'pct', 'diff')):
            continue
        prepared, contribution = build_signal_component(
            prepared, variable, options, group_cols,
            keep_derived_columns=keep_derived_columns,
        )
        total_score = total_score.add(contribution, fill_value=0.0)
        active_signals += 1

    if not active_signals:
        raise ValueError('Aucun signal actif et disponible n’a été fourni.')

    prepared[score_col] = total_score
    return neutralize_score(prepared, score_col, higher_is_better=True, group_cols=group_cols)


def describe_signal_config(signal_config, role='signal'):
    """Convertit une configuration de signaux en composition longue et explicite."""
    components = []
    for raw_variable, options in signal_config.items():
        denominator = options.get('denominator')
        prepared_variable = (
            f'{raw_variable}__over__{denominator}' if denominator else raw_variable
        )
        for dimension in ('level', 'pct', 'diff'):
            if not options.get(f'use_{dimension}', False):
                continue
            derived_variable = (
                prepared_variable if dimension == 'level'
                else f'{prepared_variable}__{dimension}'
            )
            components.append({
                'role': role,
                'raw_variable': raw_variable,
                'prepared_variable': prepared_variable,
                'derived_variable': derived_variable,
                'denominator': denominator,
                'dimension': dimension,
                'higher_is_better': options.get('higher_is_better'),
                'weight': options.get(f'weight_{dimension}', 0.0),
            })
    return components


def summarize_component_weights(components):
    """Regroupe les poids actifs par variable brute et par dimension."""
    summary = {}
    total_absolute_weight = sum(abs(float(item.get('weight', 0.0))) for item in components)
    for component in components:
        raw_variable = component.get('raw_variable')
        weight = float(component.get('weight', 0.0))
        variable = summary.setdefault(raw_variable, {
            'weight_level': 0.0,
            'weight_pct': 0.0,
            'weight_diff': 0.0,
            'total_weight': 0.0,
            'absolute_weight': 0.0,
            'absolute_weight_share': 0.0,
        })
        variable[f"weight_{component.get('dimension')}"] = weight
        variable['total_weight'] += weight
        variable['absolute_weight'] += abs(weight)
    if total_absolute_weight:
        for variable in summary.values():
            variable['absolute_weight_share'] = (
                variable['absolute_weight'] / total_absolute_weight
            )
    return summary


def run_top_worst_backtest(screen, returns, metric, list_noire_path, bench=DEFAULT_BENCHMARK,
                           percentile=DEFAULT_PERCENTILE, show_plot=True,
                           save_path=None, metadata=None, period_breakpoints=None,
                           build_figure=True, use_benchmark_cache=True):
    """Exécute un backtest Top/Worst pour une variable déjà disponible dans screen."""
    backtest_screen, backtest_returns = _backtest_inputs(
        screen, returns, metric=metric, bench=bench,
    )
    cache_key = _benchmark_cache_key(
        backtest_screen, backtest_returns, bench, DEFAULT_START_DATE,
    )
    cached_benchmark = (
        _BENCHMARK_PERFORMANCE_CACHE.get(cache_key)
        if use_benchmark_cache else None
    )
    resolved_breakpoints = list(
        RECOMMENDED_PERIOD_BREAKPOINTS
        if period_breakpoints is None else period_breakpoints
    )
    builder_top = PtfBuilder(
        backtest_screen, backtest_returns, ptf_name=f'{metric}_top', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=True,
    )
    builder_worst = PtfBuilder(
        backtest_screen, backtest_returns, ptf_name=f'{metric}_worst', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=False,
    )

    if cached_benchmark is not None:
        builder_top.perf_bench = cached_benchmark.copy()
        builder_worst.perf_bench = cached_benchmark.copy()
        print('Performance du benchmark réutilisée depuis le cache.')

    for builder in (builder_top, builder_worst):
        builder.start_date = pd.Timestamp(DEFAULT_START_DATE)
        builder.freq_rebal = 1
        builder.fill_method = 'copy'

    comparison = builder_top.calculate_top_vs_bottom_results(
        builder_bottom=builder_worst,
        period_breakpoints=resolved_breakpoints,
    )
    if use_benchmark_cache and cached_benchmark is None:
        _BENCHMARK_PERFORMANCE_CACHE[cache_key] = builder_top.perf_bench.copy()
        print('Performance du benchmark calculée puis mise en cache.')
    print(
        f"Résultats pour {metric} : "
        f"score de robustesse {comparison['robust_score']:.4f}, "
        f"Top/Bench {comparison['top_bench_ratio']:.4f}, "
        f"Top/Worst {comparison['top_worst_ratio']:.4f}"
    )
    should_build_figure = build_figure or show_plot or save_path is not None
    comparison['figure'] = (
        builder_top.plot_top_vs_bottom_results(
            result=comparison,
            title=f'Analyse factorielle : {metric}',
            save_path=save_path,
            show_plot=show_plot,
        )
        if should_build_figure else None
    )
    result = {
        'top_builder': builder_top,
        'worst_builder': builder_worst,
        'top_holdings': builder_top.sec_list_historical,
        'worst_holdings': builder_worst.sec_list_historical,
        'metadata': {
            'metric': metric,
            'benchmark': bench,
            'percentile': percentile,
            'start_date': DEFAULT_START_DATE,
            'frequency_rebalancing': 1,
            'fill_method': 'copy',
            'period_breakpoints': resolved_breakpoints,
            'benchmark_cache_hit': cached_benchmark is not None,
            'components': [],
        },
    }
    if metadata:
        result['metadata'].update(copy.deepcopy(metadata))
    if isinstance(comparison, dict):
        result.update(comparison)
    components = copy.deepcopy(result['metadata'].get('components', []))
    result['composition'] = pd.DataFrame(components)
    result['raw_variables'] = list(dict.fromkeys(
        component.get('raw_variable') for component in components
    ))
    result['raw_variable_weights'] = summarize_component_weights(components)
    return result


def test_unitary_signals(screen, returns, signal_config, list_noire_path,
                         dimensions=('level', 'pct', 'diff'), **backtest_options):
    """Teste séparément les dimensions demandées et accepte aussi une simple liste."""
    results = {}
    if not isinstance(signal_config, dict):
        signal_config = {
            variable: {'higher_is_better': True}
            for variable in _config_variables(signal_config)
        }
    dimension_options = {
        'level': ('use_level', 'weight_level'),
        'pct': ('use_pct', 'weight_pct'),
        'diff': ('use_diff', 'weight_diff'),
    }

    for variable, options in signal_config.items():
        denominator = options.get('denominator')
        if variable not in screen.columns or (denominator and denominator not in screen.columns):
            print(f'Avertissement : données insuffisantes pour {variable}. Signal ignoré.')
            continue
        for label, (enabled_key, weight_key) in dimension_options.items():
            if label not in dimensions:
                continue

            unitary_options = copy.deepcopy(options)
            for key in ('use_level', 'use_pct', 'use_diff'):
                unitary_options[key] = False
            for key in ('weight_level', 'weight_pct', 'weight_diff'):
                unitary_options[key] = 0.0
            unitary_options[enabled_key] = True
            unitary_options[weight_key] = 1.0

            metric = f'Unitary_{label}_{variable}'
            print(f'Test de signal unitaire : {variable} | {label}')
            scored = calculate_composite_score(screen, metric, {variable: unitary_options})
            results[f'{variable} | {label}'] = run_top_worst_backtest(
                scored, returns, metric, list_noire_path,
                metadata={
                    'test_type': 'unitary',
                    'test_name': f'{variable} | {label}',
                    'components': describe_signal_config(
                        {variable: unitary_options}, role='unitary',
                    ),
                },
                **backtest_options,
            )
    return results


def test_incremental_signals(screen, returns, baseline_config, candidate_config,
                             list_noire_path, **backtest_options):
    """Compare un score de base avec ce score enrichi d'un signal candidat."""
    results = {}
    baseline_metric = 'Score_Baseline'
    baseline_screen = calculate_composite_score(screen, baseline_metric, baseline_config)
    results['Baseline'] = {
        'screen': baseline_screen.copy(),
        'backtest': run_top_worst_backtest(
            baseline_screen, returns, baseline_metric, list_noire_path,
            metadata={
                'test_type': 'incremental_baseline',
                'test_name': 'Baseline',
                'components': describe_signal_config(baseline_config, role='baseline'),
            },
            **backtest_options,
        ),
    }

    for variable, options in candidate_config.items():
        if variable in baseline_config:
            print(f'Avertissement : {variable} appartient déjà à la base. Signal ignoré.')
            continue

        incremental_config = copy.deepcopy(baseline_config)
        incremental_config[variable] = copy.deepcopy(options)
        metric = f'Score_Incremental_{variable}'
        print(f'Test incrémental : {variable}')
        incremental_screen = calculate_composite_score(screen, metric, incremental_config)
        results[variable] = {
            'screen': incremental_screen.copy(),
            'backtest': run_top_worst_backtest(
                incremental_screen, returns, metric, list_noire_path,
                metadata={
                    'test_type': 'incremental_candidate',
                    'test_name': variable,
                    'components': (
                        describe_signal_config(baseline_config, role='baseline')
                        + describe_signal_config({variable: options}, role='candidate')
                    ),
                },
                **backtest_options,
            ),
        }
    return results


def test_composite_signal(screen, returns, score_col, signal_config,
                          list_noire_path, **backtest_options):
    """Construit puis teste un score composite tout en enregistrant sa composition."""
    scored_screen = calculate_composite_score(screen, score_col, signal_config)
    backtest_result = run_top_worst_backtest(
        scored_screen, returns, score_col, list_noire_path,
        metadata={
            'test_type': 'composite',
            'test_name': score_col,
            'components': describe_signal_config(signal_config, role='composite'),
        },
        **backtest_options,
    )
    return {
        'screen': scored_screen.copy(),
        'composition': backtest_result['composition'].copy(),
        'raw_variables': copy.deepcopy(backtest_result['raw_variables']),
        'raw_variable_weights': copy.deepcopy(backtest_result['raw_variable_weights']),
        'backtest': backtest_result,
    }


def _iter_backtest_results(results, path=()):
    """Parcourt récursivement les différentes structures de résultats du module."""
    if not isinstance(results, dict):
        return
    if 'backtest' in results and isinstance(results['backtest'], dict):
        yield ' / '.join(path), results['backtest']
        return
    if 'figure' in results and ('top_builder' in results or 'metadata' in results):
        yield ' / '.join(path), results
        return
    for name, value in results.items():
        if isinstance(value, dict):
            yield from _iter_backtest_results(value, path + (str(name),))


def _component_recipe(components):
    """Produit une description compacte et lisible de la recette d'un test."""
    recipe = []
    for component in components:
        direction = component.get('higher_is_better')
        direction_label = 'higher' if direction is True else 'lower' if direction is False else 'raw'
        recipe.append(
            f"{component.get('role')}:{component.get('raw_variable')}"
            f"[{component.get('dimension')}]x{component.get('weight')}"
            f"({direction_label})"
        )
    return ' + '.join(recipe)


def _composite_display_name(metadata, raw_variables):
    """Décrit un composite avec ses variables, dimensions et poids."""
    if metadata.get('test_type') != 'composite' or not raw_variables:
        return None
    components = metadata.get('components', [])
    variable_labels = []
    for raw_variable in raw_variables:
        dimensions = [
            f"{component.get('dimension')}×{component.get('weight')}"
            for component in components
            if component.get('raw_variable') == raw_variable
        ]
        variable_label = str(raw_variable)
        if dimensions:
            variable_label += f'[{", ".join(dimensions)}]'
        variable_labels.append(variable_label)
    return f'composite / {" | ".join(variable_labels)}'


def _period_metric_records(period_metrics):
    """Normalise les métriques par période sous forme de liste de dictionnaires."""
    if isinstance(period_metrics, pd.DataFrame):
        return period_metrics.to_dict(orient='records')
    if isinstance(period_metrics, list):
        return period_metrics
    if isinstance(period_metrics, dict):
        return list(period_metrics.values())
    return []


def compare_backtest_results(results):
    """Crée une table comparable des scores, pénalités et compositions enregistrées."""
    rows = []
    for test_path, result in _iter_backtest_results(results):
        metadata = result.get('metadata', {})
        components = metadata.get('components', [])
        raw_variables = result.get('raw_variables') or list(dict.fromkeys(
            component.get('raw_variable') for component in components
        ))
        raw_variable_weights = (
            result.get('raw_variable_weights') or summarize_component_weights(components)
        )
        source_test_name = metadata.get('test_name', metadata.get('metric'))
        test_name = (
            _composite_display_name(metadata, raw_variables) or source_test_name
        )
        row = {
            'test_path': test_path,
            'test_group': test_path.rsplit(' / ', 1)[0] if ' / ' in test_path else test_path,
            'test_type': metadata.get('test_type'),
            'test_name': test_name,
            'metric': metadata.get('metric'),
            'benchmark': metadata.get('benchmark'),
            'percentile': metadata.get('percentile'),
            'start_date': metadata.get('start_date'),
            'frequency_rebalancing': metadata.get('frequency_rebalancing'),
            'fill_method': metadata.get('fill_method'),
            'robust_score': result.get('robust_score'),
            'top_bench_ratio': result.get('top_bench_ratio'),
            'top_worst_ratio': result.get('top_worst_ratio'),
            'active_max_drawdown': result.get('active_max_drawdown'),
            'tracking_error_annualized': result.get('tracking_error_annualized'),
            'min_rolling_3y_cagr': result.get('min_rolling_3y_cagr'),
            'observation_count': result.get('observation_count'),
            'component_count': len(components),
            'raw_variable_count': len(raw_variables),
            'raw_variables': ' | '.join(str(variable) for variable in raw_variables),
            'raw_variable_weights': json.dumps(
                raw_variable_weights, ensure_ascii=False, sort_keys=False,
            ),
            'recipe': _component_recipe(components),
        }
        row.update({column: result.get(column) for column in CLASSIC_METRIC_COLUMNS})
        for period_row in _period_metric_records(result.get('period_metrics')):
            period_id = re.sub(
                r'[^A-Za-z0-9]+', '_', str(period_row.get('period_id', 'period')),
            ).strip('_').lower()
            prefix = f'period_{period_id or "period"}'
            row[f'{prefix}_label'] = period_row.get('period_label')
            for metric in PERIOD_SUMMARY_METRICS:
                row[f'{prefix}_{metric}'] = period_row.get(metric)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary['robust_rank_global'] = summary['robust_score'].rank(
        ascending=False, method='min', na_option='bottom',
    )
    summary['robust_rank_within_type'] = summary.groupby('test_type')['robust_score'].rank(
        ascending=False, method='min', na_option='bottom',
    )
    for test_group, group_rows in summary.groupby('test_group'):
        baseline_rows = group_rows[group_rows['test_type'] == 'incremental_baseline']
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        candidate_mask = (
            (summary['test_group'] == test_group)
            & (summary['test_type'] == 'incremental_candidate')
        )
        for metric in ('robust_score', 'top_bench_ratio', 'top_worst_ratio'):
            summary.loc[candidate_mask, f'{metric}_delta_vs_baseline'] = (
                summary.loc[candidate_mask, metric] - baseline[metric]
            )
    return summary.sort_values(['test_type', 'robust_rank_within_type', 'test_name'])


def _safe_filename(value):
    """Transforme un nom de test en nom de fichier portable."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    return cleaned or 'backtest'


def _write_tabular(value, path):
    """Écrit une série ou une table sans imposer de format aux autres objets."""
    if isinstance(value, pd.Series):
        value.to_frame().to_csv(path, index=True)
    elif isinstance(value, pd.DataFrame):
        value.to_csv(path, index=True)


def _read_performance_csv(path):
    """Recharge une courbe de performance exportée et restaure son index temporel."""
    performance = pd.read_csv(path)
    if performance.empty:
        raise ValueError(f'Fichier de performance vide : {path}')
    date_column = 'Date' if 'Date' in performance.columns else performance.columns[0]
    performance[date_column] = pd.to_datetime(performance[date_column], errors='coerce')
    performance = performance.loc[performance[date_column].notna()].set_index(date_column)
    performance.index.name = 'Date'
    for column in performance.columns:
        performance[column] = pd.to_numeric(performance[column], errors='coerce')
    return performance.sort_index()


def _load_saved_performances(export_dir):
    """Recharge les performances locales en utilisant le registre quand il existe."""
    export_dir = Path(export_dir)
    data_dir = export_dir / 'data'
    registry_path = export_dir / 'backtest_registry.json'
    sources = {}
    registered_files = set()

    if registry_path.exists():
        with registry_path.open('r', encoding='utf-8') as registry_file:
            registry = json.load(registry_file)
        for entry in registry:
            test_path = entry.get('test_path')
            metadata = entry.get('metadata', {})
            components = metadata.get('components', [])
            test_name = metadata.get('test_name') or metadata.get('metric') or test_path
            relative_path = entry.get('files', {}).get('performance')
            if relative_path:
                performance_path = export_dir / relative_path
            else:
                file_stem = _safe_filename(f'{test_path}_{test_name}')
                performance_path = data_dir / f'{file_stem}_performance.csv'
            if not test_path or not performance_path.exists():
                continue
            resolved_path = performance_path.resolve()
            registered_files.add(resolved_path)
            sources[test_path] = {
                'test_name': test_name,
                'metadata': metadata,
                'raw_variables': entry.get('raw_variables') or list(dict.fromkeys(
                    component.get('raw_variable') for component in components
                )),
                'raw_variable_weights': (
                    entry.get('raw_variable_weights')
                    or summarize_component_weights(components)
                ),
                'performance': _read_performance_csv(performance_path),
                'origin': str(performance_path),
            }

    if data_dir.exists():
        for performance_path in sorted(data_dir.glob('*_performance.csv')):
            if performance_path.resolve() in registered_files:
                continue
            test_path = performance_path.stem.removesuffix('_performance')
            sources[test_path] = {
                'test_name': test_path,
                'metadata': {},
                'raw_variables': [],
                'raw_variable_weights': {},
                'performance': _read_performance_csv(performance_path),
                'origin': str(performance_path),
            }
    return sources


def _collect_performance_sources(results=None, export_dir=None):
    """Réunit les performances du disque et de la mémoire dans un registre unique."""
    sources = _load_saved_performances(export_dir) if export_dir is not None else {}
    if results is not None:
        for test_path, result in _iter_backtest_results(results):
            performance = result.get('performance')
            if not isinstance(performance, pd.DataFrame) or performance.empty:
                continue
            metadata = result.get('metadata', {})
            components = metadata.get('components', [])
            sources[test_path] = {
                'test_name': metadata.get('test_name') or metadata.get('metric') or test_path,
                'metadata': metadata,
                'raw_variables': result.get('raw_variables') or list(dict.fromkeys(
                    component.get('raw_variable') for component in components
                )),
                'raw_variable_weights': (
                    result.get('raw_variable_weights')
                    or summarize_component_weights(components)
                ),
                'performance': performance.copy(),
                'origin': 'mémoire',
            }
    if not sources:
        raise ValueError('Aucune performance disponible en mémoire ou dans le dossier exporté.')
    return sources


def _resolve_performance_source(identifier, sources):
    """Résout un chemin de test exact ou un nom de test non ambigu."""
    if identifier in sources:
        return identifier, sources[identifier]
    matches = [
        (test_path, source)
        for test_path, source in sources.items()
        if source.get('test_name') == identifier
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = ', '.join(test_path for test_path, _ in matches)
        raise ValueError(
            f'Nom de test ambigu « {identifier} ». Utilisez un chemin parmi : {paths}'
        )
    available = ', '.join(sorted(sources))
    raise KeyError(f'Test introuvable « {identifier} ». Tests disponibles : {available}')


def _performance_display_path(test_path, source):
    """Décrit un composite avec ses variables, dimensions et poids."""
    metadata = source.get('metadata', {})
    return (
        _composite_display_name(metadata, source.get('raw_variables', []))
        or test_path
    )


def _performance_composition_table(selected_sources):
    """Construit la composition détaillée des tests présents dans la comparaison."""
    rows = []
    seen_paths = set()
    for test_path, source in selected_sources:
        if test_path in seen_paths:
            continue
        seen_paths.add(test_path)
        metadata = source.get('metadata', {})
        components = metadata.get('components', [])
        weight_summary = (
            source.get('raw_variable_weights')
            or summarize_component_weights(components)
        )
        for component in components:
            raw_variable = component.get('raw_variable')
            variable_summary = weight_summary.get(raw_variable, {})
            rows.append({
                'display_path': _performance_display_path(test_path, source),
                'test_path': test_path,
                'test_type': metadata.get('test_type'),
                'test_name': source.get('test_name'),
                **component,
                'raw_variable_total_weight': variable_summary.get('total_weight'),
                'raw_variable_absolute_weight': variable_summary.get('absolute_weight'),
                'raw_variable_absolute_weight_share': variable_summary.get(
                    'absolute_weight_share'
                ),
            })
    return pd.DataFrame(rows)


def combine_backtest_performances(results=None, export_dir=None, selections=None,
                                  portfolios=('Top',), save_path=None,
                                  return_composition=False):
    """Combine des performances en mémoire et complète les absences depuis le disque.

    ``selections`` associe le nom final d'une colonne à un couple
    ``(chemin_ou_nom_du_test, portefeuille)``. Sans sélection, tous les tests
    disponibles sont combinés pour les portefeuilles demandés. Pour un composite,
    le libellé automatique contient directement toutes ses variables brutes.
    Avec ``return_composition=True``, la fonction renvoie aussi la table détaillée
    des dimensions, directions, dénominateurs et poids.
    """
    sources = _collect_performance_sources(results=results, export_dir=export_dir)

    series = {}
    selected_sources = []
    if selections is None:
        for test_path, source in sources.items():
            display_path = _performance_display_path(test_path, source)
            for portfolio in portfolios:
                if portfolio in source['performance'].columns:
                    label = f'{display_path} | {portfolio}'
                    if label in series:
                        label = f'{label} [{source.get("test_name")}]'
                    series[label] = source['performance'][portfolio]
                    selected_sources.append((test_path, source))
    else:
        for label, selection in selections.items():
            if isinstance(selection, str):
                identifier, portfolio = selection, 'Top'
            else:
                try:
                    identifier, portfolio = selection
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f'Sélection invalide pour « {label} » : utilisez (test, portefeuille).'
                    ) from error
            test_path, source = _resolve_performance_source(identifier, sources)
            if portfolio not in source['performance'].columns:
                raise KeyError(
                    f'Portefeuille « {portfolio} » absent pour le test « {identifier} ». '
                    f'Colonnes disponibles : {", ".join(source["performance"].columns)}'
                )
            series[label] = source['performance'][portfolio]
            selected_sources.append((test_path, source))

    if not series:
        raise ValueError('Aucune série ne correspond aux portefeuilles demandés.')
    combined = pd.concat(series, axis=1).sort_index()
    combined.index.name = 'Date'
    composition = _performance_composition_table(selected_sources)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=True)
        print(f'Comparaison des performances exportée : {save_path}')
    if return_composition:
        return combined, composition
    return combined


def build_performance_comparison_definitions(results=None, export_dir=None):
    """Crée automatiquement les sélections et ratios de tous les tests disponibles."""
    sources = _collect_performance_sources(results=results, export_dir=export_dir)
    selections = {}
    labels_by_test = {}
    benchmark_selection = None

    for test_path, source in sources.items():
        performance = source['performance']
        display_path = _performance_display_path(test_path, source)
        test_labels = {}
        for portfolio in ('Top', 'Worst'):
            if portfolio not in performance.columns:
                continue
            label = f'{display_path} | {portfolio}'
            if label in selections:
                label = f'{label} [{source.get("test_name")}]'
            selections[label] = (test_path, portfolio)
            test_labels[portfolio] = label
        labels_by_test[test_path] = test_labels
        if benchmark_selection is None and 'Bench' in performance.columns:
            benchmark_selection = (test_path, 'Bench')

    if benchmark_selection is None:
        raise KeyError('Aucune performance Benchmark n’est disponible dans les tests.')
    selections['Benchmark'] = benchmark_selection

    ratio_definitions = {}
    for test_path, labels in labels_by_test.items():
        for portfolio in ('Top', 'Worst'):
            label = labels.get(portfolio)
            if label:
                ratio_definitions[f'{label} / Benchmark'] = (label, 'Benchmark')
        if {'Top', 'Worst'}.issubset(labels):
            top_label = labels['Top']
            worst_label = labels['Worst']
            ratio_definitions[f'{top_label} / Worst'] = (top_label, worst_label)
    return selections, ratio_definitions


def prepare_performance_comparison(results=None, export_dir=None, save_path=None):
    """Prépare toutes les performances, compositions et ratios sans construire de figure."""
    selections, ratio_definitions = build_performance_comparison_definitions(
        results=results, export_dir=export_dir,
    )
    performance, composition = combine_backtest_performances(
        results=results,
        export_dir=export_dir,
        selections=selections,
        save_path=save_path,
        return_composition=True,
    )
    ratios = calculate_performance_ratios(
        performance,
        benchmark_column='Benchmark',
        ratio_definitions=ratio_definitions,
    )
    return {
        'performance': performance,
        'ratios': ratios,
        'composition': composition,
        'performance_selection': selections,
        'ratio_definitions': ratio_definitions,
    }


def calculate_performance_ratios(performance, benchmark_column='Benchmark',
                                 ratio_definitions=None):
    """Calcule les ratios demandés sans construire de figure."""
    if not isinstance(performance, pd.DataFrame) or performance.empty:
        raise ValueError('La table de performances doit être un DataFrame non vide.')
    if ratio_definitions is None:
        if benchmark_column not in performance.columns:
            raise KeyError(
                f'Benchmark « {benchmark_column} » absent. '
                f'Colonnes disponibles : {", ".join(performance.columns)}'
            )
        comparison_columns = [
            column for column in performance.columns if column != benchmark_column
        ]
        if not comparison_columns:
            raise ValueError('Ajoutez au moins une performance à comparer au benchmark.')
        benchmark = performance[benchmark_column].replace(0, np.nan)
        ratios = performance[comparison_columns].div(benchmark, axis=0)
    else:
        if not ratio_definitions:
            raise ValueError('Ajoutez au moins une définition de ratio.')
        ratio_series = {}
        for label, definition in ratio_definitions.items():
            try:
                numerator, denominator = definition
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'Définition invalide pour « {label} » : utilisez '
                    '(numérateur, dénominateur).'
                ) from error
            missing_columns = [
                column for column in (numerator, denominator)
                if column not in performance.columns
            ]
            if missing_columns:
                raise KeyError(
                    f'Colonnes absentes pour le ratio « {label} » : '
                    f'{missing_columns}'
                )
            ratio_series[label] = (
                performance[numerator]
                / performance[denominator].replace(0, np.nan)
            )
        ratios = pd.DataFrame(ratio_series, index=performance.index)
    ratios.index.name = performance.index.name
    return ratios


def _build_analysis_tables_from_results(results):
    """Construit en mémoire les mêmes tables d'analyse que l'export CSV."""
    summary = compare_backtest_results(results)
    composition_rows = []
    classic_metric_rows = []
    period_metric_rows = []

    for test_path, result in _iter_backtest_results(results):
        metadata = result.get('metadata', {})
        components = metadata.get('components', [])
        raw_variables = result.get('raw_variables') or list(dict.fromkeys(
            component.get('raw_variable') for component in components
        ))
        source_test_name = metadata.get('test_name') or metadata.get('metric') or test_path
        test_name = (
            _composite_display_name(metadata, raw_variables) or source_test_name
        )
        weight_summary = summarize_component_weights(components)
        for component in components:
            variable_summary = weight_summary.get(component.get('raw_variable'), {})
            composition_rows.append({
                'test_path': test_path,
                'test_name': test_name,
                **component,
                'raw_variable_total_weight': variable_summary.get('total_weight'),
                'raw_variable_absolute_weight': variable_summary.get('absolute_weight'),
                'raw_variable_absolute_weight_share': variable_summary.get(
                    'absolute_weight_share'
                ),
            })
        for portfolio, metrics in result.get('classic_metrics', {}).items():
            for metric, value in metrics.items():
                classic_metric_rows.append({
                    'test_path': test_path,
                    'test_name': test_name,
                    'portfolio': portfolio,
                    'metric': metric,
                    'value': value,
                })
        for period_row in _period_metric_records(result.get('period_metrics')):
            period_metric_rows.append({
                'test_path': test_path,
                'test_name': test_name,
                'test_type': metadata.get('test_type'),
                'metric': metadata.get('metric'),
                **period_row,
            })

    period_metrics = pd.DataFrame(period_metric_rows)
    if not period_metrics.empty:
        for metric in ('active_cagr', 'top_worst_cagr', 'top_sharpe_ratio'):
            if metric not in period_metrics.columns:
                continue
            period_metrics[f'{metric}_rank_global'] = period_metrics.groupby(
                'period_id', dropna=False,
            )[metric].rank(ascending=False, method='min', na_option='bottom')
            period_metrics[f'{metric}_rank_within_type'] = period_metrics.groupby(
                ['period_id', 'test_type'], dropna=False,
            )[metric].rank(ascending=False, method='min', na_option='bottom')
    return {
        'summary': summary,
        'classic_metrics': pd.DataFrame(classic_metric_rows),
        'period_metrics': period_metrics,
        'signal_composition': pd.DataFrame(composition_rows),
    }


def _combine_total_and_period_metrics(summary, period_metrics):
    """Réunit la période totale et les sous-périodes dans une table comparable."""
    total_rows = []

    def relative_cagr(portfolio_return, reference_return):
        if pd.isna(portfolio_return) or pd.isna(reference_return):
            return float('nan')
        if 1 + reference_return <= 0:
            return float('nan')
        return (1 + portfolio_return) / (1 + reference_return) - 1

    for _, summary_row in summary.iterrows():
        observation_count = summary_row.get('observation_count')
        years = (
            max((observation_count - 1) / 252, 0)
            if pd.notna(observation_count) else float('nan')
        )
        total_row = {
            'scope': 'total',
            'test_path': summary_row.get('test_path'),
            'test_name': summary_row.get('test_name'),
            'test_type': summary_row.get('test_type'),
            'metric': summary_row.get('metric'),
            'period_id': 'total',
            'period_label': 'Période totale',
            'requested_start_date': summary_row.get('start_date'),
            'requested_end_date': None,
            'actual_start_date': None,
            'actual_end_date': None,
            'observation_count': observation_count,
            'years': years,
            'top_cagr': summary_row.get('top_annualized_return'),
            'worst_cagr': summary_row.get('worst_annualized_return'),
            'bench_cagr': summary_row.get('bench_annualized_return'),
            'active_cagr': relative_cagr(
                summary_row.get('top_annualized_return'),
                summary_row.get('bench_annualized_return'),
            ),
            'top_worst_cagr': relative_cagr(
                summary_row.get('top_annualized_return'),
                summary_row.get('worst_annualized_return'),
            ),
            'robust_score': summary_row.get('robust_score'),
            'top_bench_ratio': summary_row.get('top_bench_ratio'),
            'top_worst_ratio': summary_row.get('top_worst_ratio'),
            'active_max_drawdown': summary_row.get('active_max_drawdown'),
            'tracking_error_annualized': summary_row.get('tracking_error_annualized'),
            'min_rolling_3y_cagr': summary_row.get('min_rolling_3y_cagr'),
            'raw_variables': summary_row.get('raw_variables'),
            'raw_variable_weights': summary_row.get('raw_variable_weights'),
            'recipe': summary_row.get('recipe'),
        }
        total_row.update({
            column: summary_row.get(column) for column in CLASSIC_METRIC_COLUMNS
        })
        total_rows.append(total_row)

    total_metrics = pd.DataFrame(total_rows)
    subperiod_metrics = period_metrics.copy()
    if not subperiod_metrics.empty:
        subperiod_metrics.insert(0, 'scope', 'subperiod')
    combined = pd.concat(
        [total_metrics, subperiod_metrics],
        ignore_index=True,
        sort=False,
    )
    if combined.empty:
        return combined
    for metric in ('active_cagr', 'top_worst_cagr', 'top_sharpe_ratio'):
        if metric not in combined.columns:
            continue
        combined[f'{metric}_rank_global'] = combined.groupby(
            'period_id', dropna=False,
        )[metric].rank(ascending=False, method='min', na_option='bottom')
        combined[f'{metric}_rank_within_type'] = combined.groupby(
            ['period_id', 'test_type'], dropna=False,
        )[metric].rank(ascending=False, method='min', na_option='bottom')
    scope_order = combined['scope'].map({'total': 0, 'subperiod': 1}).fillna(2)
    return combined.assign(_scope_order=scope_order).sort_values(
        ['_scope_order', 'period_id', 'test_type', 'test_name'],
    ).drop(columns='_scope_order').reset_index(drop=True)


def reconstruct_backtest_analysis(results=None, export_dir=None, selections=None,
                                  portfolios=('Top',), performance_save_path=None):
    """Restaure performances, scores et métriques totales ou par période."""
    performance, performance_composition = combine_backtest_performances(
        results=results,
        export_dir=export_dir,
        selections=selections,
        portfolios=portfolios,
        save_path=performance_save_path,
        return_composition=True,
    )

    if results is not None:
        tables = _build_analysis_tables_from_results(results)
        source = 'mémoire'
    else:
        if export_dir is None:
            raise ValueError('Indiquez export_dir lorsque les résultats mémoire sont absents.')
        export_dir = Path(export_dir)
        table_files = {
            'summary': 'backtest_summary.csv',
            'classic_metrics': 'classic_metrics.csv',
            'period_metrics': 'period_metrics.csv',
            'signal_composition': 'signal_composition.csv',
        }
        missing_files = [
            filename for filename in table_files.values()
            if not (export_dir / filename).exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                f'Tables exportées absentes dans {export_dir} : {missing_files}'
            )
        tables = {
            key: pd.read_csv(export_dir / filename)
            for key, filename in table_files.items()
        }
        source = 'disque'

    composite_names = (
        performance_composition.loc[
            performance_composition['test_type'].eq('composite'),
            ['test_path', 'display_path'],
        ]
        .drop_duplicates('test_path')
        .set_index('test_path')['display_path']
    )
    for table_name, table in tables.items():
        if table.empty or not {'test_path', 'test_name'}.issubset(table.columns):
            continue
        display_names = table['test_path'].map(composite_names)
        if display_names.notna().any():
            table = table.copy()
            table.loc[display_names.notna(), 'test_name'] = display_names.dropna()
            tables[table_name] = table

    metrics_by_period = _combine_total_and_period_metrics(
        tables['summary'], tables['period_metrics'],
    )
    return {
        'source': source,
        'performance': performance,
        'performance_composition': performance_composition,
        'summary': tables['summary'],
        'total_metrics': tables['summary'],
        'classic_metrics': tables['classic_metrics'],
        'period_metrics': tables['period_metrics'],
        'metrics_by_period': metrics_by_period,
        'signal_composition': tables['signal_composition'],
    }


def _rebase_frame(frame, base_value):
    """Rebase chaque série sur sa première observation valide et non nulle."""
    rebased = frame.copy()
    for column in rebased.columns:
        valid = rebased[column].dropna()
        valid = valid[valid.ne(0)]
        if not valid.empty:
            rebased[column] = rebased[column] / valid.iloc[0] * base_value
    return rebased


def plot_performance_comparison(performance, ratios, benchmark_column='Benchmark',
                                title='Comparaison des performances',
                                save_path=None, show_plot=True, rebase=True):
    """Trace en Plotly des performances et ratios déjà calculés."""
    if benchmark_column not in performance.columns:
        raise KeyError(f'Benchmark « {benchmark_column} » absent des performances.')
    if rebase:
        performance = _rebase_frame(performance, base_value=100.0)
        ratios = _rebase_frame(ratios, base_value=1.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'Performance cumulée',
            'Ratios relatifs',
        ),
    )
    non_benchmark_columns = [
        column for column in performance.columns if column != benchmark_column
    ]
    color_map = {
        column: qualitative.Plotly[index % len(qualitative.Plotly)]
        for index, column in enumerate(non_benchmark_columns)
    }
    color_map[benchmark_column] = 'black'

    for column in performance.columns:
        fig.add_trace(
            go.Scatter(
                x=performance.index,
                y=performance[column],
                mode='lines',
                name=column,
                legendgroup=column,
                line=dict(
                    color=color_map[column],
                    width=3 if column == benchmark_column else 2,
                ),
            ),
            row=1,
            col=1,
        )

    for index, column in enumerate(ratios.columns):
        is_legacy_benchmark_ratio = column in performance.columns
        ratio_name = (
            f'{column} / {benchmark_column}'
            if is_legacy_benchmark_ratio else column
        )
        numerator = ratio_name.split(' / ', 1)[0]
        fig.add_trace(
            go.Scatter(
                x=ratios.index,
                y=ratios[column],
                mode='lines',
                name=ratio_name,
                legendgroup=ratio_name,
                line=dict(
                    color=color_map.get(
                        numerator,
                        qualitative.Plotly[index % len(qualitative.Plotly)],
                    ),
                    width=2,
                    dash=(
                        'solid'
                        if ratio_name.endswith(f' / {benchmark_column}')
                        else 'dash'
                    ),
                ),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='grey',
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        width=1400,
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='v',
            x=1.01,
            xanchor='left',
            y=1,
            yanchor='top',
        ),
        margin=dict(r=420, t=100),
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
    if show_plot:
        fig.show()
    return fig


def export_backtest_results(results, output_dir, export_name=None, export_png=True,
                            export_html=True):
    """Exporte toutes les données avant les figures HTML et PNG optionnelles."""
    export_name = export_name or datetime.now().strftime('backtest_export_%Y%m%d_%H%M%S')
    export_dir = Path(output_dir) / _safe_filename(export_name)
    figures_dir = export_dir / 'figures'
    data_dir = export_dir / 'data'
    holdings_dir = export_dir / 'holdings'
    for directory in (figures_dir, data_dir, holdings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    flattened = list(_iter_backtest_results(results))
    summary = compare_backtest_results(results)
    composition_rows = []
    classic_metric_rows = []
    period_metric_rows = []
    registry = []
    figure_jobs = []
    png_enabled = export_png

    for test_path, result in flattened:
        metadata = copy.deepcopy(result.get('metadata', {}))
        source_test_name = metadata.get('test_name') or metadata.get('metric') or test_path
        components = metadata.get('components', [])
        raw_variables = result.get('raw_variables') or list(dict.fromkeys(
            component.get('raw_variable') for component in components
        ))
        test_name = (
            _composite_display_name(metadata, raw_variables) or source_test_name
        )
        file_stem = _safe_filename(f'{test_path}_{source_test_name}')
        performance_path = data_dir / f'{file_stem}_performance.csv'
        ratios_path = data_dir / f'{file_stem}_ratios.csv'
        top_holdings_path = holdings_dir / f'{file_stem}_top.csv'
        worst_holdings_path = holdings_dir / f'{file_stem}_worst.csv'
        html_path = figures_dir / f'{file_stem}.html'
        png_path = figures_dir / f'{file_stem}.png'

        _write_tabular(result.get('performance'), performance_path)
        _write_tabular(result.get('ratios'), ratios_path)
        _write_tabular(result.get('top_holdings'), top_holdings_path)
        _write_tabular(result.get('worst_holdings'), worst_holdings_path)

        figure = result.get('figure')
        if figure is not None:
            figure_jobs.append((figure, html_path, png_path))

        weight_summary = summarize_component_weights(components)
        for component in components:
            variable_summary = weight_summary.get(component.get('raw_variable'), {})
            composition_rows.append({
                'test_path': test_path,
                'test_name': test_name,
                **component,
                'raw_variable_total_weight': variable_summary.get('total_weight'),
                'raw_variable_absolute_weight': variable_summary.get('absolute_weight'),
                'raw_variable_absolute_weight_share': variable_summary.get(
                    'absolute_weight_share'
                ),
            })

        for portfolio, metrics in result.get('classic_metrics', {}).items():
            for metric, value in metrics.items():
                classic_metric_rows.append({
                    'test_path': test_path,
                    'test_name': test_name,
                    'portfolio': portfolio,
                    'metric': metric,
                    'value': value,
                })

        for period_row in _period_metric_records(result.get('period_metrics')):
            period_metric_rows.append({
                'test_path': test_path,
                'test_name': test_name,
                'test_type': metadata.get('test_type'),
                'metric': metadata.get('metric'),
                **period_row,
            })

        registry.append({
            'test_path': test_path,
            'metadata': metadata,
            'metrics': {
                key: result.get(key) for key in (
                    'robust_score', 'top_bench_ratio', 'top_worst_ratio',
                    'active_max_drawdown', 'tracking_error_annualized',
                    'min_rolling_3y_cagr', 'observation_count',
                )
            },
            'raw_variables': result.get('raw_variables', []),
            'raw_variable_weights': result.get('raw_variable_weights', {}),
            'classic_metrics': result.get('classic_metrics', {}),
            'period_metrics': _period_metric_records(result.get('period_metrics')),
            'files': {
                'performance': performance_path.relative_to(export_dir).as_posix(),
                'ratios': ratios_path.relative_to(export_dir).as_posix(),
                'top_holdings': top_holdings_path.relative_to(export_dir).as_posix(),
                'worst_holdings': worst_holdings_path.relative_to(export_dir).as_posix(),
                'html': html_path.relative_to(export_dir).as_posix() if export_html and figure is not None else None,
                'png': png_path.relative_to(export_dir).as_posix() if export_png and figure is not None else None,
            },
        })

    composition = pd.DataFrame(composition_rows)
    classic_metrics = pd.DataFrame(classic_metric_rows)
    period_metrics = pd.DataFrame(period_metric_rows)
    if not period_metrics.empty:
        ranking_metrics = ('active_cagr', 'top_worst_cagr', 'top_sharpe_ratio')
        for metric in ranking_metrics:
            if metric not in period_metrics.columns:
                continue
            period_metrics[f'{metric}_rank_global'] = period_metrics.groupby(
                'period_id', dropna=False,
            )[metric].rank(ascending=False, method='min', na_option='bottom')
            period_metrics[f'{metric}_rank_within_type'] = period_metrics.groupby(
                ['period_id', 'test_type'], dropna=False,
            )[metric].rank(ascending=False, method='min', na_option='bottom')
    metrics_by_period = _combine_total_and_period_metrics(summary, period_metrics)
    summary.to_csv(export_dir / 'backtest_summary.csv', index=False)
    composition.to_csv(export_dir / 'signal_composition.csv', index=False)
    classic_metrics.to_csv(export_dir / 'classic_metrics.csv', index=False)
    period_metrics.to_csv(export_dir / 'period_metrics.csv', index=False)
    metrics_by_period.to_csv(export_dir / 'metrics_by_period.csv', index=False)
    with (export_dir / 'backtest_registry.json').open('w', encoding='utf-8') as registry_file:
        json.dump(registry, registry_file, ensure_ascii=False, indent=2, default=str)

    print(f'Données exportées avant les figures : {export_dir}')
    for figure, html_path, png_path in figure_jobs:
        if export_html:
            figure.write_html(html_path)
        if png_enabled:
            try:
                figure.write_image(png_path)
            except Exception as error:
                png_enabled = False
                print(f'Avertissement : export PNG indisponible ({error}).')

    print(f'Export terminé : {export_dir}')
    return {
        'export_dir': export_dir,
        'summary': summary,
        'composition': composition,
        'classic_metrics': classic_metrics,
        'period_metrics': period_metrics,
        'metrics_by_period': metrics_by_period,
        'registry': registry,
    }
