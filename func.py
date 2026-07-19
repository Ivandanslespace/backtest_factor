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

import numpy as np
import pandas as pd

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
                           save_path=None, metadata=None, period_breakpoints=None):
    """Exécute un backtest Top/Worst pour une variable déjà disponible dans screen."""
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    if market_cap_column not in screen.columns:
        source_column = market_cap_column.rstrip()
        if source_column not in screen.columns:
            raise KeyError(f'Colonne requise absente : {market_cap_column}')
        screen[market_cap_column] = screen[source_column]
    resolved_breakpoints = list(
        RECOMMENDED_PERIOD_BREAKPOINTS
        if period_breakpoints is None else period_breakpoints
    )
    builder_top = PtfBuilder(
        screen, returns, ptf_name=f'{metric}_top', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=True,
    )
    builder_worst = PtfBuilder(
        screen, returns, ptf_name=f'{metric}_worst', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=False,
    )

    for builder in (builder_top, builder_worst):
        builder.start_date = pd.Timestamp(DEFAULT_START_DATE)
        builder.freq_rebal = 1
        builder.fill_method = 'copy'

    comparison = builder_top.backtest_plot_top_vs_bottom(
        builder_bottom=builder_worst,
        title=f'Factor Analysis: {metric}',
        save_path=save_path,
        show_plot=show_plot,
        period_breakpoints=resolved_breakpoints,
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


def test_raw_variables(screen, returns, variables, list_noire_path, **backtest_options):
    """Teste directement le Top/Worst de chaque variable brute sans configuration."""
    results = {}
    for variable in variables:
        if variable not in screen.columns:
            print(f'Avertissement : {variable} absent des données. Variable ignorée.')
            continue
        print(f'Test de variable brute : {variable}')
        results[variable] = run_top_worst_backtest(
            screen, returns, variable, list_noire_path,
            metadata={
                'test_type': 'raw',
                'test_name': variable,
                'components': [{
                    'role': 'raw',
                    'raw_variable': variable,
                    'prepared_variable': variable,
                    'derived_variable': variable,
                    'denominator': None,
                    'dimension': 'raw',
                    'higher_is_better': None,
                    'weight': 1.0,
                }],
            },
            **backtest_options,
        )
    return results


def test_unitary_signals(screen, returns, signal_config, list_noire_path,
                         dimensions=('level', 'pct', 'diff'), **backtest_options):
    """Teste séparément les dimensions demandées et les conserve dans screen."""
    results = {}
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
        row = {
            'test_path': test_path,
            'test_group': test_path.rsplit(' / ', 1)[0] if ' / ' in test_path else test_path,
            'test_type': metadata.get('test_type'),
            'test_name': metadata.get('test_name', metadata.get('metric')),
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
                'performance': _read_performance_csv(performance_path),
                'origin': str(performance_path),
            }
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


def combine_backtest_performances(results=None, export_dir=None, selections=None,
                                  portfolios=('Top',), save_path=None):
    """Combine des performances en mémoire et complète les absences depuis le disque.

    ``selections`` associe le nom final d'une colonne à un couple
    ``(chemin_ou_nom_du_test, portefeuille)``. Sans sélection, tous les tests
    disponibles sont combinés pour les portefeuilles demandés.
    """
    sources = _load_saved_performances(export_dir) if export_dir is not None else {}
    if results is not None:
        for test_path, result in _iter_backtest_results(results):
            performance = result.get('performance')
            if not isinstance(performance, pd.DataFrame) or performance.empty:
                continue
            metadata = result.get('metadata', {})
            sources[test_path] = {
                'test_name': metadata.get('test_name') or metadata.get('metric') or test_path,
                'performance': performance.copy(),
                'origin': 'mémoire',
            }
    if not sources:
        raise ValueError('Aucune performance disponible en mémoire ou dans le dossier exporté.')

    series = {}
    if selections is None:
        for test_path, source in sources.items():
            for portfolio in portfolios:
                if portfolio in source['performance'].columns:
                    series[f'{test_path} | {portfolio}'] = source['performance'][portfolio]
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
            _, source = _resolve_performance_source(identifier, sources)
            if portfolio not in source['performance'].columns:
                raise KeyError(
                    f'Portefeuille « {portfolio} » absent pour le test « {identifier} ». '
                    f'Colonnes disponibles : {", ".join(source["performance"].columns)}'
                )
            series[label] = source['performance'][portfolio]

    if not series:
        raise ValueError('Aucune série ne correspond aux portefeuilles demandés.')
    combined = pd.concat(series, axis=1).sort_index()
    combined.index.name = 'Date'
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=True)
        print(f'Comparaison des performances exportée : {save_path}')
    return combined


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
        test_name = metadata.get('test_name') or metadata.get('metric') or test_path
        file_stem = _safe_filename(f'{test_path}_{test_name}')
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

        weight_summary = summarize_component_weights(metadata.get('components', []))
        for component in metadata.get('components', []):
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
    summary.to_csv(export_dir / 'backtest_summary.csv', index=False)
    composition.to_csv(export_dir / 'signal_composition.csv', index=False)
    classic_metrics.to_csv(export_dir / 'classic_metrics.csv', index=False)
    period_metrics.to_csv(export_dir / 'period_metrics.csv', index=False)
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
        'registry': registry,
    }
