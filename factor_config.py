"""Catalogue des colonnes de facteurs disponibles dans le screen de référence.

Les listes sont prêtes à être importées dans un notebook. Elles restent séparées
des réglages de backtest : ce fichier décrit les familles et fournit seulement
une configuration large par défaut, modifiable dans une cellule du notebook.
Une dimension est active uniquement lorsque son poids est strictement positif.
"""

from copy import deepcopy


FACTOR_FAMILIES = {
    'growth': [
        '5Y_Hist EPS TrendStab',
        '5Y_Hist GrossInc TrendStab',
        '5Y_Hist Sales TrendStab',
        'EPS Forward Growth CGR3',
        'EPS Growth FY1',
        'EPS Growth NTM',
        'Gross Income Growth FY1',
        'Gross Income Growth NTM',
        'Sales Growth FY1',
        'Sales Growth NTM',
        'Growth Avg Percentile',
        'Growth_Forward_Avg Percentile',
        'Growth_Historical_Avg Percentile',
        'Growth_NTM Avg Percentile',
        'Growth_NTM_Avg Percentile',
        'PCT EPS Growth FY1',
        'PCT EPS Growth NTM',
        'PCT Gross Income Growth FY1',
        'PCT Gross Income Growth NTM',
        'PCT Hist EPS',
        'PCT Hist GrossInc',
        'PCT Hist Sales',
        'PCT Sales FY0',
        'PCT Sales Growth FY1',
        'PCT Sales Growth NTM',
        'Ebit 5Y CAGR',
        'EPS Growth FY1 CIQ',
        'SP Est 5Y EPS Gr CIQ',
        'CFO 5Y CAGR',
        'Revenue 5Y CAGR',
        'Const Earning 5Y CAGR',
        'Sales_5Y_growth',
        'EPS_5Y_growth',
        'EBITDA Growth FY1 CIQ',
        'Gross Profit 5Y CAGR',
        'Sales Growth FY1 CIQ',
        'Ebitda 5Y CAGR',
        'GROWTH_RANK_FS_SECTOR',
        'GROWTH_SCORE_FS_SECTOR',
    ],
    'quality_profitability': [
        'Asset TO exFIN',
        'Combined Ratio FY1',
        'Combined Ratio NTM',
        'Oper Margin',
        'Quality Avg Percentile',
        'Quality_NTM Avg Percentile',
        'ROE avg FY0',
        'ROTE avg FY1',
        'ROTE avg NTM',
        'PCT Asset TO',
        'PCT Assets FY0',
        'PCT CombinedRatio',
        'PCT CombinedRatio NTM',
        'PCT OM FY0',
        'PCT ROE',
        'PCT ROTE',
        'FCF Conversion',
        'Gross Margin',
        'Ebitda Margin',
        'Cont Op Earning Margin',
        'Ebitda to Int expense',
        'Coverage Ratio CIQ',
        'Current Ratio',
        'MARGIN_RANK_FS_SECTOR',
        'MARGIN_SCORE_FS_SECTOR',
    ],
    'value': [
        'EV To EBITDA FY1',
        'EV To EBITDA LTM',
        'EV To EBITDA NTM',
        'EV to Ebit FY1',
        'EV to Sales FY1',
        'EV to Sales LTM',
        'EV to Sales NTM',
        'Earns Yield FY0',
        'Earns Yield FY1',
        'Earns Yield NTM',
        'Book Value Per Share',
        'FE Val Ev_Ebit Mean NTM',
        'PB / PTangibleBook FY1',
        'PB / PTangibleBook LTM',
        'PB / PTangibleBook NTM',
        'PB LTM',
        'PE FY1',
        'PE LTM',
        'PE NTM',
        'PFCF LTM',
        'PTangibleBook LTM',
        'Price to Book FY1',
        'Price to Book NTM',
        'Price to FreeCF FY1',
        'Price to FreeCF NTM',
        'Value Avg Percentile',
        'Value_Forward Avg Percentile',
        'Value_NTM Avg Percentile',
        'Value_NTM Avg Percentile.1',
        'Value_Spot_Avg Percentile',
        'PCT EV to Sales FY1',
        'PCT EV to Sales LTM',
        'PCT EV to Sales NTM',
        'PCT EVEBIT FY1',
        'PCT EVEBIT NTM',
        'PCT EVEBITDA FY1',
        'PCT EVEBITDA LTM',
        'PCT EVEBITDA NTM',
        'PCT PB FY1',
        'PCT PB LTM',
        'PCT PB NTM',
        'PCT PE FY1',
        'PCT PE LTM',
        'PCT PE NTM',
        'PCT PFCF FY1',
        'PCT PFCF LTM',
        'PCT PFCF NTM',
        'Price Cont Op Earning',
        'PER_5Y_Rank',
        'PEG',
        'EV to Ebit FY1 CIQ',
        'PER_10Y_Rank',
        'PE FY1 CIQ',
        'P to CFO',
        'EV to Ebit',
        'VALUE_RANK_FS_SECTOR',
        'VALUE_SCORE_FS_SECTOR',
    ],
    'momentum_revisions': [
        'EPS Med NTM -3M',
        'EPS Med NTM 0',
        'EPS NTM 3M Growth',
        'EPS Revision Ratio',
        'MOM Score',
        'Mom Avg Percentile',
        'PMOM 12M1M',
        'PCT EPSM3M',
        'PCT ERR',
        'PCT MOM 12M1M',
        'PCT MOM Score',
        'Perf5D',
        'Perf1M',
        'Perf3M',
        'Perf6M',
        'SP Price Target CIQ',
        'SP Price Close CIQ',
        'EPS Estimates FY1',
        'Pct_Short_Interest',
        'MOMENTUM_RANK_FS_SECTOR',
        'MOMENTUM_SCORE_FS_SECTOR',
    ],
    'low_volatility_risk': [
        'Daily Vol 260J',
        'Daily Vol 60J',
        'Daily Vol 90J',
        'LowVol Avg Percentile',
        'PCT DVol 260J',
        'PCT DVol 60J',
        'PCT DVol 90J',
        'Volatilite Rolling ewma 250D',
        'Maximum Drawdown Rolling 250D',
        'VaR 1% Rolling 250D',
        'Beta vs SXXP (Rolling ewma 250D)',
        'Beta vs Regional Benchmark (Rolling ewma 250D)',
        'Beta Up vs SXXP (252D)',
        'Beta Down vs SXXP (252D)',
        'LOW_VOL_RANK_FS_SECTOR',
        'LOW_VOL_SCORE_FS_SECTOR',
    ],
    'dividend': [
        'DPS 1Y Growth FY1',
        'DPS 1Y Growth NTM',
        'DPS FY1',
        'DPS NTM',
        'DVD Payout FY0',
        'DVD Yield FY0',
        'DVD Yield FY1',
        'DVD Yield NTM',
        'D_DPS TrendStab',
        'Dividend Avg Percentile',
        'Dividend_NTM Avg Percentile',
        'PCT DPS GR FY1',
        'PCT DPS GR NTM',
        'PCT DvdYield FY1',
        'PCT DvdYield NTM',
        'PCT Payout Ratio',
        'CF total div paid CIQ',
        'FCF Div Cov Ratio',
        'CFO Div Cov Ratio',
        'Repurchase Stock CIQ',
    ],
    'balance_sheet_leverage': [
        'NetDebt to EBITDA exFIN',
        'PCT NBEBITDA',
        'PCT TIER1',
        'Risk adj Assets CIQ',
        'Total Deposit CIQ',
        'shTerm Debt CIQ',
        'Total Asset CIQ',
        'Net Debt',
        'Gross Loans CIQ',
        'Tier1 Ratio CIQ',
        'TIER1 Ratio FY0',
        'Core Tier1 Ratio CIQ',
        'Total Debt',
        'Current Liabilities CIQ',
        'Prov for Loan Losses CIQ',
        'Total Equity',
        'Total Capital Ratio CIQ',
        'Non perf Loans to Total Loans CIQ',
        'Net Debt to Market Cap',
        'Net Debt to Ebit',
        'Net PropPlantEquipm CIQ',
        'Total Cash and Equiv',
        'Current Assets CIQ',
        'Non Perf Assets CIQ',
        'Net Debt to Tot Equity',
        'net WorkCapital CIQ',
        'Goodwill CIQ',
        'Non Perf loan CIQ',
        'netD to EBITDA FY1',
        'LEVERAGE_RANK_FS_SECTOR',
        'LEVERAGE_SCORE_FS_SECTOR',
    ],
    'cash_flow_investment': [
        'R&D Expense CIQ',
        'Goodwill Impairment CIQ',
        'CF from Investing',
        'Capex CIQ',
        'Depreciation and Amort CIQ',
        'FCF',
        'CFO',
        'change Net WorkCapital CIQ',
    ],
    'size': [
        'Size Avg Percentile',
        'PCT Mkt Value',
        'Benchmark Market Value Millions in EUR',
        'Total Employees CIQ',
    ],
    'esg_carbon': [
        'ESG_G',
        'CARBON_IMPACT_SCORE',
        'ESG_S',
        'CarbonIntensity_Sales',
        'Decile_CarbIntensity',
        'ESG_E',
        'CarbonIntensity_EV',
        'ESG_ANALYST_SCORE',
    ],
    'multifactor_models': [
        'Multi Avg Percentile',
        'Score ML',
        'FIVE_FACTOR_RANK_FS_SECTOR',
        'FIVE_FACTOR_SCORE_FS_SECTOR',
        'RECO_TOP_FLAG_FS_SECTOR',
        'RECO_WORST_FLAG_FS_SECTOR',
        'RECO_SCORE_FS_SECTOR',
        'MACRO_SIGNAL_AVG_FS_SECTOR',
        'RATE_SIGNAL_FS_SECTOR',
        'Score ML_IF',
    ],
}


LOWER_IS_BETTER = {
    'EV To EBITDA FY1', 'EV To EBITDA LTM', 'EV To EBITDA NTM',
    'EV to Ebit FY1', 'EV to Sales FY1', 'EV to Sales LTM',
    'EV to Sales NTM', 'FE Val Ev_Ebit Mean NTM',
    'PB / PTangibleBook FY1', 'PB / PTangibleBook LTM',
    'PB / PTangibleBook NTM', 'PB LTM', 'PE FY1', 'PE LTM', 'PE NTM',
    'PFCF LTM', 'PTangibleBook LTM', 'Price to Book FY1',
    'Price to Book NTM', 'Price to FreeCF FY1', 'Price to FreeCF NTM',
    'Price Cont Op Earning', 'PEG', 'EV to Ebit FY1 CIQ', 'PE FY1 CIQ',
    'P to CFO', 'EV to Ebit', 'Daily Vol 260J', 'Daily Vol 60J',
    'PCT EV to Sales FY1', 'PCT EV to Sales LTM', 'PCT EV to Sales NTM',
    'PCT EVEBIT FY1', 'PCT EVEBIT NTM', 'PCT EVEBITDA FY1',
    'PCT EVEBITDA LTM', 'PCT EVEBITDA NTM', 'PCT PB FY1', 'PCT PB LTM',
    'PCT PB NTM', 'PCT PE FY1', 'PCT PE LTM', 'PCT PE NTM',
    'PCT PFCF FY1', 'PCT PFCF LTM', 'PCT PFCF NTM',
    'Daily Vol 90J', 'Volatilite Rolling ewma 250D',
    'PCT DVol 260J', 'PCT DVol 60J', 'PCT DVol 90J',
    'Maximum Drawdown Rolling 250D', 'VaR 1% Rolling 250D',
    'Beta vs SXXP (Rolling ewma 250D)',
    'Beta vs Regional Benchmark (Rolling ewma 250D)',
    'Beta Up vs SXXP (252D)', 'Beta Down vs SXXP (252D)',
    'NetDebt to EBITDA exFIN', 'shTerm Debt CIQ', 'Net Debt', 'Total Debt',
    'PCT NBEBITDA', 'Combined Ratio FY1', 'Combined Ratio NTM',
    'PCT CombinedRatio', 'PCT CombinedRatio NTM',
    'Prov for Loan Losses CIQ', 'Non perf Loans to Total Loans CIQ',
    'Net Debt to Market Cap', 'Net Debt to Ebit', 'Non Perf Assets CIQ',
    'Net Debt to Tot Equity', 'Non Perf loan CIQ', 'netD to EBITDA FY1',
    'CarbonIntensity_Sales', 'Decile_CarbIntensity', 'CarbonIntensity_EV',
    'Pct_Short_Interest',
}


# Les horizons comptent les observations successives de chaque société.
COMPARISON_PERIODS = (1, 3, 6, 12)
COMPARISON_BASE_DIMENSIONS = ('pct', 'diff', 'rank_diff')


def make_signal_dimensions(periods=(1,), transformations=(
        'level', 'pct', 'diff', 'rank_diff')):
    """Construit les dimensions unitaires pour les horizons demandés."""
    selected_periods = tuple(dict.fromkeys(periods))
    unknown_periods = set(selected_periods) - set(COMPARISON_PERIODS)
    if unknown_periods:
        raise ValueError(f'Horizons inconnus : {sorted(unknown_periods)}')

    selected_transformations = tuple(dict.fromkeys(transformations))
    allowed_transformations = ('level',) + COMPARISON_BASE_DIMENSIONS
    unknown_transformations = (
        set(selected_transformations) - set(allowed_transformations)
    )
    if unknown_transformations:
        raise ValueError(
            f'Transformations inconnues : {sorted(unknown_transformations)}'
        )

    dimensions = []
    for transformation in selected_transformations:
        if transformation == 'level':
            dimensions.append('level')
            continue
        dimensions.extend(
            f'{transformation}_{period}' for period in selected_periods
        )
    return tuple(dimensions)


COMPARISON_DIMENSIONS = tuple(
    f'{dimension}_{period}'
    for dimension in COMPARISON_BASE_DIMENSIONS
    for period in COMPARISON_PERIODS
)
SIGNAL_DIMENSIONS = ('level',) + COMPARISON_DIMENSIONS
DEFAULT_SIGNAL_DIMENSIONS = make_signal_dimensions()
LEGACY_DIMENSION_ALIASES = {
    dimension: f'{dimension}_1' for dimension in COMPARISON_BASE_DIMENSIONS
}


def factor_columns(*families):
    """Retourne les colonnes uniques des familles demandées."""
    selected_families = families or tuple(FACTOR_FAMILIES)
    unknown = [family for family in selected_families if family not in FACTOR_FAMILIES]
    if unknown:
        raise KeyError(f'Familles inconnues : {unknown}')
    return list(dict.fromkeys(
        column
        for family in selected_families
        for column in FACTOR_FAMILIES[family]
    ))


def signal_options(higher_is_better=True, level=0.0, pct=0.0, diff=0.0,
                   denominator=None, rank_diff=0.0,
                   pct_1=None, pct_3=0.0, pct_6=0.0, pct_12=0.0,
                   diff_1=None, diff_3=0.0, diff_6=0.0, diff_12=0.0,
                   rank_diff_1=None, rank_diff_3=0.0,
                   rank_diff_6=0.0, rank_diff_12=0.0):
    """Décrit les poids par horizon ; les anciens noms désignent une période."""
    comparison_weights = {
        'pct_1': pct if pct_1 is None else pct_1,
        'pct_3': pct_3,
        'pct_6': pct_6,
        'pct_12': pct_12,
        'diff_1': diff if diff_1 is None else diff_1,
        'diff_3': diff_3,
        'diff_6': diff_6,
        'diff_12': diff_12,
        'rank_diff_1': rank_diff if rank_diff_1 is None else rank_diff_1,
        'rank_diff_3': rank_diff_3,
        'rank_diff_6': rank_diff_6,
        'rank_diff_12': rank_diff_12,
    }
    options = {
        'higher_is_better': higher_is_better,
        'weight_level': level,
        **{
            f'weight_{dimension}': comparison_weights[dimension]
            for dimension in COMPARISON_DIMENSIONS
        },
    }
    if denominator is not None:
        options['denominator'] = denominator
    return options


def make_signal_config(*families, variables=None, transformations=('level',)):
    """Construit une configuration large avec des poids unitaires modifiables."""
    if variables is not None and families:
        raise ValueError('Utilisez soit families, soit variables, mais pas les deux.')
    selected_variables = (
        list(variables) if variables is not None else factor_columns(*families)
    )
    canonical_transformations = {
        LEGACY_DIMENSION_ALIASES.get(transformation, transformation)
        for transformation in transformations
    }
    unknown_transformations = canonical_transformations - set(SIGNAL_DIMENSIONS)
    if unknown_transformations:
        raise ValueError(f'Transformations inconnues : {sorted(unknown_transformations)}')

    config = {}
    for variable in selected_variables:
        options = {'higher_is_better': variable not in LOWER_IS_BETTER}
        for transformation in SIGNAL_DIMENSIONS:
            options[f'weight_{transformation}'] = (
                1.0 if transformation in canonical_transformations else 0.0
            )
        config[variable] = options
    return config


ALL_FACTOR_COLUMNS = factor_columns()
RAW_VARIABLES = list(ALL_FACTOR_COLUMNS)
WIDE_CONFIG = make_signal_config(variables=ALL_FACTOR_COLUMNS)


def copy_wide_config():
    """Fournit une copie indépendante avant toute personnalisation dans un notebook."""
    return deepcopy(WIDE_CONFIG)
