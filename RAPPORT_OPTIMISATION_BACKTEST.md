# Rapport d’optimisation de la mémoire des backtests

Date de mesure : 22 juillet 2026

## 1. Objectif

Cette modification vise à rendre les campagnes de tests unitaires plus légères sans modifier les performances financières calculées. Le périmètre est volontairement limité à trois changements :

1. retourner une structure de résultat légère par défaut ;
2. libérer les builders Top et Worst après extraction des données utiles ;
3. supprimer les colonnes temporaires `Unitary_*` après chaque backtest, tout en conservant les variables dérivées `pct_N`, `diff_N` et `rank_diff_N`.

Les algorithmes de sélection des titres, de pondération, de calcul des performances et des métriques ne sont pas modifiés.

## 2. Structure légère des résultats

Par défaut, `run_top_worst_backtest()` conserve toujours :

- les performances Top, Worst et Benchmark ;
- les ratios Top/Benchmark, Worst/Benchmark et Top/Worst ;
- les métriques classiques et les métriques par période ;
- le score de robustesse ;
- les positions historiques Top et Worst ;
- la composition du signal et les poids réellement utilisés ;
- les métadonnées nécessaires aux exports et à la reconstruction locale.

Les objets complets `top_builder` et `worst_builder` ne sont plus ajoutés au dictionnaire de résultat. Ils deviennent donc libérables dès la fin de la fonction.

Pour un diagnostic interne exceptionnel, l’ancien comportement reste disponible :

```python
RUN_OPTIONS = {
    **RUN_OPTIONS,
    "retain_builders": True,
}
```

L’option recommandée pour les campagnes normales est :

```python
RUN_OPTIONS = {
    **RUN_OPTIONS,
    "retain_builders": False,
}
```

## 3. Nettoyage des scores unitaires temporaires

`test_unitary_signals()` utilise une colonne `Unitary_<dimension>_<variable>` pour construire les portefeuilles Top et Worst. Cette colonne normalisée est désormais supprimée immédiatement après le backtest correspondant.

Les colonnes dérivées destinées aux analyses ultérieures restent dans `screen`. Par exemple :

```text
Revenue 5Y CAGR__pct_3
Revenue 5Y CAGR__diff_6
Net Debt to Ebit__rank_diff_12
```

Avec 3 454 342 lignes, une colonne `float64` occupe environ 26,35 Mio. La suppression des scores temporaires évite donc approximativement :

| Campagne | Colonnes `Unitary_*` évitées | Mémoire numérique évitée |
|---|---:|---:|
| 10 variables × 13 dimensions | 130 | 3,35 Gio |
| 12 variables × 13 dimensions | 156 | 4,01 Gio |
| 15 variables × 13 dimensions | 195 | 5,02 Gio |

Ces chiffres concernent uniquement les colonnes temporaires du screen. Les variables dérivées demandées restent volontairement disponibles.

## 4. Protocole de mesure

Les mesures avant et après modification utilisent le même scénario :

- processeur Intel Core i9-13900HX ;
- 64 Go de mémoire physique ;
- screen réel de 3 454 342 lignes ;
- rendements de 1 288 membres du benchmark chargés ;
- benchmark `STOXX EUROPE 600` calculé une fois puis injecté ;
- période de backtest du 1er janvier 2010 à juillet 2026 ;
- variable `Revenue 5Y CAGR` ;
- dimension `level` ;
- portefeuilles Top et Worst ;
- points de rupture 2020, 2022 et 2024 ;
- `show_plot=False` et `build_figure=False`.

La mémoire est mesurée avec le RSS du processus Python. Le gain de mémoire persistante est calculé par différence entre le RSS après le test et le RSS juste avant le test. Cette méthode neutralise les différences de mémoire initiale entre deux processus indépendants.

## 5. Résultats avant et après

| Indicateur | Avant | Après | Gain mesuré |
|---|---:|---:|---:|
| Temps du test complet | 65,398 s | 67,941 s | -3,9 % |
| Pic additionnel pendant le test | 1 693,3 Mio | 1 692,2 Mio | 0,1 % |
| Mémoire persistante après le test | 513,6 Mio | 60,1 Mio | **88,3 %** |
| Colonnes dans le screen retourné | 9 | 8 | 1 colonne temporaire supprimée |

Le temps d’exécution reste statistiquement similaire. La variation de 2,5 secondes entre deux exécutions uniques ne constitue pas une régression démontrée ; cette modification cible la mémoire persistante et non l’algorithme de backtest.

Le pic de mémoire reste également inchangé, car les deux builders sont encore nécessaires simultanément pendant le calcul Top/Worst. En revanche, ils ne restent plus attachés au résultat après le calcul.

La mémoire persistante supplémentaire diminue de 453,5 Mio par test dans ce scénario complet. Une extrapolation linéaire illustre l’ordre de grandeur, sans constituer une prévision exacte :

| Campagne | Rétention de builders potentiellement évitée |
|---|---:|
| 10 variables × 13 dimensions | environ 57,6 Gio |
| 12 variables × 13 dimensions | environ 69,1 Gio |
| 15 variables × 13 dimensions | environ 86,4 Gio |

La consommation réelle d’une longue campagne dépend du réemploi des allocations pandas, de la taille des positions et du nombre de périodes. L’extrapolation montre néanmoins pourquoi la structure précédente pouvait saturer un kernel de 64 Go.

## 6. Contrôle de non-régression financière

Le scénario complet produit les mêmes résultats avant et après :

| Indicateur | Valeur avant | Valeur après |
|---|---:|---:|
| Score de robustesse | -0,7371 | -0,7371 |
| Top/Benchmark | -0,0665 | -0,0665 |
| Top/Worst | 0,0845 | 0,0845 |

La structure légère ne modifie donc pas le résultat financier du test mesuré.

Les tests automatisés vérifient également :

- l’absence des builders par défaut ;
- leur présence lorsque `retain_builders=True` ;
- la suppression des colonnes `Unitary_*` ;
- la conservation des colonnes dérivées ;
- la compatibilité des exports et de la reconstruction des résultats ;
- les directions, horizons, compositions et périodes déjà couverts.

Douze tests automatisés passent après la modification.

## 7. Limites et prochaine étape recommandée

Cette première optimisation réduit fortement la mémoire conservée entre deux tests, mais elle ne réduit presque pas le pic temporaire ni le temps d’un test individuel.

La prochaine étape recommandée est de réduire le coût interne du moteur :

1. éviter les copies profondes répétées de `screen` et `returns` dans `PtfBuilder` ;
2. partager le prétraitement mensuel entre Top et Worst ;
3. vectoriser la neutralisation sectorielle ;
4. éviter de transformer plusieurs fois toute la matrice de rendements en table longue.

Ces changements devront être couverts par des tests d’équivalence sur les performances, les positions et les métriques avant toute mise en production.
