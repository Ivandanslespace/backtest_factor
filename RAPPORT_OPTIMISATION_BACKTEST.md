# Rapport d’optimisation du moteur de backtest

Date de mesure : 22 juillet 2026

## 1. Objectif

Cette phase réduit le temps d’exécution et le pic de mémoire d’un backtest complet, sans modifier les positions ni les résultats financiers. Elle complète la première phase, qui avait déjà rendu les résultats légers, libéré les builders après chaque test et supprimé les scores `Unitary_*` temporaires.

Les optimisations portent sur six points :

1. charger uniquement la période utile avec `load_backtest_data(..., start_date, lookback_periods=12)` ;
2. partager `screen` et `returns` entre les builders en lecture seule ;
3. préparer une seule fois chaque mois pour Top et Worst ;
4. vectoriser la neutralisation sectorielle des rangs ;
5. calculer les rendements quotidiens sans convertir les matrices complètes en tables longues ;
6. vérifier strictement l’équivalence des performances, positions et métriques.

## 2. Chargement limité à la période utile

L’appel recommandé est désormais :

```python
screen, returns = load_backtest_data(
    SCREEN_PATH,
    RETURNS_PATH,
    variables=RAW_VARIABLES,
    signal_config=WIDE_CONFIG,
    bench=BENCHMARK,
    start_date=START_DATE,
    lookback_periods=12,
)
```

`start_date` filtre les rendements dès la lecture logique. Pour le screen, la fonction conserve aussi les périodes antérieures nécessaires aux transformations `pct_N`, `diff_N` et `rank_diff_N`. La valeur par défaut `lookback_periods=12` couvre donc les comparaisons à 1, 3, 6 et 12 périodes.

Dans le scénario mesuré, la projection et le filtre produisent :

| Donnée | Avant | Après | Réduction |
|---|---:|---:|---:|
| Lignes du screen | 3 454 342 | 2 255 832 | 34,7 % |
| Lignes de rendements | 5 511 | 4 232 | 23,2 % |
| Colonnes de rendements | 1 288 | 1 164 | 9,6 % |

Seules les colonnes techniques, les variables demandées, leurs éventuels dénominateurs et les membres du benchmark sont chargés.

## 3. Partage des données et du prétraitement Top/Worst

Les builders ne réalisent plus de copie profonde de `screen` et `returns` lors de leur construction. Les deux objets sont partagés en lecture seule. Une copie locale et limitée aux colonnes nécessaires est créée uniquement avant une transformation effective.

La préparation mensuelle commune comprend notamment :

- la fusion des tickers secondaires ;
- le filtre des membres du benchmark ;
- le traitement de la capitalisation ;
- le lissage des pondérations ;
- le calcul des poids sectoriels du benchmark.

Cette préparation est exécutée une fois, puis consommée séparément par Top et Worst. Les règles propres à Top, notamment l’ESG et la liste noire, restent appliquées uniquement à Top.

## 4. Neutralisation sectorielle vectorisée

Le calcul des rangs globaux et sectoriels utilise désormais les opérations vectorisées de pandas. La succession des normalisations reste identique : rang global, normalisation entre 0 et 1, rang au sein du secteur, puis nouvelle normalisation entre 0 et 1.

Les cas de valeurs manquantes et d’égalités conservent le comportement de référence. L’équivalence a été vérifiée sur un marché synthétique déterministe et sur l’historique réel complet.

## 5. Calcul quotidien des performances

L’ancien moteur transformait deux matrices complètes, `returns` et `returns_drift`, en tables longues avec `stack()`, puis les fusionnait avec chaque ligne du portefeuille. Cette étape créait plusieurs tables intermédiaires très volumineuses.

Le nouveau moteur :

1. associe chaque date de rendement à la dernière date de rebalancement avec une recherche vectorisée ;
2. rebase la matrice cumulée sur cette date ;
3. récupère directement le multiplicateur de dérive et le rendement par leurs positions ligne/colonne ;
4. évite les deux `stack()` et les deux fusions correspondantes.

Le calcul reste effectué avant toute fonction de graphique ou d’export.

## 6. Protocole de mesure

Les mesures avant et après utilisent exactement le même scénario :

- processeur Intel Core i9-13900HX et 64 Go de mémoire ;
- données réelles `screen_aggregate.parquet` et `returns.parquet` ;
- benchmark `STOXX EUROPE 600`, calculé une fois puis injecté ;
- période du 1er janvier 2010 à juillet 2026 ;
- test unitaire de `Revenue 5Y CAGR | level` ;
- portefeuilles Top et Worst à 13 % ;
- points de rupture 2020, 2022 et 2024 ;
- `fill_method="copy"`, graphiques désactivés et builders non conservés.

Le RSS du processus est échantillonné toutes les 10 ms. Le « pic additionnel » correspond au pic RSS de l’étape moins son RSS initial. Les mesures « avant » proviennent d’un snapshot créé avant les modifications de cette phase.

## 7. Résultats mesurés

| Étape | Temps avant | Temps après | Gain temps | Pic additionnel avant | Pic additionnel après | Gain pic |
|---|---:|---:|---:|---:|---:|---:|
| Chargement | 0,888 s | 0,791 s | 10,8 % | 690,1 Mio | 616,8 Mio | 10,6 % |
| Benchmark | 10,187 s | 5,530 s | 45,7 % | 1 544,6 Mio | 1 011,6 Mio | 34,5 % |
| Test unitaire complet | 62,021 s | 36,647 s | **40,9 %** | 1 691,2 Mio | 616,0 Mio | **63,6 %** |

Le temps cumulé des trois étapes passe de **73,10 s à 42,97 s**, soit un gain de **41,2 %**. Le pic RSS absolu maximal observé passe de **2 458,7 Mio à 1 655,5 Mio**, soit une baisse de **32,7 %**.

La mémoire encore retenue après le test unitaire passe de 60,1 Mio à 42,4 Mio, soit une baisse supplémentaire de 29,5 %. Cette valeur s’ajoute au gain de la première phase, qui avait déjà supprimé la rétention des builders complets dans chaque résultat.

Une mesure isolée dépend de la charge de la machine et du cache disque. Les gains les plus robustes sont la disparition des tables longues intermédiaires et la réduction structurelle du nombre de lignes/colonnes chargées.

## 8. Équivalence stricte des résultats

Le contrôle réel complet compare les objets avant et après avec `pandas.testing.assert_frame_equal(..., check_exact=True)`. Tous les contrôles sont exacts :

| Objet contrôlé | Résultat |
|---|---|
| Performance Top, Worst et Benchmark | Exact |
| Positions historiques Top | Exact |
| Positions historiques Worst | Exact |
| Ratios Top/Benchmark, Worst/Benchmark et Top/Worst | Exact |
| Métriques classiques | Exact |
| Métriques par période | Exact |
| Score robuste et métriques robustes | Exact |

Les valeurs principales restent donc :

| Indicateur | Avant | Après |
|---|---:|---:|
| Score de robustesse | -0,7371 | -0,7371 |
| Top/Benchmark | -0,0665 | -0,0665 |
| Top/Worst | 0,0845 | 0,0845 |

La suite automatisée fige en plus des empreintes SHA-256 déterministes pour les performances, les positions Top/Worst, les métriques classiques et les périodes. Elle couvre également le benchmark fourni, le partage des entrées, la préparation mensuelle commune, le cas où `ISIN` est un index parquet, les directions, les compositions et la reconstruction par période.

## 9. Audit de la neutralisation sectorielle historique

L’algorithme historique n’est pas erroné dans son périmètre, mais il faut préciser sa définition. Il transforme d’abord chaque score en rang sectoriel comparable, sélectionne ensuite globalement les meilleurs ou les moins bons rangs, puis ajuste les poids du portefeuille aux poids sectoriels du benchmark. Il garantit donc surtout une neutralité des **poids** après sélection. Il ne garantit pas qu’exactement 13 % des titres de chaque secteur soient sélectionnés.

Sur `Revenue 5Y CAGR` dans le `STOXX EUROPE 600`, l’audit de 209 mois et 3 971 groupes secteur-mois montre :

- aucune ligne du benchmark sans secteur ICB 19 ;
- 572 groupes contenant au moins une égalité de valeur ;
- 19 groupes sans aucune valeur brute valide, tous en avril 2026 ;
- au dernier mois, un taux de sélection sectoriel compris entre 11,1 % et 16,7 % pour une cible globale de 13 % ;
- sur l’historique brut, un mois peut sélectionner des scores manquants lorsque toute la coupe transversale est indisponible.

Les principaux risques sont les suivants :

1. une normalisation min-max renvoie `NaN` lorsqu’un secteur contient moins de deux scores distincts ;
2. `nlargest()` ou `nsmallest()` peut alors compléter la liste avec des lignes manquantes selon leur ordre d’origine ;
3. les égalités au seuil ne disposent pas d’une règle secondaire métier explicite ;
4. le nombre cible est calculé avant certains filtres d’éligibilité, ce qui peut modifier le percentile effectif ;
5. la neutralisation utilise le secteur, mais pas la combinaison région × secteur nécessaire à un benchmark réellement multirégional ;
6. le premier rang global et sa normalisation sont redondants lorsque le rang sectoriel est ensuite recalculé.

La vectorisation livrée conserve volontairement cette définition historique afin de garantir l’équivalence des backtests. Le cas d’un secteur manquant conserve également le rang global, comme dans l’ancienne boucle.

Une future version, activée par une option distincte afin de ne pas casser l’historique, devrait :

1. appliquer les filtres d’éligibilité avant de calculer les quotas ;
2. contrôler la couverture, le nombre de valeurs valides et le nombre de valeurs distinctes par date et groupe ;
3. définir une politique explicite pour les groupes invalides : erreur, conservation du portefeuille précédent ou score neutre ;
4. classer directement dans chaque groupe région × secteur, sans préclassement global ;
5. attribuer un quota par groupe avec la méthode des plus forts restes afin de respecter à la fois le percentile sectoriel et le nombre total ;
6. départager les égalités de manière stable avec une clé secondaire documentée, par exemple capitalisation puis ISIN ;
7. exporter les diagnostics de couverture, quotas, égalités et solutions de repli avec les résultats.

## 10. Limites et prochaines optimisations possibles

Le coût restant est principalement concentré dans la génération mensuelle des listes et dans certaines opérations pandas historiques encore basées sur des `groupby.apply()` ou des boucles de dates.

Les prochaines pistes, à traiter avec le même protocole d’équivalence, sont :

1. remplacer les derniers `groupby.apply()` de sélection/pondération par des opérations groupées plus directes ;
2. mettre en cache les préparations communes entre plusieurs signaux utilisant exactement le même univers et la même date ;
3. utiliser des types plus compacts pour les identifiants et secteurs lorsque le parquet les permet ;
4. profiler séparément la création des variables dérivées sur les campagnes de 10 à 15 variables ;
5. ajouter un benchmark automatisé non bloquant afin de suivre les régressions de temps et de mémoire sans rendre les tests fonctionnels instables.
