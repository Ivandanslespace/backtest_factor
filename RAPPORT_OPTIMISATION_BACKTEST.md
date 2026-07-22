# Rapport d’optimisation du moteur de backtest

Date de mesure : 22 juillet 2026

## 1. Objectif

Cette phase réduit le temps d’exécution et le pic de mémoire d’un backtest complet, sans modifier les positions ni les résultats financiers. Elle complète la première phase, qui avait déjà rendu les résultats légers, libéré les builders après chaque test et supprimé les scores `Unitary_*` temporaires.

Une seconde passe, mesurée le même jour, optimise spécifiquement les campagnes comportant plusieurs variables et plusieurs horizons. Elle regroupe la création des dérivées, conserve une base mensuelle commune entre les signaux et compacte les identifiants répétés.

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

## 10. Profilage des variables dérivées

Le profilage séparé utilise dix variables réelles, les treize dimensions `level`, `pct_N`, `diff_N` et `rank_diff_N`, ainsi que les quatre horizons 1, 3, 6 et 12. La fenêtre contrôlée commence le 1er juillet 2025 et contient 130 072 lignes. Elle produit 120 colonnes dérivées conservées dans `screen`.

Avant cette passe, chaque dimension dérivée reconstruisait et triait sa propre table. Une variable demandant douze dérivées déclenchait donc douze tris. La nouvelle fonction prépare la valeur brute et le rang une fois, trie une seule fois par ISIN et date, puis génère tous les horizons à partir de cette table ordonnée.

| Mesure profilée | Avant | Après | Gain |
|---|---:|---:|---:|
| Temps total | 41,400 s | 16,445 s | **60,3 %** |
| Nombre de tris | 120 | 10 | **91,7 %** |
| Pic additionnel | 383,1 Mio | 383,1 Mio | stable |
| Mémoire retenue | 139,1 Mio | 134,5 Mio | 3,3 % |

Le pic reste stable parce que les 120 colonnes finales doivent volontairement rester disponibles dans `screen`. Le gain porte donc surtout sur les calculs et les tables temporaires. Les 130 colonnes de sortie, comprenant les dix niveaux bruts et les 120 dérivées, ont été comparées avec `check_exact=True` : elles sont strictement identiques à la référence.

## 11. Base mensuelle commune entre les signaux

`monthly_base_cache` conserve, pour chaque date et pour un univers de backtest donné, uniquement les colonnes techniques déjà préparées : date, secteur, poids du benchmark et capitalisation. Le score courant est rattaché à cette base au moment du test. Top et Worst, puis les signaux suivants, réutilisent ainsi la même préparation mensuelle.

Le cache est lié à l’identité de `screen` et se vide automatiquement si un autre objet est fourni. Une valeur `None` le désactive explicitement. Dans le notebook, un dictionnaire partagé dans `RUN_OPTIONS` permet de le conserver entre les appels unitaires, incrémentaux et composites.

Une première version conservait accidentellement les catégories ISIN globales dans chaque table mensuelle et occupait 361,1 Mio. Le cache final utilise un index ISIN local et ne conserve pas SEDOL, qui n’intervient pas dans la sélection mensuelle. Pour 198 mois, il n’occupe plus que **9,5 Mio**, soit une réduction de **97,4 %** par rapport à cette version intermédiaire.

Sur le test réel complet, le premier signal construit le cache en 11,52 s. Le second signal, exécuté avec le même univers et les mêmes options, le réutilise en 10,92 s, soit un gain de 5,2 %. Le gain global est volontairement limité : après les optimisations précédentes, le calcul quotidien des performances et des métriques constitue désormais la majeure partie du temps d’un signal.

## 12. Types compacts pour les identifiants et secteurs

`load_backtest_data(..., compact_dtypes=True)` est activé par défaut. Il utilise :

- un `CategoricalIndex` pour ISIN ;
- le type `category` pour SEDOL et la région ;
- `Int8` nullable pour les codes de supersecteur et d’industrie lorsqu’ils sont entiers et compatibles ;
- les types numériques d’origine pour les variables financières et les rendements, afin de ne perdre aucune précision.

Sur le screen projeté de 2 255 832 lignes et sept colonnes, la mémoire pandas mesurée passe de 485,7 Mio à **72,8 Mio**, soit une baisse de **85,0 %**. Le RSS réellement retenu par l’étape de chargement passe de 555,4 Mio lors de la mesure précédente à 356,8 Mio dans la nouvelle mesure. Le temps de lecture peut légèrement augmenter à cause de la conversion des catégories ; ce coût est payé une seule fois et bénéficie ensuite à toute la campagne.

## 13. Mesure complète et équivalence de la nouvelle passe

La mesure complète conserve le scénario de la section 6. Les résultats suivants comparent la passe précédente, déjà optimisée, à la présente passe :

| Étape | Passe précédente | Nouvelle passe | Évolution |
|---|---:|---:|---:|
| Chargement | 0,791 s | 1,338 s | conversion compacte incluse |
| Pic additionnel du chargement | 616,8 Mio | 535,4 Mio | **-13,2 %** |
| Benchmark | 5,530 s | 3,140 s | **-43,2 %** |
| Pic additionnel du benchmark | 1 011,6 Mio | 811,8 Mio | **-19,8 %** |
| Premier signal complet | 36,647 s | 11,520 s | **-68,6 %** |
| Pic additionnel du premier signal | 616,0 Mio | 487,9 Mio | **-20,8 %** |
| Signal suivant avec cache | — | 10,918 s | base mensuelle réutilisée |

Les temps isolés restent sensibles au cache disque et à la charge de la machine. Les mesures structurelles les plus fiables sont la réduction du nombre de tris, la taille pandas du screen et la taille du cache mensuel.

Après ces modifications, le contrôle sur l’historique réel complet reste exact pour les performances, les positions Top et Worst, les ratios, les métriques classiques, les métriques par période et tous les scalaires du score robuste. La suite automatisée contient désormais 22 tests et couvre explicitement le tri unique, les types compacts et la réutilisation de la base mensuelle.

## 14. Limites et prochaines optimisations possibles

Le coût restant est principalement concentré dans le calcul quotidien répété pour chaque signal et dans certaines opérations pandas historiques encore basées sur `groupby.apply()` ou sur des boucles de dates.

Les prochaines pistes, à traiter avec le même protocole d’équivalence, sont :

1. remplacer les derniers `groupby.apply()` de sélection et de pondération par des opérations groupées plus directes ;
2. mutualiser, lorsque cela est mathématiquement possible, les étapes quotidiennes communes à plusieurs portefeuilles ;
3. ajouter un benchmark automatisé non bloquant afin de suivre les régressions de temps et de mémoire sans rendre les tests fonctionnels instables ;
4. étudier un cache persistant optionnel uniquement si les campagnes traversant plusieurs redémarrages de kernel le justifient.
