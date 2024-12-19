# LinearModel Package

## Description

Le package LinearModel est une bibliothèque Python conçue pour effectuer des analyses statistiques avancées et implémenter des modèles de régression linéaire utilisant les moindres carrés ordinaires (OLS). Il inclut des modules pour la prétraitement des données, l'analyse exploratoire des données, la visualisation et l'évaluation des modèles. Ce package a été développé dans le contexte de l'analyse des émissions de CO2 des véhicules en fonction de diverses caractéristiques. Chaque module peut être lancé indépendamment et contient une fonction main pour une utilisation rapide sur le jeu de donné. Mais, il est possible d'utiliser les fonctions de chaque module dans un script python indépendant.

Le fichier main est utilisé comme un notebook pour l'analyse des données et l'entraînement du modèle. Il contient des visualisations et des interprétations des résultats et sert d'exemple d'utilisation du package. Les tests unitaires sont fournis pour valider le bon fonctionnement des méthodes dans chaque module.


## Fonctionnalités

### 1. Module myStats

Ce module fournit des outils pour :

- Charger et inspecter les données : `load_data`, `data_overview`, `columns_overview`.
- Analyser les statistiques descriptives : `numerical_summary`, `categorical_summary`.
- Identifier et traiter les valeurs manquantes : `missing_values_analysis`, `delete_missing_values`.
- Identifier les valeurs extrêmes : `outliers_count`, `vehcile_outliers`.
- Analyser les corrélations : `correlation_analysis`.

### 2. Module visualization

Ce module fournit des outils de visualisation pour :

- Afficher des histogrammes et boxplots.
- Créer des heatmaps de corrélation.
- Visualiser les distributions catégoriques et les relations entre variables cibles et explicatives.

### 3. Module ordinaryLeastSquares

La classe `OrdinaryLeastSquares` implémente un modèle de régression linéaire basé sur les moindres carrés ordinaires (OLS). Ses fonctionnalités incluent :

- Ajustement du modèle avec `fit()`.
- Prédiction avec `predict()`.
- Calcul des métriques : `R²`, `MSE`, `RMSE`, `MAE`, `MAPE`.
- Tests d’hypothèses sur les coefficients.
- Vérification des hypothèses du modèle : indépendance des erreurs, distribution des résidus.

### 4. Fichier main.py

Un programme principal qui :

- Effectue une analyse statistique et visualisation des données.
- Prépare les données pour l'entraînement du modèle linéaire (nettoyage, encodage, standardisation).
- Entraîne un modèle linéaire avec `OrdinaryLeastSquares`.
- Évalue les performances du modèle sur des jeux de données de test et d'entraînement.
- Fournit des visualisations et interprétations des résultats.

### 5. Tests

Les tests unitaires se trouvent dans le dossier `tests` et utilisent `pytest`. Ils valident :

- Le bon fonctionnement des méthodes dans `myStats` et `visualization`.
- La précision des prédictions et métriques calculées dans `ordinaryLeastSquares`.


## Installation

Une fois que vous avez accès au fichier dist/linearmodel-version.tar.gz, vous pouvez installer le package en utilisant la commande suivante :

```sh
pip install dist/linearmodel-version.tar.gz
```

Pour installer le package depuis le dépôt GitHub, vous pouvez utiliser la commande suivante :

```sh
git clone urlDuDepot
cd LinearModel
pip install .
```


## Prérequis

Assurez-vous d'avoir installé les bibliothèques nécessaires en utilisant le fichier `requirements.txt` fourni. Vous pouvez les installer avec la commande suivante :

```sh
pip install -r requirements.txt
```

## Utilisation

### Exemple rapide

Voici un exemple d'utilisation du package pour analyser un jeu de données et entraîner un modèle linéaire :

```python
from linearmodel.myStats import load_data, numerical_summary
from linearmodel.visualization import plot_correlation_heatmap
from linearmodel.ordinaryLeastSquares import OrdinaryLeastSquares

# Charger les données
file_path = "data/vehicles.csv"
data = load_data(file_path)

# Analyse des statistiques
print(numerical_summary(data))

# Visualisation des corrélations
plot_correlation_heatmap(data)

# Préparation des données et entraînement du modèle
X = data[["Engine size (L)", "Cylinders"]]
y = data["CO2 emissions (g/km)"]
model = OrdinaryLeastSquares(intercept=True)
model.fit(X, y)

# Évaluation du modèle
print("Coefficients:", model.get_coeffs())
print("R²:", model.r2_score(X, y))
```

## Contraintes

- Librairies autorisées : NumPy, Pandas, Matplotlib, Seaborn.
- Librairies proscrites : sklearn, statsmodels.

## Auteur

Issa KA


