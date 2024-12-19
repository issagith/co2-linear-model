import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_histograms(data):
    """Affiche des histogrammes pour toutes les colonnes numériques sur un même graphique."""
    numerical_data = data.select_dtypes(include=['number'])
    num_columns = len(numerical_data.columns)
    cols = 3  # Nombre de colonnes dans la figure
    rows = (num_columns + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for i, column in enumerate(numerical_data.columns):
        axes[i].hist(data[column].dropna(), bins=15, edgecolor='k', alpha=0.7)
        axes[i].set_title(f"Histogramme de {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Fréquence")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Supprime les axes inutilisés si le nombre de colonnes est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2.0)  # Ajoute de l'espace vertical entre les graphiques
    plt.show()


def plot_boxplots(data):
    """Affiche des boxplots pour toutes les colonnes numériques sur un même graphique."""
    numerical_data = data.select_dtypes(include=['number'])
    num_columns = len(numerical_data.columns)
    cols = 3  # Nombre de colonnes dans la figure
    rows = (num_columns + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for i, column in enumerate(numerical_data.columns):
        axes[i].boxplot(data[column].dropna(), vert=False, patch_artist=True)
        axes[i].set_title(f"Boxplot de {column}")
        axes[i].set_xlabel(column)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)

    # Supprime les axes inutilisés si le nombre de colonnes est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2.0)  # Ajoute de l'espace vertical entre les graphiques
    plt.show()


def plot_histogram(data, column):
    """Affiche un histogramme pour une colonne numérique spécifique."""
    if column not in data.columns:
        raise ValueError(f"La colonne {column} n'existe pas dans les données.")
    plt.figure(figsize=(8, 6))
    plt.hist(data[column].dropna(), bins=15, edgecolor='k', alpha=0.7)
    plt.title(f"Histogramme de {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_boxplot(data, column):
    """Affiche un boxplot pour une colonne numérique spécifique."""
    if column not in data.columns:
        raise ValueError(f"La colonne {column} n'existe pas dans les données.")
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[column].dropna(), vert=False, patch_artist=True)
    plt.title(f"Boxplot de {column}")
    plt.xlabel(column)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def plot_correlation_heatmap(data):
    """Affiche une heatmap des corrélations entre les colonnes numériques."""
    numerical_data = data.select_dtypes(include=['number'])
    if numerical_data.empty:
        raise ValueError("Les données ne contiennent pas de colonnes numériques.")
    corr_matrix = numerical_data.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title("Heatmap des corrélations")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(data):
    """Affiche des graphiques en barres pour toutes les colonnes catégorielles sur un même graphique."""
    categorical_data = data.select_dtypes(include=['object'])
    num_columns = len(categorical_data.columns)
    cols = 3  # Nombre de colonnes dans la figure
    rows = (num_columns + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for i, column in enumerate(categorical_data.columns):
        value_counts = data[column].value_counts()
        axes[i].bar(value_counts.index, value_counts.values, color='lightgreen', edgecolor='k')
        axes[i].set_title(f"Distribution des valeurs pour {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Fréquence")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Supprime les axes inutilisés si le nombre de colonnes est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2.0)  # Ajoute de l'espace vertical entre les graphiques
    plt.show()

def plot_categorical_distribution(data, column):
    """Affiche un graphique en barres pour une colonne catégorielle spécifique."""
    if column not in data.columns:
        raise ValueError(f"La colonne {column} n'existe pas dans les données.")
    if not pd.api.types.is_object_dtype(data[column]):
        raise ValueError(f"La colonne {column} n'est pas catégorielle.")
    value_counts = data[column].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index, value_counts.values, color='lightgreen', edgecolor='k')
    plt.title(f"Distribution des valeurs pour {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_target_correlations(data, target):
    """Affiche des scatter plots entre la variable cible et toutes les autres colonnes numériques."""
    numerical_data = data.select_dtypes(include=['number'])
    if target not in numerical_data.columns:
        raise ValueError(f"La variable cible {target} n'est pas numérique ou n'existe pas dans les données.")
    features = numerical_data.drop(columns=[target]).columns
    num_features = len(features)
    cols = 3
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        axes[i].scatter(data[feature], data[target], alpha=0.7, edgecolor='k')
        axes[i].set_title(f"{feature} vs {target}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].grid(linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2.0)  # Ajoute de l'espace vertical entre les graphiques
    plt.show()

def plot_target_correlation(data, target, feature):
    """Affiche un scatter plot entre la variable cible et une colonne numérique spécifique."""
    if target not in data.columns or feature not in data.columns:
        raise ValueError(f"Une ou plusieurs colonnes spécifiées n'existent pas dans les données.")
    plt.figure(figsize=(8, 6))
    plt.scatter(data[feature], data[target], alpha=0.7, edgecolor='k')
    plt.title(f"{feature} vs {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.grid(linestyle='--', alpha=0.7)
    plt.show()

def plot_missing_values(data):
    """Affiche un graphique en barres pour les valeurs manquantes de chaque colonne."""
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        print("Aucune valeur manquante dans les données.")
        return
    plt.figure(figsize=(10, 6))
    plt.bar(missing_values.index, missing_values.values, color='lightcoral', edgecolor='k')
    plt.title("Nombre de valeurs manquantes par colonne")
    plt.xlabel("Colonnes")
    plt.ylabel("Nombre de valeurs manquantes")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_boxplot_by_category(data, numeric_column, category_column):
    """Affiche un boxplot d'une variable numérique selon une variable catégorielle."""
    if numeric_column not in data.columns or category_column not in data.columns:
        raise ValueError(f"Les colonnes {numeric_column} ou {category_column} n'existent pas dans les données.")
    plt.figure(figsize=(10, 6))
    data.boxplot(column=numeric_column, by=category_column, grid=False, patch_artist=True)
    plt.title(f"{numeric_column} par {category_column}")
    plt.suptitle("")  # Supprime le titre par défaut
    plt.xlabel(category_column)
    plt.ylabel(numeric_column)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def main():
    """Exécute des exemples pour démontrer les visualisations."""
    # Charger un jeu de données exemple
    file_path = "../data/vehicles.csv"
    try:
        data = pd.read_csv(file_path)
        print(f"Données chargées depuis {file_path}.")
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return

    # Afficher les valeurs manquantes
    print("\n--- Valeurs manquantes ---")
    plot_missing_values(data)

    # Afficher les histogrammes pour toutes les colonnes numériques
    print("\n--- Histogrammes des colonnes numériques ---")
    plot_histograms(data)

    # Afficher les boxplots pour toutes les colonnes numériques
    print("\n--- Boxplots des colonnes numériques ---")
    plot_boxplots(data)

    # Afficher un boxplot en fonction d'une variable catégorielle
    print("\n--- Boxplot d'une variable numérique par catégorie ---")
    plot_boxplot_by_category(data, "CO2 emissions (g/km)", "Fuel type")

    # Afficher la heatmap des corrélations
    print("\n--- Heatmap des corrélations ---")
    plot_correlation_heatmap(data)

    # Afficher les distributions pour toutes les colonnes catégorielles
    print("\n--- Distributions des colonnes catégorielles ---")
    plot_categorical_distributions(data)

    # Afficher les corrélations avec la variable cible
    target_column = "CO2 emissions (g/km)"  # Exemple de variable cible
    if target_column in data.columns:
        print(f"\n--- Corrélations entre {target_column} et les autres variables numériques ---")
        plot_target_correlations(data, target_column)
    else:
        print(f"La variable cible {target_column} n'existe pas dans les données.")


if __name__ == "__main__":
    main()
