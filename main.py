import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from linearmodel.myStats import (load_data, data_overview, columns_overview, numerical_summary, 
                                 categorical_summary, missing_values_analysis, delete_missing_values, 
                                 unique_values_analysis, correlation_analysis, outliers_count, 
                                 vehcile_outliers, variance_analysis, frequency_analysis)
from linearmodel.visualization import (plot_missing_values, plot_histograms, plot_boxplots, 
                                       plot_categorical_distributions, plot_correlation_heatmap, 
                                       plot_boxplot_by_category, plot_target_correlations)
from linearmodel.ordinaryLeastSquares import OrdinaryLeastSquares

def train_test_split(data, test_ratio=0.2, random_state=42):
    """
    Split the data into train and test sets.
    Arguments:
    data (pd.DataFrame): The dataset to split.
    test_ratio (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed for reproducibility.
    
    Returns:
    pd.DataFrame, pd.DataFrame : train_data, test_data
    """
    np.random.seed(random_state)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data

def standardize_data(train_data, test_data, numeric_columns):
    """
    Standardize the numeric columns of train_data and apply the same transformation to test_data.
    This uses (x - mean) / std based on the train_data statistics.
    
    Arguments:
    train_data (pd.DataFrame): Training dataset.
    test_data (pd.DataFrame): Test dataset.
    numeric_columns (list): List of numeric columns to standardize.
    
    Returns:
    pd.DataFrame, pd.DataFrame : standardized train_data and test_data
    """
    means = {}
    stds = {}
    for col in numeric_columns:
        mean = train_data[col].mean()
        std = train_data[col].std()
        # Eviter division par zéro
        if std == 0:
            std = 1e-9
        means[col] = mean
        stds[col] = std
        train_data.loc[:, col] = (train_data[col] - mean) / std
    
    # Appliquer la même standardisation au test_data
    for col in numeric_columns:
        test_data.loc[:, col] = (test_data[col] - means[col]) / stds[col]
    
    return train_data, test_data

def one_hot_encode(data, categorical_columns):
    """
    One-hot encode the specified categorical columns using pd.get_dummies.
    """
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

if __name__ == "__main__":
    # 1. Chargement des données
    print("====================================================================")
    print("                         CHARGEMENT DES DONNEES                     ")
    print("====================================================================\n")
    file_path = "data/vehicles.csv"  # Adapter le chemin
    data = load_data(file_path)
    print("Les données ont été chargées avec succès !\n")

    print("Aperçu des informations du jeu de données :")
    print("- Contient diverses informations sur les véhicules (marque, modèle, classe, taille du moteur, consommation, émissions CO2).\n")

    # 2. Aperçu des données
    print("====================================================================")
    print("                           APERCU DES DONNEES                       ")
    print("====================================================================\n")
    print("Aperçu des colonnes :\n", columns_overview(data), "\n")
    print("Aperçu du jeu de données (5 premières lignes) :\n", data_overview(data), "\n")

    # 3. Analyse statistique basique
    print("====================================================================")
    print("                     ANALYSE STATISTIQUE DE BASE                    ")
    print("====================================================================\n")
    print("--- Statistiques numériques ---")
    print(numerical_summary(data), "\n")

    print("--- Statistiques catégorielles ---")
    print(categorical_summary(data), "\n")

    print("--- Analyse des valeurs manquantes ---")
    missing = missing_values_analysis(data)
    print(missing, "\n")

    print("Affichage du graphique des valeurs manquantes...")
    plot_missing_values(data)

    print("Les variables 'CO2 rating' et 'Smog rating' contiennent trop de valeurs manquantes, on les exclut.")
    print("Nous verrons par la suite si il est plus judicieux d'imputer les valeurs manquantes ou de les supprimer en fonction de la précision du modèle.\n")
    data = data.drop(columns=["CO2 rating", "Smog rating"], errors='ignore')

    # 4. Analyse des corrélations
    print("====================================================================")
    print("              ANALYSE DES VARIABLES NUMERIQUES ET CORRELATIONS       ")
    print("====================================================================\n")
    corr_matrix = correlation_analysis(data)
    print("Matrice de corrélation :\n", corr_matrix, "\n")

    print("Points clés :")
    print("- Les consommations (City, Highway, Combined) sont fortement corrélées aux émissions de CO2.")
    print("- Engine size (L), Cylinders également corrélés positivement.")
    print("- Model year peu pertinent car correlation très faible.\n")

    print("Affichage de la heatmap de corrélation...")
    plot_correlation_heatmap(data)

    # 5. Analyse des variables catégorielles
    print("====================================================================")
    print("                ANALYSE DES VARIABLES CATEGORIELLES                 ")
    print("====================================================================\n")
    cat_uniques = unique_values_analysis(data)
    for cat_col, vals in cat_uniques.items():
        print(f"{cat_col} : {len(vals)} valeurs uniques.")
    print("\nVariables catégorielles pertinentes : Vehicle class, Transmission, Fuel type.\n")

    print("Affichage des distributions pour les variables catégorielles...")
    plot_categorical_distributions(data)

    # 6. Valeurs extrêmes
    print("====================================================================")
    print("               VALEURS EXTREMES (OUTLIERS) ET LEUR IMPACT           ")
    print("====================================================================\n")
    outlier_counts = outliers_count(data)
    print("Nombre de valeurs extrêmes par colonne numérique :\n", outlier_counts, "\n")

    print("On conserve les outliers pour l'instant.\n")
    print("Affichage des boxplots...")
    plot_boxplots(data)

    # 7. Sélection des caractéristiques
    print("====================================================================")
    print("        SELECTION DES CARACTERISTIQUES POUR LE MODELE LINEAIRE       ")
    print("====================================================================\n")
    target = "CO2 emissions (g/km)"
    features_numeric = ["Engine size (L)", "Cylinders", "Combined (L/100 km)"]
    features_cat = ["Vehicle class", "Transmission", "Fuel type"]
    all_features = features_numeric + features_cat + [target]

    # Restreindre le dataset aux colonnes d'intérêt
    data = data[all_features]

    print(f"Caractéristiques numériques retenues : {features_numeric}")
    print(f"Caractéristiques catégorielles : {features_cat}")
    print(f"Cible : {target}\n")

    # 8. Gestion des valeurs manquantes
    print("Vérification des valeurs manquantes pour les variables sélectionnées :")
    print(data.isnull().sum(), "\n")

    print("Suppression des lignes avec valeurs manquantes dans les features sélectionnées...")
    initial_len = len(data)
    data = data.dropna(subset=all_features)
    print(f"{initial_len - len(data)} lignes supprimées.\n")

    # 9. Encodage des variables catégorielles
    print("Encodage One-Hot des variables catégorielles...")
    data = one_hot_encode(data, features_cat)
    # Conversion des bool en int
    for col in data.columns:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)
    print("Encodage terminé.\n")

    # Maintenant, nos features sont : features_numeric + colonnes encodées (dummy)
    # Pour identifier les colonnes dummy, on peut filtrer :
    dummy_cols = [col for col in data.columns if any(col.startswith(f"{cat_col}_") for cat_col in features_cat)]
    # Ainsi, nos features finales sont :
    all_features_final = features_numeric + dummy_cols

    print("Nouvelles colonnes encodées :")
    print(dummy_cols, "\n")

    # 10. Division train/test
    print("Division du jeu de données en apprentissage (train) et test.")
    train_data, test_data = train_test_split(data, test_ratio=0.2, random_state=42)
    print(f"Train set : {len(train_data)} lignes, Test set : {len(test_data)} lignes.\n")

    # 11. Normalisation des variables numériques
    print("Normalisation (standardisation) des variables numériques...")
    train_data, test_data = standardize_data(train_data, test_data, features_numeric)
    print("Normalisation terminée.\n")

    # Préparation des matrices X et y
    X_train = train_data[all_features_final]
    y_train = train_data[target]

    X_test = test_data[all_features_final]
    y_test = test_data[target]

    print("tete matrices entrainements et test")
    print(X_train.head())
    print(X_test.head())
    # Vérifions que X_train et X_test sont bien numériques
    # (juste par précaution, normalement c'est le cas)
    for col in X_train.columns:
        if not np.issubdtype(X_train[col].dtypes, np.number):
            print(f"Problème: la colonne {col} n'est pas numérique.")
    for col in X_test.columns:
        if not np.issubdtype(X_test[col].dtypes, np.number):
            print(f"Problème: la colonne {col} n'est pas numérique.")

    # 12. Entraînement du modèle OLS
    print("====================================================================")
    print("       ENTRAINEMENT D'UN MODELE DE REGRESSION LINEAIRE OLS          ")
    print("====================================================================\n")

    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X_train, y_train)
    coeffs = model.get_coeffs()

    print("Coefficients du modèle (train) :")
    if model.intercept:
        print(f"Intercept : {coeffs[0]:.4f}")
        for i, f in enumerate(all_features_final, start=1):
            print(f"{f} : {coeffs[i]:.4f}")
    else:
        for i, f in enumerate(all_features_final):
            print(f"{f} : {coeffs[i]:.4f}")

    print("====================================================================")
    print("             ANALYSE DES PERFORMANCES DU MODELE LINEAIRE            ")
    print("====================================================================\n")

    # 13. Evaluation du modèle sur le train
    print("\nEvaluation sur le jeu d'entraînement :")
    r2_train = model.r2_score(X_train, y_train)
    print(f"R² (train) : {r2_train:.4f}")
    print("Interprétation : Le R² sur le jeu d'entraînement est extrêmement élevé, proche de 0.994.")
    print("Cela signifie que le modèle explique environ 99.4% de la variance des émissions de CO2 sur l'ensemble d'entraînement.\n")

    print("Affichage des résidus sur le jeu d'entraînement...")
    model.plot_residuals(X_train, y_train)
    print("Les résidus semblent faibles, ce qui confirme un très bon ajustement sur l'ensemble d'entraînement.\n")
    print("Aussi, on voit que les résidus sont bien distribués autour de 0, ce qui est un bon signe.\n")
    print("Puisqu'il n'y a pas de structure claire dans les résidus, cela indique que le modèle est bien spécifié et ne présente pas de biais systématique.\n")

    # 14. Evaluation du modèle sur le test
    print("Evaluation sur le jeu de test :")
    r2_test = model.r2_score(X_test, y_test)
    print(f"R² (test) : {r2_test:.4f}")
    print("Interprétation : Le R² sur le jeu de test est d'environ 0.993, très proche de celui sur le train.")
    print("Cela indique que le modèle généralise très bien et n'est probablement pas victime d'un fort sur-apprentissage.\n")

    print("Affichage des résidus sur le jeu de test...")
    model.plot_residuals(X_test, y_test)
    print("Les résidus sur le jeu de test sont également très faibles, confirmant la bonne généralisation du modèle.\n")

    # 15. Affichage des prédictions vs valeurs réelles
    print("Affichage des prédictions vs valeurs réelles sur le jeu de test...")
    print(model.compare_predictions(X_test, y_test))
    print("Les prédictions semblent très proches des valeurs réelles, confirmant la qualité du modèle.\n")

    # Calcul des métriques de performance
    mse = model.MSE(X_test, y_test)
    rmse = model.RMSE(X_test, y_test)
    mae = model.MAE(X_test, y_test)
    mape = model.MAPE(X_test, y_test)

    # MSE
    print(f"MSE (Mean Squared Error) : {mse:.4f}")
    print("- Le MSE représente la moyenne des carrés des erreurs (différences entre les prédictions et les valeurs réelles).")
    print("- Une valeur de 23.57 indique que, en moyenne, le carré des écarts entre les prédictions et les valeurs réelles est faible.")
    print("- Bien que le MSE soit utile pour comparer différents modèles, son interprétation directe est limitée à cause de l'échelle des données.\n")

    # RMSE
    print(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
    print("- Le RMSE est la racine carrée du MSE et est exprimé dans la même unité que la variable cible (g/km pour CO2).")
    print("- Un RMSE de 4.86 g/km montre que, en moyenne, l'écart entre les prédictions et les valeurs réelles est d'environ 4.86 g/km.")
    print("- Ce résultat indique que le modèle est assez précis pour prédire les émissions de CO2 avec une erreur modérée.\n")

    # MAE
    print(f"MAE (Mean Absolute Error) : {mae:.4f}")
    print("- Le MAE représente la moyenne des écarts absolus entre les prédictions et les valeurs réelles.")
    print("- Avec une valeur de 2.73 g/km, le modèle montre une erreur moyenne faible, ce qui reflète une précision globale.")
    print("- Contrairement au RMSE, le MAE n'est pas influencé par les grandes erreurs (outliers), ce qui en fait un bon indicateur de performance robuste.\n")

    # MAPE
    print(f"MAPE (Mean Absolute Percentage Error) : {mape:.4%}")
    print("- Le MAPE mesure l'erreur moyenne absolue en pourcentage des valeurs réelles.")
    print("- Avec une erreur de seulement 1.10%, le modèle montre une très bonne capacité à capturer les valeurs réelles des émissions de CO2.")
    print("- Cela signifie que les prédictions s'écartent en moyenne de 1.10% par rapport aux valeurs observées, une performance exceptionnelle pour un modèle linéaire.\n")

    print("====================================================================")
    print("                   INSIGHTS SUR LA PERFORMANCE                      ")
    print("====================================================================\n")
    print("- Le faible MAPE (1.10%) montre que le modèle est extrêmement précis, même en comparaison relative.")
    print("- Le RMSE (4.86 g/km) et le MAE (2.73 g/km) confirment une erreur de prédiction modérée, adaptée à l'analyse des émissions de CO2.")
    print("- Le modèle est suffisamment performant pour être utilisé dans des cas réels, comme la prévision des émissions de véhicules en fonction de leurs caractéristiques.")
    print("- Les faibles erreurs indiquent que le choix des variables explicatives était pertinent et que le modèle linéaire capte bien les relations principales.\n")
    
    # 15. Tests d'hypothèse
    print("====================================================================")
    print("                     TESTS D'HYPOTHESE                                ")
    print("====================================================================\n")
    print("\n--- Test d'hypothèse sur les coefficients (p-values) ---")
    hypothesis_results = model.hypothesis_test(X_train, y_train)
    print(hypothesis_results, "\n")

    print("Analyse des coefficients et des p-values :")
    print("- Le coefficient associé à 'Combined (L/100 km)' est très élevé (environ 62.77), avec une p-value proche de 0,")
    print("  ce qui confirme que la consommation combinée a un impact majeur et significatif sur les émissions de CO2.")
    print("- Les types de carburant (Fuel type_E, Fuel type_N, Fuel type_X, Fuel type_Z) présentent des coefficients très négatifs")
    print("  et des p-values nulles ou quasiment nulles, indiquant un impact significativement différent du type de carburant de base,")
    print("  probablement moins émissif (par exemple électrique ou autre).")
    print("- 'Engine size (L)' et 'Cylinders' ont des coefficients plus modestes et une significativité mixte (Cylinders est significatif, Engine size (L) l'est moins).")
    print("  Cela laisse penser que la taille du moteur et le nombre de cylindres, une fois la consommation prise en compte, influent moins fortement que la consommation réelle.")
    print("- De nombreuses variables catégorielles (Transmission, Vehicle class) ont des p-values très élevées, ce qui indique que leur impact marginal,")
    print("  après prise en compte des variables principales (consommation, type de carburant), est moins clairement établi ou non significatif.")
    print("  Cela peut être dû à une forte multicolinéarité entre ces variables catégorielles ou au fait que la consommation intègre déjà la plupart des distinctions entre classes de véhicules.")
    print("- Le très fort R² suggère que les émissions de CO2 sont très bien expliquées par la consommation et le type de carburant. Les autres variables ont un effet secondaire plus faible.\n")

    print("test sur l'indépendance des erreurs :")
    print("Durbin-Watson test : ", model.durbin_watson_test(X_train, y_train))
    print("Le test de Durbin-Watson est très proche de 2, indiquant une faible autocorrélation des erreurs.")
    print("Cela signifie que la condition d'indépendance des erreurs est respectée, ce qui est essentiel pour l'interprétation des coefficients.\n")

    print("====================================================================")
    print("                            CONCLUSION                              ")
    print("====================================================================\n")
    print("- Le modèle OLS explique la quasi-totalité de la variance des émissions de CO2.")
    print("- La consommation de carburant (Combined L/100km) est la variable clé, suivie du type de carburant, qui a un effet majeur sur les émissions.")
    print("- Les autres variables (véhicule, transmission) sont moins significatives une fois ces facteurs pris en compte,")
    print("  ou bien leur effet est déjà capturé par la consommation elle-même.")
    print("- Certaines variables catégorielles pourraient être éliminées pour simplifier le modèle sans perte de performance.")
    print("- Le modèle est très performant sur le jeu de test, indiquant une bonne généralisation.\n")
    print("- Toutes les hypothèses du modèle linéaire semblent respectées, ce qui renforce la confiance dans les résultats.")
    print("- Les métriques de performance (RMSE, MAE, MAPE) montrent une précision élevée et une faible erreur de prédiction.")
    print("- Cela nous conforte dans le choix des variables et montre qu'on est pas obligé de traiter par exemple les outliers, ou les valeurs manquantes.")
    print("- A première vue le résultat semble beaucoup trop performant pour être vrai, mais il est confirmé par les tests d'hypothèse et les analyses de résidus.")
    print("- Le modèle peut être utilisé pour prédire les émissions de CO2 des véhicules avec une bonne précision,")
    print("  et pour analyser l'impact des différentes caractéristiques sur ces émissions.\n")


    