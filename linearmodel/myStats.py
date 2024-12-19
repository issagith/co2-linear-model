import numpy as np 
import pandas as pd
import os 
import matplotlib.pyplot as plt

def load_data(file_path):
    """Loads the dataset from the given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path and try again.")
    return pd.read_csv(file_path)

def data_overview(data):
    """Returns an overview of the dataset."""
    return data.head()

def columns_overview(data):
    """Returns an overview of the columns in the dataset."""
    return data.columns

def numerical_summary(data):
    """Returns summary statistics for numerical columns."""
    summary = data.describe()
    return summary

def categorical_summary(data):
    """Returns summary statistics for categorical columns."""
    summary = data.describe(include=['object'])
    return summary

def missing_values_analysis(data):
    """returns missing values in the dataset."""
    missing_values = data.isnull().sum()
    return missing_values

def delete_missing_values(data):
    """Deletes rows with missing values."""
    data.dropna(inplace=True)
    return data

def unique_values_analysis(data):
    """Analyzes unique values for categorical columns."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    unique_values = {col: data[col].unique() for col in categorical_cols}
    return unique_values

def correlation_analysis(data):
    """Analyzes correlations between numerical columns."""
    numerical_data = data.select_dtypes(include=['number'])  # Select only numerical columns
    corr_matrix = numerical_data.corr()
    return corr_matrix


def outliers_count(data):
    """Returns the count of outliers in numerical columns."""
    numerical_data = data.select_dtypes(include=['number'])
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).sum()
    return outliers

def vehcile_outliers(data):
    """Returns the rows with outliers in numerical columns and the columns that have outliers along with their quartile values."""
    numerical_data = data.select_dtypes(include=['number'])
    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR)))
    outlier_rows = data[outliers.any(axis=1)].copy()
    outlier_columns = outliers.columns[outliers.any()].tolist()
    outlier_rows['outlier_columns'] = outliers[outliers.any(axis=1)].apply(lambda row: [col for col in outlier_columns if row[col]], axis=1)
    outlier_rows['Q1'] = outlier_rows.apply(lambda row: {col: Q1[col] for col in row['outlier_columns']}, axis=1)
    outlier_rows['Q3'] = outlier_rows.apply(lambda row: {col: Q3[col] for col in row['outlier_columns']}, axis=1)
    outlier_rows['outlier_values'] = outlier_rows.apply(lambda row: {col: row[col] for col in row['outlier_columns']}, axis=1)
    return outlier_rows

def variance_analysis(data):
    """Returns the variance of numerical columns."""
    numerical_data = data.select_dtypes(include=['number'])
    variance = numerical_data.var()
    return variance

def frequency_analysis(data):
    """Returns the frequency of values for categorical columns."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    frequency = {col: data[col].value_counts() for col in categorical_cols}
    return frequency

def main():
    # Load the dataset
    file_path = "../data/vehicles.csv"
    data = load_data(file_path)

    # Overview of the columns
    print("\n--- APERCU DES COLONNES ---\n")
    print(columns_overview(data))

    # Overview of the dataset
    print("\n--- APERCU DES DONNEES ---\n")
    print(data_overview(data))

    # Summary statistics
    print("\n--- STATISTIQUES NUMERIQUE ---\n")
    print(numerical_summary(data))
    
    print("\n--- STATISTIQUES CATEGORIQUE ---\n")
    print(categorical_summary(data))

    # Missing values analysis
    print("\n  --- ANALYSE DES VALEURS MANQUANTES ---\n")
    print(missing_values_analysis(data))

    # Unique values analysis
    print("\n  --- ANALYSE DES VALEURS UNIQUES ---\n")
    print(unique_values_analysis(data))

    # Correlation analysis
    print("\n  --- ANALYSE DE LA CORRELATION ---\n")
    print(correlation_analysis(data))

    # Variance analysis
    print("\n  --- ANALYSE DE LA VARIANCE ---\n")
    print(variance_analysis(data))

    # Frequency analysis
    print("\n  --- ANALYSE DE LA FREQUENCE ---\n")
    print(frequency_analysis(data))

    # Outliers
    print("\n  --- ANALYSE DU NOMBRE DE VALEURS AYANT DES VALEURS EXTREMES ---\n")
    print(outliers_count(data))

    print("\n  --- ANALYSE DES VALEURS EXTREMES ---\n")
    print(vehcile_outliers(data))

    print("\n Toutes ces données seront analysées en détail pour l'entraînement optimal du modèle de régression linéaire dans le fichier main_linear.py.\n")
    print("Ici, on se contente d'afficher les résultats des analyses statistiques.\n")

if __name__ == "__main__":
    main()
