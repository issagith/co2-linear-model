import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OrdinaryLeastSquares:
    def __init__(self, intercept=True):
        """
        Initialise le modèle OLS.

        Arguments :
        intercept (bool) : si True, ajoute une constante au modèle.
        """
        self.intercept = intercept
        self.coefficients = None

    def fit(self, X, y):
        """
        Entraîne le modèle OLS sur les données données.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calcul des coefficients via la formule des moindres carrés
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les nouvelles données.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).

        Retour :
        np.ndarray : Prédictions.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.coefficients

    def get_coeffs(self):
        """
        Retourne les coefficients estimés du modèle.

        Retour :
        np.ndarray : Coefficients estimés.
        """
        return self.coefficients

    def r2_score(self, X, y):
        """
        Calcule le coefficient de détermination R².

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : R².
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def hypothesis_test(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Calcul des résidus sur X SANS ajouter manuellement l'intercept, 
        # puisque predict() va le faire :
        y_pred = self.predict(X)
        residuals = y - y_pred

        # Maintenant, pour le calcul des erreurs standards, on doit utiliser la même matrice X 
        # que dans fit() : c'est-à-dire avec l'intercept si self.intercept est True.
        # Recréons X_aug l'équivalent de X utilisé dans fit().
        if self.intercept:
            X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_aug = X

        n, p = X_aug.shape
        sigma2 = np.sum(residuals ** 2) / (n - p)
        cov_matrix = sigma2 * np.linalg.inv(X_aug.T @ X_aug)
        std_errors = np.sqrt(np.diag(cov_matrix))

        t_stats = self.coefficients / std_errors
        p_values = 2 * (1 - self._approx_cdf_t(np.abs(t_stats), df=n - p))

        results = pd.DataFrame({
            'Coefficients': self.coefficients,
            'Std Errors': std_errors,
            't-Statistic': t_stats,
            'p-Value': p_values
        })
        return results

    def _approx_cdf_t(self, t, df):
        """
        Approximation de la fonction de distribution cumulée pour la statistique t.

        Arguments :
        t (float ou np.ndarray) : Valeur de la statistique t.
        df (int) : Degrés de liberté.

        Retour :
        np.ndarray : Valeur approchée de la CDF.
        """
        return 1 - (1 / (1 + (t ** 2 / df)) ** ((df + 1) / 2))

    def _approx_cdf_f(self, f, df1, df2):
        """
        Approximation de la fonction de distribution cumulée pour la statistique F.

        Arguments :
        f (float ou np.ndarray) : Valeur de la statistique F.
        df1 (int) : Degrés de liberté du numérateur.
        df2 (int) : Degrés de liberté du dénominateur.

        Retour :
        np.ndarray : Valeur approchée de la CDF.
        """
        return 1 - (1 / (1 + (f * df1 / df2)) ** ((df2 + df1 - 1) / 2))

    def plot_residuals(self, X, y):
        """
        Trace un graphique des résidus pour évaluer la qualité du modèle.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Valeurs prédites')
        plt.ylabel('Résidus')
        plt.title('Graphique des résidus')
        plt.show()

    def confidence_intervals(self, X, y, confidence=0.95):
        """
        Calcule les intervalles de confiance pour les coefficients estimés.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.
        confidence (float) : Niveau de confiance.

        Retour :
        pd.DataFrame : Intervalles de confiance.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        y_pred = self.predict(X)
        residuals = y - y_pred
        n, p = X.shape
        sigma2 = np.sum(residuals ** 2) / (n - p)
        cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
        std_errors = np.sqrt(np.diag(cov_matrix))

        t_value = self._approx_t_value(confidence, df=n - p)
        lower_bounds = self.coefficients - t_value * std_errors
        upper_bounds = self.coefficients + t_value * std_errors

        return pd.DataFrame({
            'Coefficients': self.coefficients,
            'Lower Bound': lower_bounds,
            'Upper Bound': upper_bounds
        })

    def _approx_t_value(self, confidence, df):
        """
        Approximation de la valeur critique t pour un niveau de confiance donné.

        Arguments :
        confidence (float) : Niveau de confiance.
        df (int) : Degrés de liberté.

        Retour :
        float : Valeur critique t.
        """
        alpha = 1 - confidence
        return np.sqrt(df / (1 - alpha))
    
    def compare_predictions(self, X, y):
        """
        Compare les valeurs prédites avec les valeurs réelles pour chaque ligne.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        pd.DataFrame : Comparaison des valeurs réelles et prédites.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)
        comparison = pd.DataFrame({
            'Valeur réelle': y,
            'Valeur prédite': y_pred
        })
        return comparison
    
    def MSE(self, X, y):
        """
        Calcule l'erreur quadratique moyenne (MSE).

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : MSE.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def RMSE(self, X, y):
        """
        Calcule l'erreur quadratique moyenne (RMSE).

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : RMSE.
        """
        return np.sqrt(self.MSE(X, y))
    
    def MAE(self, X, y):
        """
        Calcule l'erreur absolue moyenne (MAE).

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : MAE.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))
    
    def MAPE(self, X, y):
        """
        Calcule le pourcentage d'erreur absolue moyenne (MAPE).

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : MAPE.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        y_pred = self.predict(X)
        return np.mean(np.abs((y - y_pred) / y)) * 100

    def durbin_watson_test(self, X, y):
        """
        Calcule le test de Durbin-Watson pour vérifier l'indépendance des erreurs.

        Arguments :
        X (np.ndarray ou pd.DataFrame) : Matrice des caractéristiques (features).
        y (np.ndarray ou pd.Series) : Vecteur cible.

        Retour :
        float : Statistique de Durbin-Watson.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        residuals = y - self.predict(X)
        diff_residuals = np.diff(residuals)
        dw_statistic = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)
        return dw_statistic