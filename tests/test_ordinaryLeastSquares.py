import pytest
import numpy as np
import pandas as pd
from linearmodel.ordinaryLeastSquares import OrdinaryLeastSquares

@pytest.fixture
def sample_regression_data():
    # Génération de données factices pour un petit problème de régression linéaire
    np.random.seed(0)
    X = pd.DataFrame({'X1': np.random.rand(100), 'X2': np.random.rand(100)})
    # Véritable relation linéaire : y = 2 + 3*X1 - 1*X2 + petit bruit
    y = 2 + 3*X['X1'] - 1*X['X2'] + np.random.randn(100)*0.1
    return X, y

def test_fit_predict(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Vérifions que la prédiction est proche de y (R² élevé)
    r2 = model.r2_score(X, y)
    assert r2 > 0.9

def test_get_coeffs(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares(intercept=True)
    model.fit(X, y)
    coeffs = model.get_coeffs()
    # On s'attend à quelque chose proche de [2, 3, -1]
    assert len(coeffs) == 3
    assert abs(coeffs[0] - 2) < 0.2  # intercept
    assert abs(coeffs[1] - 3) < 0.2  # pour X1
    assert abs(coeffs[2] + 1) < 0.2  # pour X2 (notez +1 car on s'attend à -1)


def test_r2_score(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    r2 = model.r2_score(X, y)
    assert 0 <= r2 <= 1


def test_MSE(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    mse = model.MSE(X, y)
    # Comme les données sont propres et presque parfaitement linéaires, le MSE doit être faible
    assert mse < 0.05


def test_RMSE(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    rmse = model.RMSE(X, y)
    assert rmse < 0.3


def test_MAE(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    mae = model.MAE(X, y)
    # Erreur absolue moyenne également faible
    assert mae < 0.25


def test_MAPE(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    mape = model.MAPE(X, y)
    # Très faible MAPE attendu
    assert mape < 5.0


def test_hypothesis_test(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    results = model.hypothesis_test(X, y)
    # Vérification du format de sortie
    assert isinstance(results, pd.DataFrame)
    for col in ['Coefficients', 'Std Errors', 't-Statistic', 'p-Value']:
        assert col in results.columns


def test_durbin_watson_test(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    dw = model.durbin_watson_test(X, y)
    # Le Durbin-Watson est souvent proche de 2 pour des erreurs indépendantes
    assert 0 <= dw <= 4


def test_confidence_intervals(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    ci = model.confidence_intervals(X, y, confidence=0.95)
    # Vérification du DataFrame de CI
    assert isinstance(ci, pd.DataFrame)
    for col in ['Coefficients', 'Lower Bound', 'Upper Bound']:
        assert col in ci.columns


def test_compare_predictions(sample_regression_data):
    X, y = sample_regression_data
    model = OrdinaryLeastSquares()
    model.fit(X, y)
    comp = model.compare_predictions(X, y)
    # comp doit être un DataFrame avec 2 colonnes : 'Valeur réelle' et 'Valeur prédite'
    assert isinstance(comp, pd.DataFrame)
    assert 'Valeur réelle' in comp.columns
    assert 'Valeur prédite' in comp.columns
    assert len(comp) == len(X)
