"""
Medical Cost - Szablon rozwiązania
==================================

Autor: [Imię Nazwisko]
Numer albumu: [XXXXX]

INSTRUKCJE:
-----------
1. Uzupełnij wszystkie funkcje oznaczone "TODO"
2. Nie zmieniaj sygnatur funkcji (nazw, parametrów, typów zwracanych)
3. Nie zmieniaj nazwy pliku
4. Możesz dodawać własne funkcje pomocnicze (zaczynające się od _)
5. Wymagane biblioteki: pandas, numpy, scikit-learn, matplotlib, seaborn

DANE WEJŚCIOWE:
---------------
Plik insurance.csv zawiera następujące kolumny:
- age:      int   - wiek ubezpieczonego
- sex:      str   - płeć (female, male)
- bmi:      float - wskaźnik masy ciała (Body Mass Index)
- children: int   - liczba dzieci na utrzymaniu (0-5)
- smoker:   str   - czy osoba pali (yes, no)
- region:   str   - region zamieszkania (northeast, northwest, southeast, southwest)
- charges:  float - ZMIENNA CELU - roczne koszty leczenia (w USD)

Zbiór nie zawiera braków danych ani kolumn wymagających usunięcia.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CZĘŚĆ 1: WCZYTANIE I EKSPLORACJA DANYCH
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Wczytuje dane z pliku CSV.

    Parameters
    ----------
    filepath : str
        Ścieżka do pliku insurance.csv

    Returns
    -------
    pd.DataFrame
        Surowe dane bez żadnych modyfikacji
    """
    # TODO: Zaimplementuj wczytywanie danych
    pass


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca podstawowe informacje o zbiorze danych.

    Returns
    -------
    Dict[str, Any]
        - 'shape': Tuple[int, int]
        - 'dtypes': pd.Series
        - 'missing_values': pd.Series   (liczba braków w każdej kolumnie)
        - 'duplicates_count': int        (liczba zduplikowanych wierszy)
    """
    # TODO: Zaimplementuj
    pass


def get_target_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Zwraca statystyki opisowe zmiennej celu 'charges'.

    Returns
    -------
    Dict[str, float]
        Klucze: 'mean', 'median', 'std', 'min', 'max', 'q1', 'q3'
    """
    # TODO: Zaimplementuj
    pass


def get_feature_info(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """
    Zwraca informacje o wskazanej cesze.

    Returns
    -------
    Dict[str, Any]
        - 'dtype': str
        - 'unique_values': np.ndarray
        - 'unique_count': int
        - 'describe': pd.Series
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 2: ANALIZA EKSPLORACYJNA
# =============================================================================

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zwraca macierz korelacji dla cech numerycznych (age, bmi, children) oraz
    zmiennej celu (charges).

    Uwaga: korelacja Pearsona dotyczy wyłącznie cech numerycznych. Zależność
    cech kategorycznych (sex, smoker, region) ze zmienną celu zbadasz w
    get_group_statistics().
    """
    # TODO: Zaimplementuj
    pass


def get_group_statistics(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Zwraca średnią wartość 'charges' w podziale na grupy kategoryczne.

    Returns
    -------
    Dict[str, pd.Series]
        - 'by_smoker': średnia charges wg palenia
        - 'by_region': średnia charges wg regionu
        - 'by_sex':    średnia charges wg płci
    """
    # TODO: Zaimplementuj
    pass


def check_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Bada rozkład zmiennej celu przed i po transformacji logarytmicznej.

    Returns
    -------
    Dict[str, float]
        - 'skewness', 'kurtosis'          - dla charges
        - 'log_skewness', 'log_kurtosis'  - dla log(1 + charges)
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 3: PRZYGOTOWANIE DANYCH
# =============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rozdziela dane na cechy (X) i zmienną celu (y = charges).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    # TODO: Zaimplementuj
    pass


def get_column_lists() -> Dict[str, List[str]]:
    """
    Zwraca podział kolumn na kategoryczne i numeryczne.

    Returns
    -------
    Dict[str, List[str]]
        - 'categorical': ['sex', 'smoker', 'region']
        - 'numerical':   ['age', 'bmi', 'children']
    """
    # TODO: Zaimplementuj
    pass


def create_preprocessor() -> ColumnTransformer:
    """
    Tworzy preprocessor transformujący dane.

    Używa:
    - OneHotEncoder(drop='first', sparse_output=False) dla zmiennych kategorycznych
    - StandardScaler() dla zmiennych numerycznych

    Wskazówka: użyj get_column_lists() do pobrania list kolumn.
    """
    # TODO: Zaimplementuj
    pass


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Dzieli dane na zbiór treningowy i testowy.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 4: BUDOWA MODELI
# =============================================================================

def get_models() -> Dict[str, Any]:
    """
    Zwraca słownik modeli regresyjnych:

    - 'LinearRegression': LinearRegression()
    - 'Ridge':            Ridge(alpha=1.0)
    - 'Lasso':            Lasso(alpha=0.1)
    - 'DecisionTree':     DecisionTreeRegressor(max_depth=5, random_state=42)
    - 'RandomForest':     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    - 'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                    max_depth=3, random_state=42)
    """
    # TODO: Zaimplementuj
    pass


def train_model(model: Any, X_train: np.ndarray, y_train: pd.Series) -> Any:
    """
    Trenuje pojedynczy model na przetworzonych cechach i zwraca go.
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 5: OCENA MODELI
# =============================================================================

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Oblicza metryki regresji.

    Returns
    -------
    Dict[str, float]
        Klucze: 'MAE', 'MSE', 'RMSE', 'R2'
    """
    # TODO: Zaimplementuj
    pass


def evaluate_all_models(models: Dict[str, Any],
                        X_train: np.ndarray, X_test: np.ndarray,
                        y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Trenuje i ocenia wszystkie modele na zbiorze treningowym i testowym.

    Returns
    -------
    pd.DataFrame
        Kolumny: 'Model', 'MAE_train', 'MSE_train', 'RMSE_train', 'R2_train',
                 'MAE_test', 'MSE_test', 'RMSE_test', 'R2_test'
    """
    # TODO: Zaimplementuj
    pass


def get_best_model_name(results_df: pd.DataFrame, metric: str = 'R2_test') -> str:
    """
    Zwraca nazwę najlepszego modelu według wskazanej metryki.
    (dla R2 - wartość najwyższa, dla MAE/MSE/RMSE - najniższa)
    """
    # TODO: Zaimplementuj
    pass


def get_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    """Zwraca predykcje modelu dla podanych cech."""
    # TODO: Zaimplementuj
    pass


def calculate_residuals(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Zwraca reszty (y_true - y_pred)."""
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 6: INTERPRETACJA MODELU
# =============================================================================

def get_feature_importance(model: RandomForestRegressor,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Zwraca ważność cech modelu RandomForest.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'importance'), posortowane malejąco po 'importance'.
    """
    # TODO: Zaimplementuj
    pass


def get_linear_coefficients(model: LinearRegression,
                            feature_names: List[str]) -> pd.DataFrame:
    """
    Zwraca współczynniki regresji liniowej.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'coefficient'), posortowane malejąco po |coefficient|.
    """
    # TODO: Zaimplementuj
    pass


def get_top_features(importance_df: pd.DataFrame, n: int = 3) -> List[str]:
    """Zwraca listę n najważniejszych cech."""
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 7: WIZUALIZACJE
# =============================================================================

def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram zmiennej celu (charges)."""
    # TODO: Zaimplementuj
    pass


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Heatmapa macierzy korelacji."""
    # TODO: Zaimplementuj
    pass


def plot_feature_vs_target(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Wykres zależności cechy od zmiennej celu (scatter lub box plot)."""
    # TODO: Zaimplementuj
    pass


def plot_group_statistics(group_stats: Dict[str, pd.Series]) -> plt.Figure:
    """Trzy wykresy słupkowe: średnie charges wg smoker, region, sex."""
    # TODO: Zaimplementuj
    pass


def plot_predictions_vs_actual(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """Wykres punktowy predykcji vs wartości rzeczywiste z linią y=x."""
    # TODO: Zaimplementuj
    pass


def plot_residuals(y_pred: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    """Wykres reszt względem predykcji."""
    # TODO: Zaimplementuj
    pass


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Wykres słupkowy ważności cech."""
    # TODO: Zaimplementuj
    pass


def plot_coefficients(coef_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Wykres słupkowy współczynników regresji."""
    # TODO: Zaimplementuj
    pass


# =============================================================================
# GENEROWANIE WYKRESÓW
# =============================================================================

def generate_all_figures(df: pd.DataFrame, results: Dict[str, Any],
                         output_dir: str = "figures") -> Dict[str, str]:
    """
    Generuje i zapisuje wszystkie wykresy do wskazanego folderu.

    Returns
    -------
    Dict[str, str]
        Słownik {nazwa_wykresu: ścieżka_do_pliku}
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# FUNKCJA GŁÓWNA
# =============================================================================

def run_full_analysis(filepath: str, generate_figures: bool = True,
                      figures_dir: str = "figures") -> Dict[str, Any]:
    """
    Uruchamia pełną analizę i zwraca słownik ze wszystkimi wynikami.

    Słownik wynikowy (w nawiasach funkcja zwracająca dany element):

    - 'basic_info': Dict                  <- get_basic_info()
    - 'target_stats': Dict                <- get_target_statistics()
    - 'correlation_matrix': pd.DataFrame  <- get_correlation_matrix()
    - 'group_stats': Dict                 <- get_group_statistics()
    - 'distribution_stats': Dict          <- check_distribution()
    - 'results_df': pd.DataFrame          <- evaluate_all_models()
    - 'best_model_name': str              <- get_best_model_name()
    - 'best_model': Any                   <- wytrenowany najlepszy model
    - 'feature_importance': pd.DataFrame  <- get_feature_importance() [RandomForest]
    - 'linear_coefficients': pd.DataFrame <- get_linear_coefficients() [LinearRegression]
    - 'top_features_rf': List[str]        <- get_top_features(feature_importance)
    - 'top_features_lr': List[str]        <- get_top_features(linear_coefficients)
    - 'y_test': pd.Series                 <- z split_data()
    - 'y_pred_test': np.ndarray           <- get_predictions(best_model, X_test)
    - 'residuals': np.ndarray             <- calculate_residuals()
    - 'saved_figures': Dict[str, str]     <- generate_all_figures() [jeśli generate_figures=True]

    Sugerowana kolejność kroków:
    1.  load_data() -> df
    2.  get_basic_info(), get_target_statistics()
    3.  get_correlation_matrix(), get_group_statistics(), check_distribution()
    4.  prepare_features() -> X, y
    5.  get_column_lists()
    6.  split_data() -> X_train, X_test, y_train, y_test
    7.  create_preprocessor() -> preprocessor
    8.  preprocessor.fit_transform(X_train), preprocessor.transform(X_test)
    9.  preprocessor.get_feature_names_out() -> nazwy cech po preprocessingu
    10. get_models()
    11. evaluate_all_models() -> results_df
    12. get_best_model_name()
    13. wytrenuj best_model oraz RandomForest i LinearRegression (do interpretacji)
    14. get_feature_importance(), get_linear_coefficients()
    15. get_top_features()
    16. get_predictions(), calculate_residuals()
    17. generate_all_figures() [opcjonalnie]
    """
    # TODO: Zaimplementuj łącząc wszystkie powyższe funkcje
    pass


# =============================================================================
# PUNKT WEJŚCIA
# =============================================================================

if __name__ == "__main__":
    # Przykład użycia:
    # results = run_full_analysis("insurance.csv")
    # print(results['results_df'])
    pass
