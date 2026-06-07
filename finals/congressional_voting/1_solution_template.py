"""
Congressional Voting Records - Szablon rozwiązania
==================================================

Autor: Paweł Jancewicz
Numer albumu: pd5003

INSTRUKCJE:
-----------
1. Uzupełnij wszystkie funkcje oznaczone "TODO"
2. Nie zmieniaj sygnatur funkcji (nazw, parametrów, typów zwracanych)
3. Nie zmieniaj nazwy pliku
4. Możesz dodawać własne funkcje pomocnicze (zaczynające się od _)
5. Wymagane biblioteki: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

DANE WEJŚCIOWE:
---------------
Plik house-votes-84.data NIE zawiera nagłówków - nazwy kolumn przypisz z COLUMN_NAMES.
Kolumna 'class' to zmienna celu (democrat / republican).
Pozostałe 16 kolumn to głosowania, każde o wartościach: 'y' (za), 'n' (przeciw),
'?' (pozycja nieujawniona).

WAŻNE - braki danych:
Symbol '?' występuje we WSZYSTKICH 16 kolumnach głosowań i NIE jest klasycznym brakiem
(oznacza nieujawnioną pozycję, np. wstrzymanie się). Przemyśl strategię jego obsługi.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from scipy.stats import chi2_contingency


# =============================================================================
# STAŁE
# =============================================================================

COLUMN_NAMES = [
    'class', 'handicapped-infants', 'water-project-cost-sharing',
    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
    'el-salvador-aid', 'religious-groups-in-schools',
    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile',
    'immigration', 'synfuels-corporation-cutback', 'education-spending',
    'superfund-right-to-sue', 'crime', 'duty-free-exports',
    'export-administration-act-south-africa'
]


# =============================================================================
# CZĘŚĆ 1: WCZYTANIE I EKSPLORACJA DANYCH
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Wczytuje dane i przypisuje nazwy kolumn z COLUMN_NAMES.
    (plik nie zawiera nagłówka)
    """
    # TODO: Zaimplementuj
    pass


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca:
    - 'shape': Tuple[int, int]
    - 'dtypes': pd.Series
    - 'missing_values': pd.Series          (liczba wartości NaN w każdej kolumnie)
    - 'missing_question_marks': pd.Series   (liczba '?' w każdej kolumnie)
    """
    # TODO: Zaimplementuj
    pass


def get_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca:
    - 'counts': pd.Series         (liczność każdej klasy)
    - 'percentages': pd.Series    (udział procentowy każdej klasy)
    - 'is_balanced': bool         (True, jeśli udział KAŻDEJ klasy mieści się w [40%, 60%])
    """
    # TODO: Zaimplementuj
    pass


def get_feature_info(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """
    Zwraca:
    - 'unique_values': np.ndarray
    - 'unique_count': int
    - 'value_counts': pd.Series
    - 'missing_count': int          (liczba '?' w danej kolumnie)
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 2: ANALIZA EKSPLORACYJNA
# =============================================================================

def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Oblicza współczynnik V Craméra dla dwóch zmiennych kategorycznych.

    Wskazówka: zbuduj tabelę krzyżową (pd.crosstab), policz statystykę chi2
    (chi2_contingency), a następnie V = sqrt( (chi2/n) / min(k-1, r-1) ).
    """
    # TODO: Zaimplementuj
    pass


def get_cramers_v_for_features(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """
    Dla każdej cechy liczy V Craméra względem kolumny 'class'.
    Jeśli features=None, użyj wszystkich kolumn poza 'class'.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'cramers_v'), posortowane malejąco po 'cramers_v'.
    """
    # TODO: Zaimplementuj
    pass


def get_feature_class_crosstab(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Zwraca tabelę krzyżową cechy względem klasy, znormalizowaną po kolumnach
    (normalize='columns').
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 3: PRZYGOTOWANIE DANYCH
# =============================================================================

def handle_missing_values(df: pd.DataFrame, strategy: str = 'keep_as_category') -> pd.DataFrame:
    """
    Obsługuje wartości '?' według wybranej strategii:

    - 'keep_as_category' : '?' pozostaje samodzielną kategorią (domyślnie, zalecane)
    - 'drop_rows'        : usuwa wiersze zawierające jakiekolwiek '?'
    - 'impute_mode'      : zastępuje '?' najczęstszą wartością w danej kolumnie

    Returns
    -------
    pd.DataFrame
        Dane po obsłudze braków (nie modyfikuj oryginalnego df).
    """
    # TODO: Zaimplementuj
    pass


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rozdziela dane na cechy (X) i zmienną celu (y).
    Kodowanie celu: democrat=0, republican=1.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    # TODO: Zaimplementuj
    pass


def encode_features(X: pd.DataFrame, method: str = 'onehot') -> np.ndarray:
    """
    Koduje cechy kategoryczne.

    - 'onehot':  OneHotEncoder (domyślnie)
    - 'ordinal': OrdinalEncoder (kodowanie całkowitoliczbowe)

    Returns
    -------
    np.ndarray
        Zakodowane cechy.
    """
    # TODO: Zaimplementuj
    pass


def get_feature_names_after_encoding(X: pd.DataFrame) -> List[str]:
    """
    Zwraca nazwy cech po kodowaniu OneHot.

    Wskazówka: OneHotEncoder().fit(X).get_feature_names_out()
    """
    # TODO: Zaimplementuj
    pass


def split_data(X: np.ndarray, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Dzieli dane na zbiór treningowy i testowy ZE STRATYFIKACJĄ (stratify=y).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 4: BUDOWA MODELI
# =============================================================================

def get_models() -> Dict[str, Any]:
    """
    Zwraca słownik modeli klasyfikacyjnych:

    - 'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    - 'DecisionTree':       DecisionTreeClassifier(max_depth=4, random_state=42)
    - 'RandomForest':       RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    - 'GradientBoosting':   GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                                       max_depth=2, random_state=42)
    - 'NaiveBayes':         CategoricalNB()

    UWAGA: Hiperparametry celowo ograniczone, aby uwidocznić różnice między modelami.
    """
    # TODO: Zaimplementuj
    pass


def train_model(model: Any, X_train: np.ndarray, y_train: pd.Series) -> Any:
    """Trenuje pojedynczy model i zwraca go."""
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 5: OCENA MODELI
# =============================================================================

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray,
                      y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Oblicza metryki klasyfikacji.

    Returns
    -------
    Dict[str, float]
        - 'accuracy', 'precision', 'recall', 'f1'
        - 'auc': float jeśli podano y_prob, w przeciwnym razie None
    """
    # TODO: Zaimplementuj
    pass


def evaluate_all_models(models: Dict[str, Any],
                        X_train: np.ndarray, X_test: np.ndarray,
                        y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Trenuje i ocenia wszystkie modele.

    Returns
    -------
    pd.DataFrame
        Kolumny: 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'
    """
    # TODO: Zaimplementuj
    pass


def get_best_model_name(results_df: pd.DataFrame, metric: str = 'F1') -> str:
    """Zwraca nazwę najlepszego modelu według wskazanej metryki (wartość najwyższa)."""
    # TODO: Zaimplementuj
    pass


def get_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Zwraca macierz pomyłek 2x2."""
    # TODO: Zaimplementuj
    pass


def get_roc_curve_data(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Zwraca dane krzywej ROC.

    Returns
    -------
    Dict[str, Any]
        - 'fpr': np.ndarray
        - 'tpr': np.ndarray
        - 'thresholds': np.ndarray
        - 'auc': float
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 6: INTERPRETACJA MODELU
# =============================================================================

def get_feature_importance(model: RandomForestClassifier,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Zwraca ważność cech modelu RandomForest.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'importance'), posortowane malejąco.
    """
    # TODO: Zaimplementuj
    pass


def get_top_features(importance_df: pd.DataFrame, n: int = 5) -> List[str]:
    """Zwraca listę n najważniejszych cech."""
    # TODO: Zaimplementuj
    pass


def analyze_misclassification_risk(cm: np.ndarray) -> Dict[str, Any]:
    """
    Analiza błędów klasyfikacji dla macierzy 2x2.
    Przyjmij klasę pozytywną = 1 (republican); cm = [[TN, FP], [FN, TP]].

    Returns
    -------
    Dict[str, Any]
        - 'false_positives': int
        - 'false_negatives': int
        - 'fn_rate': float          (FN / (FN + TP))
        - 'fp_rate': float          (FP / (FP + TN))
        - 'error_assessment': str   (krótki, neutralny opis przewagi błędów FN/FP)
    """
    # TODO: Zaimplementuj
    pass


# =============================================================================
# CZĘŚĆ 7: WIZUALIZACJE
# =============================================================================

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Wykres słupkowy rozkładu klas."""
    # TODO: Zaimplementuj
    pass


def plot_feature_vs_class(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Wykres słupkowy wartości cechy w podziale na klasy."""
    # TODO: Zaimplementuj
    pass


def plot_cramers_v_ranking(cramers_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Poziomy wykres słupkowy rankingu Cramér's V."""
    # TODO: Zaimplementuj
    pass


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None) -> plt.Figure:
    """Heatmapa macierzy pomyłek."""
    # TODO: Zaimplementuj
    pass


def plot_roc_curve(roc_data: Dict[str, Any]) -> plt.Figure:
    """Wykres krzywej ROC z przekątną odniesienia i wartością AUC."""
    # TODO: Zaimplementuj
    pass


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Poziomy wykres słupkowy ważności cech."""
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
    - 'class_distribution': Dict          <- get_class_distribution()
    - 'cramers_v': pd.DataFrame           <- get_cramers_v_for_features()
    - 'results_df': pd.DataFrame          <- evaluate_all_models()
    - 'best_model_name': str              <- get_best_model_name()
    - 'best_model': Any                   <- wytrenowany najlepszy model
    - 'confusion_matrix': np.ndarray      <- get_confusion_matrix()
    - 'roc_data': Dict                    <- get_roc_curve_data()
    - 'feature_importance': pd.DataFrame  <- get_feature_importance() [RandomForest]
    - 'top_features': List[str]           <- get_top_features()
    - 'misclassification_risk': Dict      <- analyze_misclassification_risk()
    - 'y_test': pd.Series                 <- z split_data()
    - 'y_pred': np.ndarray                <- predykcje najlepszego modelu
    - 'y_prob': np.ndarray                <- prawdopodobieństwa klasy pozytywnej
    - 'saved_figures': Dict[str, str]     <- generate_all_figures() [jeśli generate_figures=True]

    Sugerowana kolejność kroków:
    1.  load_data()
    2.  get_basic_info(), get_class_distribution()
    3.  get_cramers_v_for_features()
    4.  handle_missing_values()
    5.  prepare_features() -> X, y
    6.  get_feature_names_after_encoding(X)
    7.  encode_features(X) -> X_enc
    8.  split_data(X_enc, y)
    9.  get_models()
    10. evaluate_all_models() -> results_df
    11. get_best_model_name()
    12. wytrenuj best_model oraz RandomForest (do ważności cech)
    13. predykcje + predict_proba dla best_model
    14. get_confusion_matrix(), get_roc_curve_data()
    15. get_feature_importance(), get_top_features(), analyze_misclassification_risk()
    16. generate_all_figures() [opcjonalnie]
    """
    # TODO: Zaimplementuj łącząc wszystkie powyższe funkcje
    pass


# =============================================================================
# PUNKT WEJŚCIA
# =============================================================================

if __name__ == "__main__":
    # Przykład użycia:
    # results = run_full_analysis("house-votes-84.data")
    # print(results['results_df'])
    pass
