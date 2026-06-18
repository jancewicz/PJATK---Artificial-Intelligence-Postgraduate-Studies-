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
from typing import Tuple, Dict, List, Any, Literal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from scipy.stats import chi2_contingency


# =============================================================================
# STAŁE
# =============================================================================

COLUMN_NAMES = [
    "class",
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa",
]


# =============================================================================
# CZĘŚĆ 1: WCZYTANIE I EKSPLORACJA DANYCH
# =============================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """
    Wczytuje dane i przypisuje nazwy kolumn z COLUMN_NAMES.
    (plik nie zawiera nagłówka)
    """
    return pd.read_csv(filepath, names=COLUMN_NAMES)


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca:
    - 'shape': Tuple[int, int]
    - 'dtypes': pd.Series
    - 'missing_values': pd.Series          (liczba wartości NaN w każdej kolumnie)
    - 'missing_question_marks': pd.Series   (liczba '?' w każdej kolumnie)
    """

    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing_values": df.isna().sum(),
        "missing_question_marks": df.apply(lambda x: (x == "?").sum()),
    }


def get_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca:
    - 'counts': pd.Series         (liczność każdej klasy)
    - 'percentages': pd.Series    (udział procentowy każdej klasy)
    - 'is_balanced': bool         (True, jeśli udział KAŻDEJ klasy mieści się w [40%, 60%])
    """

    counts = df.apply(lambda x: x.value_counts()).T.stack().dropna()
    percentages = (
        df.apply(lambda x: (x.value_counts(normalize=True) * 100))
        .T.stack()
        .dropna()
        .round(decimals=2)
    )
    is_balanced = (percentages > 60).groupby(level=0).any()

    return {
        "counts": counts,
        "percentages": percentages,
        "is_balanced": is_balanced,
    }


def get_feature_info(df: pd.DataFrame, feature: str) -> Dict[str, Any]:
    """
    Zwraca:
    - 'unique_values': np.ndarray
    - 'unique_count': int
    - 'value_counts': pd.Series
    - 'missing_count': int          (liczba '?' w danej kolumnie)
    """
    return {
        "unique_values": df[feature].unique(),
        "unique_count": len(df[feature].unique()),
        "value_counts": df[feature].value_counts(),
        "missing_count": (df[feature] == "?").sum(),
    }


# =============================================================================
# CZĘŚĆ 2: ANALIZA EKSPLORACYJNA
# =============================================================================


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Oblicza współczynnik V Craméra dla dwóch zmiennych kategorycznych.

    Wskazówka: zbuduj tabelę krzyżową (pd.crosstab), policz statystykę chi2
    (chi2_contingency), a następnie V = sqrt( (chi2/n) / min(k-1, r-1) ).
    """
    cross_tab = pd.crosstab(y, x)
    chi2 = chi2_contingency(cross_tab)[0]

    n = cross_tab.sum().sum()
    r, k = cross_tab.shape

    return np.sqrt((chi2 / n) / min(k - 1, r - 1))


def get_cramers_v_for_features(
    df: pd.DataFrame, features: List[str] = None
) -> pd.DataFrame:
    """
    Dla każdej cechy liczy V Craméra względem kolumny 'class'.
    Jeśli features=None, użyj wszystkich kolumn poza 'class'.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'cramers_v'), posortowane malejąco po 'cramers_v'.
    """
    cramers_df = pd.DataFrame(columns=["feature", "cramers_v"])
    y = df["class"]

    if features is None:
        df_no_labels = df.drop(labels=["class"], axis=1)
        for idx, feature in enumerate(df_no_labels.columns):
            cramers_df.loc[idx] = [
                feature,
                calculate_cramers_v(df_no_labels[feature], y=y),
            ]
    else:
        for idx, feature in enumerate(features):
            cramers_df.loc[idx] = [feature, calculate_cramers_v(df[feature], y=y)]

    return cramers_df.sort_values(by="cramers_v", ascending=False)


def get_feature_class_crosstab(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Zwraca tabelę krzyżową cechy względem klasy, znormalizowaną po kolumnach
    (normalize='columns').
    """
    return pd.crosstab(df["class"], df[feature], normalize="columns")


# =============================================================================
# CZĘŚĆ 3: PRZYGOTOWANIE DANYCH
# =============================================================================


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "keep_as_category"
) -> pd.DataFrame:
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
    strategies: list[str] = ["keep_as_category", "drop_rows", "impute_mode"]
    if strategy not in strategies:
        raise ValueError("Unknown strategy")

    df_cp = df.copy()

    match strategy:
        case "drop_rows":
            mask = (
                df_cp.astype(str)
                .apply(lambda col: col.str.contains("?", case=False, regex=False))
                .any(axis=1)
            )
            return df_cp[~mask]
        case "impute_mode":
            # helper function to replace each '?' with the most frequent value from column
            def replace_with_the_most_freq_val(col: pd.Series):
                mode_val = col[col != "?"].mode()
                if not mode_val.empty:
                    return col.replace("?", mode_val.iloc[0])
                else:
                    return col

            # Apply replace
            return df_cp.apply(replace_with_the_most_freq_val)
        case _:
            # No changes, keep df as it is with default param 'keep_as_category'
            return df_cp


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rozdziela dane na cechy (X) i zmienną celu (y).
    Kodowanie celu: democrat=0, republican=1.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """

    def encode_political_parties(val: Literal["democrat", "republican"]):
        match val:
            case "democrat":
                return 0
            case "republican":
                return 1
            case _:
                raise ValueError(f"Error: Unknow label {val}")

    X = df.drop(labels=["class"], axis=1)
    y = df["class"].apply(encode_political_parties)

    return X, y


def encode_features(X: pd.DataFrame, method: str = "onehot") -> np.ndarray:
    """
    Koduje cechy kategoryczne.

    - 'onehot':  OneHotEncoder (domyślnie)
    - 'ordinal': OrdinalEncoder (kodowanie całkowitoliczbowe)

    Returns
    -------
    np.ndarray
        Zakodowane cechy.
    """
    if method not in ["onehot", "ordinal"]:
        raise ValueError("Unknown method passed to function")

    match method:
        case "ordinal":
            return OrdinalEncoder().fit_transform(X)
        case _:
            return OneHotEncoder(drop="first", sparse_output=False).fit_transform(X)


def get_feature_names_after_encoding(X: pd.DataFrame) -> List[str]:
    """
    Zwraca nazwy cech po kodowaniu OneHot.

    Wskazówka: OneHotEncoder().fit(X).get_feature_names_out()
    """
    one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False)
    one_hot_encoder.fit(X)

    return one_hot_encoder.get_feature_names_out()


def split_data(
    X: np.ndarray, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Dzieli dane na zbiór treningowy i testowy ZE STRATYFIKACJĄ (stratify=y).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


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
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42
        ),
        "NaiveBayes": CategoricalNB(),
    }


def train_model(model: Any, X_train: np.ndarray, y_train: pd.Series) -> Any:
    """Trenuje pojedynczy model i zwraca go."""
    model.fit(X_train, y_train)
    return model


# =============================================================================
# CZĘŚĆ 5: OCENA MODELI
# =============================================================================


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Oblicza metryki klasyfikacji.

    Returns
    -------
    Dict[str, float]
        - 'accuracy', 'precision', 'recall', 'f1'
        - 'auc': float jeśli podano y_prob, w przeciwnym razie None
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1": f1_score(y_true, y_pred, average="binary"),
        "auc": roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Trenuje i ocenia wszystkie modele.

    Returns
    -------
    pd.DataFrame
        Kolumny: 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'
    """
    df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])

    for idx, (model_name, model) in enumerate(models.items()):
        trained_model = train_model(model, X_train, y_train)

        y_test_pred = trained_model.predict(X_test)

        # TODO check it
        y_test_prob = trained_model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)

        df.loc[idx] = [
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc"],
        ]

    return df


def get_best_model_name(results_df: pd.DataFrame, metric: str = "F1") -> str:
    """Zwraca nazwę najlepszego modelu według wskazanej metryki (wartość najwyższa)."""
    return results_df.at[results_df[metric].idxmax(), "Model"]


def get_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Zwraca macierz pomyłek 2x2."""
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


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

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc(fpr, tpr)}


# =============================================================================
# CZĘŚĆ 6: INTERPRETACJA MODELU
# =============================================================================


def get_feature_importance(
    model: RandomForestClassifier, feature_names: List[str]
) -> pd.DataFrame:
    """
    Zwraca ważność cech modelu RandomForest.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'importance'), posortowane malejąco.
    """
    importance_df: pd.DataFrame = pd.DataFrame(columns=["feature", "importance"])
    importances = model.feature_importances_

    for idx, (importance, feature_name) in enumerate(
        list(zip(importances, feature_names))
    ):
        importance_df.loc[idx] = [feature_name, importance]

    return importance_df.sort_values(by="importance", ascending=False)


def get_top_features(importance_df: pd.DataFrame, n: int = 5) -> List[str]:
    """Zwraca listę n najważniejszych cech."""
    return importance_df["feature"].iloc[:n].tolist()


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
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)

    return {
        "false_positives": fp,
        "false_negatives": fn,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "error_assessment": "Higher FN rate" if fn_rate > fp_rate else "Higher FP rate",
    }


# =============================================================================
# CZĘŚĆ 7: WIZUALIZACJE
# =============================================================================


def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Wykres słupkowy rozkładu klas."""
    fig, ax = plt.subplots()
    sns.countplot(df["class"], ax=ax)
    return fig


def plot_feature_vs_class(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Wykres słupkowy wartości cechy w podziale na klasy."""
    fig, ax = plt.subplots()
    sns.countplot(x=feature, hue="class", data=df, palette="hls", ax=ax)
    return fig


def plot_cramers_v_ranking(cramers_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Poziomy wykres słupkowy rankingu Cramér's V."""

    top_n_cramers_df = cramers_df.iloc[:top_n]
    fig, ax = plt.subplots()
    sns.barplot(x=top_n_cramers_df["cramers_v"], y=top_n_cramers_df["feature"], ax=ax)
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None) -> plt.Figure:
    """Heatmapa macierzy pomyłek."""
    fig, ax = plt.subplots()
    sns.heatmap(data=cm, ax=ax, annot=True, xticklabels=labels, yticklabels=labels)
    return fig


def plot_roc_curve(roc_data: Dict[str, Any]) -> plt.Figure:
    """Wykres krzywej ROC z przekątną odniesienia i wartością AUC."""
    fpr, tpr, threshold, auc_val = roc_data.values()

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Poziomy wykres słupkowy ważności cech."""
    top_n_importance_df = importance_df.iloc[:top_n]

    fig, ax = plt.subplots()
    sns.barplot(
        x=top_n_importance_df["importance"], y=top_n_importance_df["feature"], ax=ax
    )
    return


# =============================================================================
# GENEROWANIE WYKRESÓW
# =============================================================================


def generate_all_figures(
    df: pd.DataFrame, results: Dict[str, Any], output_dir: str = "figures"
) -> Dict[str, str]:
    """
    Generuje i zapisuje wszystkie wykresy do wskazanego folderu.

    Returns
    -------
    Dict[str, str]
        Słownik {nazwa_wykresu: ścieżka_do_pliku}
    """
    # plot_class_distribution
    # plot_feature_vs_class
    # plot_cramers_v_ranking
    # plot_confusion_matrix
    # plot_roc_curve
    # plot_feature_importance
    pass


# =============================================================================
# FUNKCJA GŁÓWNA
# =============================================================================


def run_full_analysis(
    filepath: str, generate_figures: bool = True, figures_dir: str = "figures"
) -> Dict[str, Any]:
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
    df = load_data(filepath)

    info = get_basic_info(df)
    cls_distribution = get_class_distribution(df)
    cramers_v = get_cramers_v_for_features(df)

    df = handle_missing_values(df, strategy="keep_as_category")
    X, y = prepare_features(df)

    features_names = get_feature_names_after_encoding(X)
    X_enc = encode_features(X)

    X_train, X_test, y_train, y_test = split_data(
        X_enc, y, test_size=0.2, random_state=42
    )

    models = get_models()

    results_df = evaluate_all_models(models, X_train, X_test, y_train, y_test)
    best_model_name: str = get_best_model_name(results_df)

    best_model = get_models()[best_model_name]
    best_model.fit(X_train, y_train)

    rf: RandomForestClassifier = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42
    )
    rf.fit(X_train, y_train)

    best_model_y_test_pred = best_model.predict(X_test)
    best_model_y_test_prob = best_model.predict_proba(X_test)[:, 1]

    # TODO check if this variable is needed
    rf_y_test_prob = rf.predict_proba(X_test)[:, 1]

    cm_best_model = get_confusion_matrix(y_test, best_model_y_test_pred)
    roc_curve_data = get_roc_curve_data(y_test, best_model_y_test_prob)

    importance = get_feature_importance(rf, features_names)
    top_features = get_top_features(importance)
    misclassification_risk = analyze_misclassification_risk(cm_best_model)

    # TODO fill the dict
    return {
        "basic_info": info,
        "class_distribution": cls_distribution,
        "cramers_v": cramers_v,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "confusion_matrix": cm_best_model,
        "roc_data": roc_curve_data,
        "feature_importance": importance,
        "top_features": top_features,
        "misclassification_risk": misclassification_risk,
        "y_test": y_test,
        "y_pred": best_model_y_test_pred,
        "y_prob": best_model_y_test_prob,
        # TODO add this
        "saved_figures": "",
    }


# =============================================================================
# PUNKT WEJŚCIA
# =============================================================================

if __name__ == "__main__":
    results = run_full_analysis("house-votes-84.data")
    print(results["results_df"])
