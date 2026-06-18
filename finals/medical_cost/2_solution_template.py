"""
Medical Cost - Szablon rozwiązania
==================================

Autor: Paweł Jancewicz
Numer albumu: pd5003

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
import os

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
    return pd.read_csv(filepath)


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
    return {
        "shape": df.shape,
        "dtypes": df.dtypes,
        "missing_values": df.isna().sum(),
        "duplicates_count": df.duplicated().sum(),
    }


def get_target_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Zwraca statystyki opisowe zmiennej celu 'charges'.

    Returns
    -------
    Dict[str, float]
        Klucze: 'mean', 'median', 'std', 'min', 'max', 'q1', 'q3'
    """
    return {
        "mean": df["charges"].mean(),
        "median": df["charges"].median(),
        "std": df["charges"].std(),
        "min": df["charges"].min(),
        "max": df["charges"].max(),
        "q1": df["charges"].quantile(0.25),
        "q3": df["charges"].quantile(0.75),
    }


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
    return {
        "dtype": df[feature].dtype,
        "unique_values": df[feature].unique(),
        "unique_count": df[feature].nunique(),
        "describe": df[feature].describe(),
    }


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
    numerical_cols = ["age", "bmi", "children", "charges"]
    return df[numerical_cols].corr(method="pearson")


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
    return {
        "by_smoker": df.groupby("smoker")["charges"].mean(),
        "by_region": df.groupby("region")["charges"].mean(),
        "by_sex": df.groupby("sex")["charges"].mean(),
    }


def check_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Bada rozkład zmiennej celu przed i po transformacji logarytmicznej.

    Returns
    -------
    Dict[str, float]
        - 'skewness', 'kurtosis'          - dla charges
        - 'log_skewness', 'log_kurtosis'  - dla log(1 + charges)
    """
    return {
        "skewness": df["charges"].skew(axis=0),
        "kurtosis": df["charges"].kurt(),
        "log_skewness": pd.Series(np.log(1 + df["charges"])).skew(axis=0),
        "log_kurtosis": pd.Series(np.log(1 + df["charges"])).kurt(),
    }


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
    X = df.drop(labels=["charges"], axis=1)
    y = df["charges"]

    return X, y


def get_column_lists() -> Dict[str, List[str]]:
    """
    Zwraca podział kolumn na kategoryczne i numeryczne.

    Returns
    -------
    Dict[str, List[str]]
        - 'categorical': ['sex', 'smoker', 'region']
        - 'numerical':   ['age', 'bmi', 'children']
    """
    return {
        "categorical": ["sex", "smoker", "region"],
        "numerical": ["age", "bmi", "children"],
    }


def create_preprocessor() -> ColumnTransformer:
    """
    Tworzy preprocessor transformujący dane.

    Używa:
    - OneHotEncoder(drop='first', sparse_output=False) dla zmiennych kategorycznych
    - StandardScaler() dla zmiennych numerycznych

    Wskazówka: użyj get_column_lists() do pobrania list kolumn.
    """
    cols = get_column_lists()
    return ColumnTransformer(
        [
            (
                "cat_preprocess",
                OneHotEncoder(drop="first", sparse_output=False),
                cols["categorical"],
            ),
            ("num_preprocess", StandardScaler(), cols["numerical"]),
        ]
    )


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Dzieli dane na zbiór treningowy i testowy.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


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
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
    }


def train_model(model: Any, X_train: np.ndarray, y_train: pd.Series) -> Any:
    """
    Trenuje pojedynczy model na przetworzonych cechach i zwraca go.
    """
    model.fit(X_train, y_train)
    return model


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
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Trenuje i ocenia wszystkie modele na zbiorze treningowym i testowym.

    Returns
    -------
    pd.DataFrame
        Kolumny: 'Model', 'MAE_train', 'MSE_train', 'RMSE_train', 'R2_train',
                 'MAE_test', 'MSE_test', 'RMSE_test', 'R2_test'
    """
    rows = []

    for model_name, model in models.items():
        trained_model = train_model(model, X_train, y_train)

        y_train_pred = trained_model.predict(X_train)
        y_test_pred = trained_model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        rows.append(
            {
                "Model": model_name,
                "MAE_train": train_metrics["MAE"],
                "MSE_train": train_metrics["MSE"],
                "RMSE_train": train_metrics["RMSE"],
                "R2_train": train_metrics["R2"],
                "MAE_test": test_metrics["MAE"],
                "MSE_test": test_metrics["MSE"],
                "RMSE_test": test_metrics["RMSE"],
                "R2_test": test_metrics["R2"],
            }
        )

    return pd.DataFrame(rows)


def get_best_model_name(results_df: pd.DataFrame, metric: str = "R2_test") -> str:
    """
    Zwraca nazwę najlepszego modelu według wskazanej metryki.
    (dla R2 - wartość najwyższa, dla MAE/MSE/RMSE - najniższa)
    """
    match metric:
        case (
            "MAE_train"
            | "MSE_train"
            | "RMSE_train"
            | "MAE_test"
            | "MSE_test"
            | "RMSE_test"
        ):
            return results_df.at[results_df[metric].idxmin(), "Model"]
        case "R2_train" | "R2_test":
            return results_df.at[results_df[metric].idxmax(), "Model"]


def get_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    """Zwraca predykcje modelu dla podanych cech."""
    return model.predict(X)


def calculate_residuals(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Zwraca reszty (y_true - y_pred)."""
    return y_true - y_pred


# =============================================================================
# CZĘŚĆ 6: INTERPRETACJA MODELU
# =============================================================================


def get_feature_importance(
    model: RandomForestRegressor, feature_names: List[str]
) -> pd.DataFrame:
    """
    Zwraca ważność cech modelu RandomForest.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'importance'), posortowane malejąco po 'importance'.
    """
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)


def get_linear_coefficients(
    model: LinearRegression, feature_names: List[str]
) -> pd.DataFrame:
    """
    Zwraca współczynniki regresji liniowej.

    Returns
    -------
    pd.DataFrame
        Kolumny ('feature', 'coefficient'), posortowane malejąco po |coefficient|.
    """
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    ).sort_values(by="coefficient", key=np.abs, ascending=False)


def get_top_features(importance_df: pd.DataFrame, n: int = 3) -> List[str]:
    """Zwraca listę n najważniejszych cech."""
    return importance_df["feature"].iloc[:n].tolist()


# =============================================================================
# CZĘŚĆ 7: WIZUALIZACJE
# =============================================================================


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram zmiennej celu (charges)."""
    fig, ax = plt.subplots()
    sns.histplot(df["charges"], ax=ax)
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Heatmapa macierzy korelacji."""
    fig, ax = plt.subplots()
    sns.heatmap(data=corr_matrix, ax=ax, annot=True)
    return fig


def plot_feature_vs_target(df: pd.DataFrame, feature: str) -> plt.Figure:
    """Wykres zależności cechy od zmiennej celu (scatter lub box plot)."""
    num_cols = (
        df.select_dtypes(include="number").drop(labels=["charges"], axis=1).columns
    )
    cat_cols = df.select_dtypes(include="object").columns

    fig, ax = plt.subplots()
    if feature in num_cols:
        sns.scatterplot(x=df[feature], y=df["charges"], ax=ax)
    elif feature in cat_cols:
        sns.boxplot(x=df[feature], y=df["charges"], ax=ax)

    return fig


def plot_group_statistics(group_stats: Dict[str, pd.Series]) -> plt.Figure:
    """Trzy wykresy słupkowe: średnie charges wg smoker, region, sex."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    sns.barplot(
        x=group_stats["by_smoker"].index, y=group_stats["by_smoker"].values, ax=ax1
    )
    sns.barplot(
        x=group_stats["by_region"].index, y=group_stats["by_region"].values, ax=ax2
    )
    sns.barplot(x=group_stats["by_sex"].index, y=group_stats["by_sex"].values, ax=ax3)

    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    """Wykres punktowy predykcji vs wartości rzeczywiste z linią y=x."""
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
    return fig


def plot_residuals(y_pred: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    """Wykres reszt względem predykcji."""
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Wykres słupkowy ważności cech."""
    fig, ax = plt.subplots()
    sns.barplot(importance_df.iloc[:top_n], x="feature", y="importance", ax=ax)
    return fig


def plot_coefficients(coef_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Wykres słupkowy współczynników regresji."""
    fig, ax = plt.subplots()
    sns.barplot(coef_df.iloc[:top_n], x="feature", y="coefficient", ax=ax)
    return fig


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
    os.makedirs(name=output_dir, exist_ok=True)

    fig = plot_target_distribution(df)
    fig.savefig(fname=f"{output_dir}/plot_target_distribution.jpg")
    plt.close(fig)

    fig = plot_correlation_heatmap(results["correlation_matrix"])
    fig.savefig(fname=f"{output_dir}/correlation_matrix.jpg")
    plt.close(fig)

    # fig = plot_feature_vs_target(df)
    # fig.savefig(fname=f"{output_dir}/feature_vs_target.jpg")
    # plt.close(fig)



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
    df = load_data(filepath)

    basic_info = get_basic_info(df)
    target_stats = get_target_statistics(df)

    corr_matrix = get_correlation_matrix(df)
    group_stats = get_group_statistics(df)
    distrib = check_distribution(df)

    X, y = prepare_features(df)
    cols = get_column_lists()

    X_train, X_test, y_train, y_test = split_data(X, y, 0.2, 42)

    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    preprocess_cols_names = preprocessor.get_feature_names_out()

    models = get_models()
    results_df = evaluate_all_models(
        models, X_train_processed, X_test_processed, y_train, y_test
    )
    best_model_name = get_best_model_name(results_df)

    trained_best_model = train_model(
        models[best_model_name], X_train_processed, y_train
    )
    trained_rf = train_model(
        models["RandomForest"],
        X_train_processed,
        y_train,
    )
    trained_lr = train_model(models["LinearRegression"], X_train_processed, y_train)

    importances = get_feature_importance(trained_rf, preprocess_cols_names)
    coeffs = get_linear_coefficients(trained_lr, preprocess_cols_names)

    top_features_rf = get_top_features(importances)
    top_features_lr = get_top_features(coeffs)

    y_test_pred = get_predictions(trained_best_model, X_test_processed)
    residuals = calculate_residuals(y_test, y_test_pred)

    results = {
        "basic_info": basic_info,
        "target_stats": target_stats,
        "correlation_matrix": corr_matrix,
        "group_stats": group_stats,
        "distribution_stats": distrib,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": trained_best_model,
        "feature_importance": importances,
        "linear_coefficients": coeffs,
        "top_features_rf": top_features_rf,
        "top_features_lr": top_features_lr,
        "y_test": y_test,
        "y_pred_test": y_test_pred,
        "residuals": residuals,
    }

    if generate_figures:
        results["saved_figures"] = generate_all_figures(df, results, figures_dir)

    return results

# =============================================================================
# PUNKT WEJŚCIA
# =============================================================================

if __name__ == "__main__":
    # Przykład użycia:
    results = run_full_analysis("insurance.csv")
    print(results['results_df'])
