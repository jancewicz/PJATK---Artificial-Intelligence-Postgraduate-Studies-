---
format:
   pdf:
      pdf-engine: xelatex
---

# Medical Cost - Wnioski

**Autor:** [Imię Nazwisko]
**Numer albumu:** [XXXXX]
**Data:** [DD.MM.YYYY]

---

## 1. Analiza eksploracyjna

*[Max 200 słów - opisz kluczowe obserwacje z analizy danych]*

<!--
Wskazówki:
- Jak wygląda rozkład zmiennej celu (charges)? Czy jest symetryczny? Co daje transformacja log?
- Które cechy numeryczne najsilniej korelują z charges?
- Pamiętaj: korelacja Pearsona obejmuje tylko cechy numeryczne. Co mówią statystyki grupowe
  (by_smoker, by_region, by_sex) o cechach kategorycznych?
- Czy były braki danych lub duplikaty?
-->

[Twoja odpowiedź tutaj]

---

## 2. Porównanie modeli

*[Max 200 słów - porównaj wyniki modeli]*

<!--
Wskazówki:
- Który model osiągnął najwyższe R² na zbiorze testowym?
- Jak wypadły modele proste (LinearRegression) vs zespołowe (RandomForest, GradientBoosting)?
- Czy występuje overfitting? (porównaj metryki train vs test - który model różni się najbardziej?)
- Dlaczego Twój najlepszy model wypadł najlepiej?
-->

[Twoja odpowiedź tutaj]

**Tabela porównawcza** (wklej lub opisz najważniejsze wartości):

| Model | R²_train | R²_test | RMSE_test |
|-------|----------|---------|-----------|
|  ...  |   ...    |   ...   |    ...    |

---

## 3. Interpretacja cech

*[Max 200 słów - zinterpretuj ważność cech]*

<!--
Wskazówki:
- Które 3 cechy są najważniejsze według RandomForest?
- Które 3 cechy mają najwyższe |współczynniki| w LinearRegression?
- Czy oba modele wskazują te same cechy jako kluczowe?
- Jak interpretujesz znaczenie tych cech (np. dlaczego palenie tak silnie wpływa na koszty)?
-->

**Top 3 cechy - RandomForest:**

1. [cecha 1]
2. [cecha 2]
3. [cecha 3]

**Top 3 cechy - LinearRegression:**

1. [cecha 1]
2. [cecha 2]
3. [cecha 3]

[Twoja interpretacja różnic/podobieństw]

---

## 4. Analiza reszt

*[Max 100 słów - oceń wykres reszt]*

<!--
Wskazówki:
- Czy reszty są rozłożone losowo wokół zera?
- Czy widzisz jakiś wzorzec (np. systematyczne niedoszacowanie dla dużych wartości charges)?
- Co to oznacza dla jakości modelu?
-->

[Twoja odpowiedź tutaj]

---

## 5. Wnioski i rekomendacje

*[Max 150 słów - podsumuj projekt]*

<!--
Wskazówki:
- Jakie jest główne wnioski z projektu?
- Czy model nadaje się do praktycznego zastosowania (np. wstępna wycena składki)?
- Co można zrobić, aby poprawić wyniki?
  - Inne cechy / feature engineering (np. interakcja smoker × bmi)?
  - Transformacja zmiennej celu (log)?
  - Inne modele (np. XGBoost)?
-->

[Twoja odpowiedź tutaj]

---

## Załączniki

### Wykres 1: Heatmapa korelacji

![Heatmapa korelacji](figures/correlation_heatmap.png)

### Wykres 2: Predykcje vs wartości rzeczywiste

![Predykcje vs rzeczywiste](figures/predictions_vs_actual.png)

### Wykres 3: Wykres reszt

![Wykres reszt](figures/residuals.png)

### Wykres 4: Ważność cech (RandomForest)

![Ważność cech](figures/feature_importance.png)

---

*Uwaga: Zapisz wykresy w folderze `figures/` i upewnij się, że ścieżki są poprawne.*
