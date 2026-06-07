---
format:
   pdf:
      pdf-engine: xelatex
---

# Congressional Voting Records - Wnioski

**Autor:** [Imię Nazwisko]
**Numer albumu:** [XXXXX]
**Data:** [DD.MM.YYYY]

---

## 1. Analiza danych

*[Max 200 słów]*

<!--
Wskazówki:
- Jaki jest rozkład klas (democrat vs republican)? Czy zbiór jest zbalansowany?
- Jak obsłużyłeś wartości '?'? Pamiętaj, że NIE jest to klasyczny brak danych.
  Dlaczego wybrałeś tę strategię, a nie inną?
- Które głosowania mają najwyższy współczynnik Cramér's V względem przynależności?
-->

[Twoja odpowiedź tutaj]

---

## 2. Porównanie modeli

*[Max 200 słów]*

<!--
Wskazówki:
- Który model osiągnął najwyższe F1 / AUC?
- Jak wypadły modele proste (LogisticRegression, NaiveBayes) vs zespołowe (RandomForest, GradientBoosting)?
- Czy CategoricalNB dobrze poradził sobie z danymi czysto kategorycznymi?
-->

[Twoja odpowiedź tutaj]

**Tabela porównawcza** (wklej lub opisz najważniejsze wartości):

| Model | Accuracy | F1 | AUC |
|-------|----------|----|----|
|  ...  |   ...    | ...| ...|

---

## 3. Analiza błędów

*[Max 150 słów]*

<!--
Wskazówki:
- Ile było błędów False Positive, a ile False Negative?
- Czy błędy rozkładają się symetrycznie, czy model myli się częściej w jedną stronę?
- W jakim hipotetycznym zastosowaniu koszt jednego typu błędu byłby wyższy niż drugiego
  (Precision vs Recall)?
-->

[Twoja odpowiedź tutaj]

---

## 4. Interpretacja cech

*[Max 150 słów]*

<!--
Wskazówki:
- Które 5 cech jest najważniejszych według RandomForest?
- Czy ranking RandomForest pokrywa się z rankingiem Cramér's V?
- Które głosowania najsilniej różnicują obie grupy?
- Analiza techniczna - bez ocen politycznych.
-->

**Top 5 cech - RandomForest:**

1. [cecha 1]
2. [cecha 2]
3. [cecha 3]
4. [cecha 4]
5. [cecha 5]

[Twoja interpretacja]

---

## 5. Wnioski

*[Max 100 słów]*

<!--
Wskazówki:
- Jakie jest główne wnioski z projektu?
- Czy zapis 16 głosowań pozwala wiarygodnie przewidywać przynależność?
- Co można by poprawić (inne kodowanie, strojenie hiperparametrów, więcej cech)?
-->

[Twoja odpowiedź tutaj]

---

## Załączniki

### Wykres 1: Rozkład klas

![Rozkład klas](figures/class_distribution.png)

### Wykres 2: physician-fee-freeze względem klasy

![physician-fee-freeze](figures/feature_1.png)

### Wykres 3: adoption-of-the-budget-resolution względem klasy

![adoption-of-the-budget-resolution](figures/feature_2.png)

### Wykres 4: Ranking Cramér's V

![Ranking Cramér's V](figures/cramers_v_ranking.png)

### Wykres 5: Macierz pomyłek

![Macierz pomyłek](figures/confusion_matrix.png)

### Wykres 6: Krzywa ROC

![Krzywa ROC](figures/roc_curve.png)

### Wykres 7: Ważność cech (RandomForest)

![Ważność cech](figures/feature_importance.png)

---

*Uwaga: Zapisz wykresy w folderze `figures/` i upewnij się, że ścieżki są poprawne.*
