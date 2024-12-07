### Instrukcja logowania wyników końcowych
#### Logujemy metryki dla 5 podgrup:
* `nq` -> `cnndm`
* `cnndm` -> `nq`
* crosswalidacja na zbiorach **QA**
* crosswalidacja na zbiorach **SUMM**
* crosswalidacja na wszystkich
#### Jak dzielimy zbiory
* `train` (domain task) [`0.8` / `0.5`]
* `validation` (domain task) [`0.2` / `0.5`]
* `test` (out of domain task)
#### Ustawienia zbiorów
* `window_size=8`
* `window_step=1`
#### Augmentacje zbiorów
* bez undersamplingu (`stratified`)
* undersampling (na treningowym wyłącznie) 
#### Jakie metryki logujemy
* `train_auc`, `validation_auc`, `test_auc`
* `train_auprc`, `validation_auprc`, `test_auprc`
* wykresy tych krzywych
#### Jakie hiperametry modelu ML logujemy
* wszystkie które się da
#### Jakie parametry uruchomienia logujemy
* model
* opis datasetów (co w train i co w test)
* sposób augmentacji zbioru treningowego
* sposób normalizacji danych (pipeline sklearn) `norm_type`
#### Dodajemy w opisie
* sposób agregacji (wyznaczenia cech)
* kod do wyznaczenia tej agregacji
* <b>label<b> (`research`, `production`)
### Instrukcja logowania wyników eksperymentalnych
Zasady obowiązują te same, ale zbiór jest ograniczony do:
* `nq`
* `cnndm`
* <b>0.2<b> pozostałych zbiorów -> nazwa zbioru `research_sample.parquet`
oraz robimy na ustawieniach (`train_size=0.8`)
### Wyniki są publikowane standardową funkcją z modułu `<place_holder>`
* lightgbm
* logistic regression
* LSTM
