# Punkty wykrycia dla tagów scen bez markerów

Funkcja jest domyślnie włączona w trybie `merged`. W `temporal_policy.yml`
odpowiada za nią `occurrence_markers: true`; `false` wyłącza dodatkowe punkty.
`--no-temporal` również ją wyłącza. Nie wymaga listy dozwolonych tagów
ani zmiany progów modeli. Tryb `separate` zachowuje dotychczasowe zachowanie.

## Uruchomienie po aktualizacji

Rozpakuj całą paczkę do katalogu projektu. Zachowaj własne ustawienia połączenia
i ścieżek oraz bazę `tagger_sources.sqlite3` używaną do tej pory.

```powershell
# Dodaj punkty na podstawie istniejących wyników, bez uruchamiania modeli:
python .\stash_ai.py --merge-only

# Opcjonalny szczegółowy podgląd jednej sceny:
python .\stash_ai.py --merge-only --scene-id 123 --dry-run --verbose

# Tylko markery, z wykorzystaniem zapisanych wyników tagowania scen:
python .\stash_ai.py --merge-only --markers-only
```

Zmiana ustawienia unieważnia cache publikacji, ale nie ukończone inferencje.
Nie trzeba usuwać bazy ani wykonywać globalnego `--restart`.

## Co będzie widoczne w Stash

- Jeden dodatkowy punkt na zaakceptowany tag AI, który nie ma własnego markera.
- Główny tag: `ai:<etykieta>`; tytuł np. `[AI] etykieta [jt+wd] · wykrycie`.
- Dodatkowy tag `ai:marker:occurrence` pozwala filtrować tę kategorię.
- Marker nie ma czasu końcowego. Wskazuje próbkę, w której wykryto etykietę.
- Ręczny marker z tym samym głównym tagiem `ai:` zapewnia już lokalizację
  i blokuje dodanie punktu. Nie jest modyfikowany przez tę regułę.

To jedno reprezentatywne miejsce, nie lista wszystkich wystąpień. Punkty
uzupełniają markery zakresowe. Jeżeli później powstanie normalny marker danego
tagu, punkt zostaje zastąpiony; identyfikator może być zachowany, gdy lokalizacje
się pokrywają. Usuwany jest wtedy także dodatkowy tag kategorii punktu.

## Wybór lokalizacji i wykluczenia

Tag musi wcześniej przejść obecną kwalifikację dla sceny. Brakująca lokalizacja
jest wyszukiwana w zapisanych maksimach lub wynikach klatek z odpowiadających
przebiegów. Zgodność modeli liczy się tylko dla tego samego wejścia i czasu;
duplikaty jednego modelu nie zwiększają liczby głosów.

Wybór preferuje liczbę modeli przekraczających własny próg sceny, liczbę modeli
z zachowanym potwierdzeniem, następnie względny margines wyniku ponad własny
próg sceny. Przy remisie wygrywa wcześniejsza próbka. Margines służy wyłącznie
do uporządkowania kandydatów: wyniki modeli nie są uśredniane i nie powstaje
wspólne prawdopodobieństwo. Poszczególne wyniki i źródła pozostają w bazie.

Obowiązują `scene_tag_exclusions.yml`, `marker_core_exclusions.yml`
i `MARKER_TAG_BLOCKLIST`. Sam dodatkowy tag długiego markera nie zapewnia
własnej lokalizacji; taka etykieta może dostać punkt, jeżeli nie jest wykluczona
z głównych tagów. Dotychczasowe progi i `MARKER_MIN_HITS` dla zakresów pozostają
bez zmian. Przy niezerowym `MAX_MARKERS_PER_SCENE` nowe punkty wypełniają tylko
miejsce pozostałe po istniejących zakresach.

Cleanup cofa wkład wykluczonych modeli, zachowując punkt wyłącznie, gdy pozostałe
opublikowane potwierdzenia nadal go uzasadniają. Nie przesuwa punktów do nowego
miejsca i nie odtwarza ręcznie usuniętych markerów. Dopiero zwykły merge może
wybrać nowe reprezentatywne miejsce. Szybki cleanup SQL obsługuje punkty tak
samo jak pozostałe markery. Surowe wyniki pozostają dostępne do kolejnego merge.

## Starsze wyniki bez czasów

Nowe inferencje zapisują `peak_time` obok `max_score`, także dla wykryć poniżej
progu zapisu markerów. Jest to jedna liczba na zaobserwowany tag i model,
zachowywana także w checkpointach. Nie zapisujemy pełnej macierzy wszystkich
klatek i wszystkich etykiet. Nie dodano kolejnego przebiegu GPU.

Dla starych wyników merge najpierw korzysta z zachowanych klatek. Jeżeli
pozostał wyłącznie wynik zbiorczy bez lokalizacji, raport zwiększa
`occurrence_missing_timestamps`. Skrypt nie zgaduje czasu. Konkretne źródła
i zapisane czasy można sprawdzić lokalnie:

```powershell
python .\stash_ai.py --explain-tag ai:nazwa_tagu --scene-id 123

# Tylko jeśli potrzebny czas nie został zachowany: odśwież daną scenę/model.
python .\stash_ai.py --models wd --scene-id 123 --restart
```

`occurrence_markers` w raporcie liczy punkty w przeliczonych scenach,
`occurrence_manual_locations` pominięcia dzięki ręcznym lokalizacjom,
a `occurrence_limited` pominięcia przez limit liczby markerów. Niezmienione sceny
nadal korzystają z `merge_reused`. Czas dodatkowej selekcji pokazuje
`occurrence_compute` w podsumowaniu etapów.

Dokładność czasowa jest ograniczona próbkowaniem VTT/sprite'ów. Czas próbki
nie jest pomiarem dokładnego początku ani długości zdarzenia.

## Weryfikacja

Testy działają offline: bez połączenia z Twoim Stash i bez uruchamiania GPU.
Wynik regresji zapisano w `validation_occurrences.txt`.
`benchmark_occurrences.py` mierzy osobno dodatkową selekcję punktów na danych
syntetycznych. Nie jest pomiarem czasu publikacji ani całej kolekcji.

Mediana pięciu prób dla 500 próbek, 100 etykiet i 4 modeli:

| Dane wejściowe | Dodatkowy czas selekcji | Nowe punkty |
|---|---:|---:|
| Stare, pojedyncze trafienia w zapisanych klatkach | 61,8 ms | 100 |
| Nowe zapisane maksima, bez zachowanych trafień w klatkach | 1,9 ms | 100 |
| Wszystkie tagi mają już własną lokalizację | 0,018 ms | 0 |

Surowe pomiary zawiera `occurrence_performance_results.json`. Rzeczywisty koszt
zależy od liczby zachowanych wykryć i brakujących lokalizacji. Niezmieniony merge
nadal pomija ten etap dzięki istniejącemu cache publikacji.

```powershell
python -m pytest -q
python .\benchmark_occurrences.py
```
