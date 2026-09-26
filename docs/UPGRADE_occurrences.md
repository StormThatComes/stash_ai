# Punkty Occurrence dla tagów scen

Occurrence jest domyślnie włączone w trybie `merged` przez
`occurrence_markers: true` w `temporal_policy.yml`. `--no-temporal` również
wyłącza tę funkcję.

## Co tworzy

Dla zaakceptowanego tagu sceny, który nie ma własnego markera zakresowego,
aplikacja może utworzyć jeden reprezentatywny punkt:

- główny tag: `ai:<etykieta>`;
- techniczny tag: `ai:marker:occurrence`;
- brak czasu końcowego;
- lokalizacja wskazuje zapisaną próbkę VTT/sprite.

To nie jest lista wszystkich wystąpień i nie jest próbą wyznaczenia dokładnego
początku zdarzenia.

Ręczny marker z tym samym głównym tagiem zapewnia już lokalizację i blokuje
tworzenie punktu; ręczny marker nie jest przez tę regułę modyfikowany.

## Użycie

Occurrence jest obliczane podczas zwykłego merge:

```powershell
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run --verbose
python .\stash_ai.py --merge-only --markers-only
```

Zmiana `occurrence_markers` wpływa na publikację, ale sama w sobie nie wymaga
nowej inferencji.

## Wybór czasu

Tag musi najpierw przejść kwalifikację sceny. Lokalizacja jest wybierana z
zachowanych maksimów lub obserwacji klatek.

Przy porównywaniu kandydatów brane są pod uwagę m.in.:
- liczba modeli potwierdzających próbkę;
- liczba zachowanych dowodów;
- względny margines względem progu sceny;
- wcześniejszy czas przy remisie.

Wyniki modeli nie są uśredniane do wspólnego prawdopodobieństwa.

Jeżeli zapisany cache nie zawiera czasu potrzebnego do lokalizacji, aplikacja nie
zgaduje. Można sprawdzić dostępne evidence:

```powershell
python .\stash_ai.py --explain-tag ai:nazwa_tagu --scene-id 123
```

Jeśli lokalizacja jest potrzebna, a cache jej nie ma, wykonaj świeżą inferencję
konkretnej sceny/modelu:

```powershell
python .\stash_ai.py --models wd --scene-id 123 --restart
```

## Wykluczenia i limity

Occurrence respektuje:
- `scene_tag_exclusions.yml`;
- `marker_core_exclusions.yml`;
- `MARKER_TAG_BLOCKLIST`;
- `MAX_MARKERS_PER_SCENE`.

Normalny marker zakresowy ma pierwszeństwo przed reprezentatywnym punktem tego
samego tagu.

Cleanup może usunąć lub zachować punkt na podstawie pozostałych opublikowanych
dowodów. Nie przesuwa istniejącego punktu do nowego miejsca; nową lokalizację
wybiera dopiero kolejny merge.

## Dokładność

Dokładność czasu jest ograniczona częstotliwością próbek VTT/sprite. Punkt
Occurrence oznacza reprezentatywne wykrycie, a nie dokładny timestamp zdarzenia.

Do syntetycznego pomiaru samej selekcji służy:

```powershell
python -m benchmarks.benchmark_occurrences
```

Benchmark nie mierzy publikacji do Stash.
