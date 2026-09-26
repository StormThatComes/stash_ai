# Cleanup SQL dużej kolekcji

Ten dokument opisuje bieżącą ścieżkę cleanupu dla dużych bibliotek.

## Strategie

`--cleanup-strategy auto` jest domyślne:
- używa read-only `querySQL`, gdy wymagane API i schema są dostępne;
- w przeciwnym razie przechodzi na odczyt scen przez GraphQL.

`--cleanup-strategy sql` wymaga ścieżki SQL i kończy się błędem, jeśli nie może
jej użyć. `--cleanup-strategy scenes` wymusza odczyt scen.

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --dry-run
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql
```

SQL służy wyłącznie do odczytu. Aplikacja nie wykonuje bezpośrednich
`DELETE`/`UPDATE` na bazie Stash; wszystkie zmiany przechodzą przez API Stash.

## Zakres cleanupu

Dostępne tryby:
- `--cleanup-only` — sceny i markery;
- `--cleanup-markers-only`;
- `--cleanup-scene-tags-only`.

Globalne wpisy z `scene_tag_exclusions.yml` mogą usuwać definicje zakazanych
tagów wraz z ich relacjami. `marker_core_exclusions.yml` dotyczy wyłącznie
zarządzanych core markerów i nie usuwa samej definicji tagu.

Tryby częściowe zachowują definicje tagów i modyfikują tylko odpowiedni rodzaj
powiązania.

## Dlaczego SQL jest szybszy

Ścieżka SQL:
- używa stałych, parametryzowanych SELECT-ów;
- stronicuje po kluczu/ID zamiast COUNT+OFFSET;
- pobiera ID kandydatów zamiast pełnych obiektów markerów;
- nie tworzy list całej biblioteki w pamięci;
- ponownie weryfikuje stan przed zapisem przez API.

Dla reguł zależnych od modelu pobierane są tylko sceny wymagające dokładnego
przeliczenia.

## Rozmiary partii

| Opcja | Domyślnie | Zakres |
|---|---:|---:|
| `--cleanup-page-size` | 2000 | 1–10000 |
| `--cleanup-write-batch-size` | 200 | 1–1000 |

Jeśli serwer długo wykonuje mutacje, można zmniejszyć write batch:

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --cleanup-write-batch-size 50
```

## Retry i wznowienie

Read-only GraphQL/SQL może zostać ponowiony po błędach przejściowych. Mutacje
GraphQL nie są retry'owane automatycznie, ponieważ po timeoutcie serwer mógł już
wykonać zapis.

Po przerwaniu uruchom tę samą operację z tym samym stanem i polityką. Aplikacja
ponownie odczyta aktualne obiekty i będzie kontynuować z tym, co pozostało.

Jeśli zamiast wznowienia chcesz użyć zmienionych reguł:

```powershell
python .\stash_ai.py --models all --cleanup-only --abandon-pending
```

Szczegóły: [UPGRADE_abandon_pending.md](UPGRADE_abandon_pending.md).

## Dry-run

`--dry-run` nie zapisuje do Stash ani źródłowej bazy. Dla wspieranej nowoczesnej
bazy aplikacja używa read-only źródła i tymczasowego overlay SQLite, więc podgląd
nie wymaga kopiowania całego pliku stanu do RAM.

## Pruning pustych definicji

Po udanym cleanupie można usunąć nieużywane liście:

```powershell
python .\stash_ai.py --models all --cleanup-only --prune-empty-tags
```

Samodzielnie:

```powershell
python .\stash_ai.py --prune-empty-tags-only --dry-run
```

Najpierw odrzucane są oczywiście używane tagi, a pełna weryfikacja liczników Stash
jest wykonywana tylko dla pozostałych kandydatów.

## Diagnostyka

Podsumowanie cleanupu raportuje m.in. liczbę odczytów SQL, odczytanych wierszy,
zapisów API i operacji obiekt–tag. Do porównań syntetycznych służy:

```powershell
python -m benchmarks.benchmark_cleanup_sql --markers 100000
```

Benchmark nie łączy się z produkcyjnym Stash.
