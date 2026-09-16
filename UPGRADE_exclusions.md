# Szybkie sprzątanie blacklisty

Podmień pliki `.py` z paczki, w tym nowy `stash_cleanup.py`. Zachowaj swoje
blacklisty YAML, ustawienia połączenia i dotychczasową bazę `tagger_sources.sqlite3`.
Migracja bazy wyników do formatu 5 odbywa się automatycznie; nie wymaga inferencji.

```powershell
python .\stash_ai.py --models all --cleanup-only
```

Jeżeli używasz innej ścieżki bazy, dodaj dotychczasowe `--state-db`.
Podgląd: dodaj `--dry-run`. Nie dodawaj `--scene-id`, jeśli chcesz użyć
szybkiego globalnego usuwania definicji tagów.

Pełny cleanup pobiera katalog tagów i wyszukuje pasujące markery przez API.
Usuwa markery z globalnie zakazanym głównym tagiem, a następnie same tagi,
partiami. Globalne wpisy z `scene_tag_exclusions.yml` obejmują `ai:`, `jt:`,
`wd:`, `dtq:` i `op:`, również ręczne przypisania tych nazw i ich użycie przy
obrazach lub innych obiektach Stash. Ręczne tagi bez prefiksów pozostają.

Zakazany tag pomocniczy znika razem z definicją; pozostały marker zachowuje
identyfikator, tytuł, początek i koniec. Dozwolone dzieci usuniętego tagu zostają,
tracąc relację z usuniętym rodzicem. Lista `marker_core_exclusions.yml` nadal
usuwa wyłącznie markery zarządzane przez skrypt, zachowując same definicje tagów.

Dla wspólnych wykluczeń skrypt nie odczytuje osobno wszystkich scen i nie
przelicza wyników modeli. Wpisy zależne od modelu są obsługiwane tylko na scenach
wyszukanych po pasujących tagach i markerach. Tryby częściowe i wybór pojedynczych
modeli zachowują swój zakres. Starsze API bez filtrowanego wyszukiwania korzysta
z dotychczasowej ścieżki, w której usunięto zbędne odczyty scen dla każdego modelu.

Paski postępu obejmują katalog, wyszukiwanie i usuwanie markerów/tagów.
Podsumowanie pokazuje `cleanup_fast_path=1`, liczbę usuniętych definicji
`blacklisted_tags_deleted` i liczbę scen wymagających dodatkowej obsługi
`cleanup_candidate_scenes`. Przy samej wspólnej blackliście ta ostatnia wynosi 0.

SQLite zachowuje surowe wyniki oraz kompaktowe reguły usunięć stosowane przy
odczycie starego stanu publikacji. Nie ma masowego przepisywania snapshotów,
pobierania sprite’ów ani dodatkowego cache obrazów.

Przerwany globalny cleanup uruchom ponownie tą samą komendą, z tymi samymi
blacklistami. Skrypt wyszukuje pozostałe obiekty i omija już usunięte.
Do zakończenia cleanupu inne zapisy korzystające z tej bazy stanu są blokowane.
Jeśli wcześniej przerwano merge, najpierw dokończ `--models all --merge-only`.
Po zakończeniu cleanupu możesz zmienić blacklistę; zdjęcie zakazu i `--merge-only`
pozwala odtworzyć wyniki z zachowanych wykryć, bez uruchamiania modeli.

Opcjonalne sprzątanie pozostałych pustych tagów po dawnych merge’ach:

```powershell
python .\stash_ai.py --models all --cleanup-only --prune-empty-tags
```

W tym dodatkowym kroku usuwane są tylko nieużywane tagi bez dzieci.
