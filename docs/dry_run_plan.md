# Concrete saved-evidence publication preview

```sh
python stash_ai.py --merge-only --dry-run --dry-run-plan review.jsonl
```

The destination must be a new `.jsonl` file. Add `--scene-id ID` to restrict the
selection or `--force-merge` to show the plan even for a reusable publication.
This mode uses existing evidence, never runs models, and writes only the report
and the temporary preview overlay. The source database and Stash remain intact.

Each line is a JSON object:

- `header`: format version and the `dry-run:` symbolic ID convention.
- `tag`: exact create/update payload, symbolic or existing ID, and reason. This
  includes generated tags, hierarchy/category changes and marker group repair.
- `scene`: scene-tag before/after IDs; marker creates/updates with exact payloads;
  deletion IDs; reasons, ID/name references and operation counts.
- `status`: an unchanged publication was reused or a scene could not be planned.
- `summary`: CLI exit code. An interrupted run instead records `interrupted`.

Tag creation in a real run receives server IDs. Symbolic IDs consistently name
those future tags and their references; existing IDs and manual associations are
preserved. The plan describes the observed state during preview, so running it
later still requires the normal fresh reads and ownership checks. JSONL keeps
memory bounded to one scene and leaves completed lines inspectable after errors.
