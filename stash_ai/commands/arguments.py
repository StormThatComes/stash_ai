"""CLI argument definitions; grouped validation lives in validation.py."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from stash_ai.commands.validation import validate_args
from stash_ai.commands.modes import Command


def build_parser(app) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Tag Stash scenes and timestamped markers with JoyTag, WD, DTQ and/or OppaiOracle.")
    parser.add_argument("--models", nargs="+", metavar="MODEL", help="joytag, wd, dtq, oppai, any subset, or all; commas also accepted.")
    parser.add_argument("--model-config", type=Path, default=app.MODELS_CONFIG_FILE, metavar="YAML", help="Selection, paths, thresholds and batch sizes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override configured device.")
    parser.add_argument("--offline", action="store_true", help="Require local model assets; never download.")
    parser.add_argument("--temporal-policy", type=Path, default=app.TEMPORAL_POLICY_FILE, metavar="YAML",
                        help="Automatic scene qualification, marker hysteresis and extras using Stash VTT/sprites only.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to Stash.")
    parser.add_argument('--dry-run-plan', type=Path, metavar='NEW.jsonl',
                        help='Write exact scene mutations/reasons to JSONL; requires --dry-run --merge-only. No inference.')
    parser.add_argument("--scene-id", action="append", metavar="ID", help="Limit to this scene; repeat for multiple scenes.")
    parser.add_argument("--state-db", type=Path, default=app.MERGE_STATE_FILE, metavar="SQLITE",
                        help="Durable per-model evidence for merged outputs; retain this file between runs.")
    parser.add_argument("--run-history-db", type=Path, metavar="SQLITE",
                        help="Independent run history DB; default: <state-db>.runs.sqlite3.")
    history = parser.add_mutually_exclusive_group()
    history.add_argument("--record-run", action="store_true",
                         help="Also record a read-only command in run history; dry runs never record history.")
    history.add_argument("--no-run-history", action="store_true",
                         help="Disable automatic history of inference, merge and write/maintenance commands.")
    parser.add_argument('--history-keep-last', type=int, metavar='N',
                        help='Keep at most N finished runs in this Stash scope; --prune-run-history only.')
    parser.add_argument('--history-max-age-days', type=float, metavar='DAYS',
                        help='Remove finished runs older than this age; --prune-run-history only.')
    parser.add_argument("--run-limit", type=int, metavar="N", help="Most recent matching runs: 50 for --run-report, 1 for --reuse-report; maximum: 10000.")
    parser.add_argument("--run-command", choices=[item.value for item in Command],
                        help="Filter run/reuse reports by command.")
    parser.add_argument("--run-status", choices=("running", "finished", "error", "interrupted"), help="Filter run/reuse reports by run status.")
    parser.add_argument("--run-model", choices=("joytag", "wd", "dtq", "oppai"), help="Filter run/reuse reports by model.")
    parser.add_argument("--db-init", action="store_true",
                        help="Rebuild a new/empty state DB from fresh inference, adopting existing AI outputs. "
                             "Requires both outputs; interrupted initialization resumes normally.")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", default=None,
                        help="Reuse matching completed results and resume checkpoints (default; merged output).")
    resume.add_argument("--restart", action="store_true", help="Request fresh inference for selected scenes/models; resume later without this flag.")
    parser.add_argument("--checkpoint-batches", type=int, default=app.CHECKPOINT_BATCHES,
                        help="Persist partial inference every N batches; also saves on Ctrl+C/errors. Default: 8.")
    parser.add_argument("--prefetch-batches", type=int, choices=range(17), default=None,
                        help="Prepare N next batches in RAM; overrides YAML prefetch_batches (default: 4); 0 disables.")
    parser.add_argument("--ram-cache-gb", type=float, default=app.RAM_CACHE_GB, metavar="GIB",
                        help="Shared result/index/media cache in GiB (default: 24); 0 disables. Capped to 75%% of --ram-budget-gb.")
    parser.add_argument("--ram-budget-gb", type=float, default=app.RAM_BUDGET_GB, metavar="GIB",
                        help="Process RAM target in GiB (default: 32); reserves 25%% for models and active work. RSS pressure evicts caches.")
    parser.add_argument("--prepare-workers", type=int, choices=range(1, 17), default=None,
                        help="CPU image preparation workers; overrides YAML prepare_workers (default: 4).")
    parser.add_argument("--merge-workers", type=int, choices=range(1, 17), default=None,
                        help="Read-only CPU workers; overrides YAML merge_workers (default: 4). Stash/state writes stay serial.")
    parser.add_argument("--model-report-workers", type=int, choices=range(1, 17),
                        default=None,
                        help="CPU report workers; overrides YAML model_report_workers (default: up to 4); 1 runs serially.")
    parser.add_argument("--tag-review-margin", type=float, default=None, metavar="SCORE",
                        help="Weak/borderline score margin above each model's own threshold (default: 0.05); --tag-review-report only.")
    parser.add_argument("--tag-review-long-marker-seconds", type=float, default=None, metavar="SECONDS",
                        help="Flag merged marker ranges at least this long (default: 300); --tag-review-report only.")
    parser.add_argument("--tag-review-examples", type=int, default=None, metavar="N",
                        help="Retain up to N example scenes per model/tag (default: 3, maximum: 20); --tag-review-report only.")
    parser.add_argument("--verbose", action="store_true", help="Include per-scene/tag diagnostics and detailed dry-run changes.")
    parser.add_argument("--profile-merge", type=Path, metavar="NEW.jsonl",
                        help="Stream per-scene wall/CPU, stage/API timings, counts and byte sizes to a new JSONL file; merged output only.")
    parser.add_argument("--force-merge", action="store_true",
                        help="Recompute even unchanged merges without rerunning models; requires --merge-only. Does not clear source state.")
    parser.add_argument("--rethreshold-scenes", action="store_true",
                        help="Apply YAML scene thresholds to retained frame evidence; requires --merge-only. "
                             "Scene/model pairs with no saved scene evidence are skipped. "
                             "May be combined with --rethreshold-markers.")
    parser.add_argument("--rethreshold-markers", action="store_true",
                        help="Apply YAML marker thresholds to complete cached frames; requires --merge-only. "
                             "Scene/model pairs with no saved marker evidence are skipped. "
                             "Never runs inference; rejects changed scene thresholds or insufficient retention.")
    parser.add_argument("--abandon-pending", action="store_true",
                        help="Abandon pending merge/cleanup plans, retaining applied changes and inference. "
                             "Alone: exit after abandonment; with --merge-only or cleanup: use current YAML policies. Honors --dry-run.")
    parser.add_argument("--cleanup-strategy", choices=("auto", "sql", "scenes"), default="auto",
                        help="Cleanup discovery: auto uses read-only SQL when available, else scans scenes; sql requires SQL; scenes scans scenes directly.")
    parser.add_argument("--cleanup-page-size", type=int, default=2000,
                        help="Maximum ID rows per SQL cleanup SELECT (default: 2000; no COUNT or OFFSET).")
    parser.add_argument("--cleanup-write-batch-size", type=int, default=200,
                        help="Maximum marker/scene IDs per cleanup mutation (default: 200). Tags are deleted one at a time.")
    parser.add_argument("--prune-empty-tags", action="store_true",
                        help="After successful merged sync, delete unused leaf tags in the selected prefixes.")
    parser.add_argument("--prune-prefixes", nargs="+", default=None, metavar="PREFIX",
                        help="Prefixes for empty-tag pruning (default: ai: m:). Global scope; m: requires managed group metadata.")
    parser.add_argument("--hierarchy", type=Path, default=app.HIERARCHY_FILE, metavar="YAML",
                        help="Shared tag hierarchy for every model; defaults to bundled tag_hierarchy.yml.")
    parser.add_argument("--marker-core-exclusions", type=Path, default=app.MARKER_CORE_EXCLUSIONS_FILE, metavar="YAML", help="Common/per-model excluded marker core labels.")
    parser.add_argument("--scene-tag-exclusions", type=Path,
                        default=app.SCENE_TAG_EXCLUSIONS_FILE, metavar="YAML",
                        help="Global blacklist for scene tags, marker cores/extras and hierarchy creation.")
    parser.add_argument("--no-hierarchy", action="store_true", help="Disable model/category parent synchronization; marker groups still use ai_markers.")
    parser.add_argument("--create-all-tags", action="store_true", help="With --sync-hierarchy-only, create full vocabularies.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--markers-only", action="store_true", help="Leave scene tags unchanged; also works with --merge-only.")
    outputs.add_argument("--scene-tags-only", action="store_true", help="Leave markers unchanged; also works with --merge-only.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sync-hierarchy-only", action="store_true", help="Organize tags without inference or scene/marker edits.")
    modes.add_argument("--validate-hierarchy", action="store_true", help="Validate vocabularies/hierarchies without Stash or weights.")
    modes.add_argument("--cleanup-markers-only", action="store_true", help="Remove excluded marker cores and global extras without inference or model files.")
    modes.add_argument("--cleanup-scene-tags-only", action="store_true", help="Apply scene blacklist without inference or model files.")
    modes.add_argument("--cleanup-only", action="store_true", help="Apply both blacklists without inference; full all-model cleanup removes globally banned tags in batches.")
    modes.add_argument("--download-models", action="store_true", help="Download selected model assets from Hugging Face without Stash or inference.")
    modes.add_argument("--merge-only", action="store_true", help="Merge current saved evidence without models/inference.")
    modes.add_argument("--sync-marker-groups-only", action="store_true",
                       help="Repair m:<core> / m:<core>:occurance groups on existing owned ai: markers; no inference or time changes.")
    modes.add_argument("--explain-tag", metavar="TAG", help="Show saved sources, scores and intervals for a raw/ai: tag; requires --scene-id; no Stash access.")
    modes.add_argument("--prune-empty-tags-only", action="store_true",
                       help="Remove unused leaf tag definitions without models/inference. Honors --dry-run.")
    modes.add_argument("--db-report", action="store_true",
                       help="Report state DB health, size and aggregate evidence/cache counts without Stash or model access.")
    modes.add_argument("--db-check", action="store_true",
                       help="Deep-check state DB logical consistency and compressed state without Stash or model access.")
    modes.add_argument("--db-update", action="store_true",
                       help="Upgrade the entire state DB to the latest supported schema without Stash or models. Honors --dry-run.")
    modes.add_argument("--db-compact", action="store_true",
                       help="Safely VACUUM the state DB while idle; refuses pending publication/cleanup plans. Honors --dry-run.")
    modes.add_argument("--anonymous-diagnostics", type=Path, metavar="NEW.json",
                       help="Write a local content-free developer diagnostics JSON from the state DB; never uploads it.")
    modes.add_argument("--model-report", type=Path, metavar="NEW.json",
                       help="Write an offline per-model utility and leave-one-model-out ablation report from saved evidence.")
    modes.add_argument("--tag-review-report", type=Path, metavar="NEW.json",
                       help="Write an offline tag review queue with reasons, examples and per-model counts; no inference or Stash requests.")
    modes.add_argument("--cache-report", type=Path, metavar="NEW.json",
                       help="Report scene/marker cache coverage against selected YAML models and plan targeted backfill; offline and read-only.")
    modes.add_argument("--run-report", type=Path, metavar="NEW.json",
                       help="Write offline run history and comparable-run profiling averages; no Stash, model or state DB writes.")
    modes.add_argument('--prune-run-history', action='store_true',
                       help='Prune finished run history by count/age; honors --dry-run, protects active runs and reuse proofs.')
    modes.add_argument("--reuse-report", type=Path, metavar="NEW.json",
                       help="Explain recorded cache/checkpoint decisions and reinference reasons from run history; offline and read-only.")

    return parser


def parse_args(app, argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(app)
    args = validate_args(app, parser, parser.parse_args(argv))
    defaults = dict(prefetch_batches=app.PREFETCH_BATCHES,
                    prepare_workers=app.PREPARE_WORKERS, merge_workers=4,
                    model_report_workers=min(4, os.cpu_count() or 1))
    used = {
        Command.INFERENCE: ('prefetch_batches', 'prepare_workers', 'merge_workers'),
        Command.MERGE: ('merge_workers',),
        Command.MODEL_REPORT: ('model_report_workers',),
    }.get(args.command, ())
    missing = [name for name in used if getattr(args, name) is None]
    configuration = {}
    if missing and (args.command is Command.INFERENCE or args.model_config.expanduser().is_file()):
        from stash_ai.config import load_model_configuration
        configuration = load_model_configuration(args.model_config.expanduser())
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, configuration.get(name, default) if name in used else default)
    return args
