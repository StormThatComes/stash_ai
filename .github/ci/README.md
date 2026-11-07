# CI job selection

Every pull request runs `Select CI jobs`, which reads the complete merge-base diff
from Git. Renames contribute both paths. Mixed changes combine their requirements.
Unavailable diffs, unclassified files and shared CI/dependency changes select all jobs.

| Changed files | PR jobs |
| --- | --- |
| Documentation only | Selection and final result |
| Numeric recognition modules listed in `tests/ci_jobs.py` | All three Linux pytest shards |
| Solver modules listed in `tests/ci_jobs.py` | Linux shards and Windows core, including solver process tests |
| Other recognition code | Linux shards, CPU model smoke and both Windows lifecycle groups |
| Tests | Linux shards and Windows jobs using those tests or their imported fixtures |
| Updater | Linux shards and minimal-profile updater tests on both operating systems |
| Shared application code/configuration | Linux, updater, model smoke and Windows lifecycle tests |
| Requirements, constraints, CI, shared fixtures, or unknown paths | Full matrix |

The Linux pytest jobs keep the complete regression suite. Module prefixes only
select the additional jobs. Unclassified recognition modules run both Windows groups
and model smoke. Adding or deleting installed recognition modules also runs the
minimal-profile updater tests. Both standalone benchmarks run once when Linux tests
are selected. Each job-selection summary records the reasons for the choice.

Each push to `main` schedules the full matrix and cancels any unfinished run from
an earlier push to `main`. A new PR revision cancels the previous run for that PR.
Manual runs use the full matrix and run independently.

Use `CI result` as the required check when configuring branch protection. It
verifies that all selected jobs succeeded and accepts jobs intentionally omitted
by the plan.

Run the planner regressions locally with
`python -m pytest tests/test_ci_jobs.py tests/test_workflow_config.py`.
Review the path rules when introducing new native
backends or operating-system-specific behavior into numeric recognition modules.
