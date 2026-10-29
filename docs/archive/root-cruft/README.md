# Root cruft, archived 2026-10-24

35 files moved here from the repository root with `git mv` (history preserved,
nothing deleted). To restore one: `git mv docs/archive/root-cruft/<name> .`.
These are historical prompts, plans, demos and one-off scripts; like the rest of
`docs/archive/` they overstate or no longer describe the repo and must not be
cited as current. `STATUS.md` is the live status document.

Scripts here were written to run from the repo root (e.g. `mle-star-command.js`
requires `./src/mle-star/index.js`); copy them back before running.

## Rule used

For every tracked root file the whole working tree was searched (tracked +
untracked-not-ignored, ~4,600 text files) for its exact filename and for its
extension-less stem (to catch `import foo` / `node foo` forms). A file moved only
if every remaining reference came from one of:

- the file itself, or `docs/archive/**`, `graphify-out/`, `.memdb/`;
- generated/tool state that only records names and is never read to open a file:
  `coverage/` (lcov HTML), `.codemap/state.json`, `.beads/issues.jsonl`,
  `.gitnexus/`;
- another file moved in the same batch (closed groups moved together).

Every other reference blocked the move (Makefile, package.json, `.github/**`,
Dockerfiles, compose files, scripts, `ralph.yml`, README/SETUP, anything else
under `docs/`, `claudedocs/`, `tests/`, `.serena/`). After the move the same
search was re-run over the live tree: zero remaining references to any moved
filename (two stem hits for `deployment-automation` are an unrelated
`--patterns` string and a Markdown anchor).

## Moved: no references at all (21)

`API_INTEGRATION_IMPLEMENTATION_PROMPT.md`, `AUTHENTICATION_IMPLEMENTATION_PROMPT.md`,
`CLEANUP_FIXES.md`, `MULTI_CLOUD_API_REFERENCE.md`, `Makefile.integration.patch`
(a snippet "to insert after line 154" of the Makefile, never applied by any target),
`NOVACRON_AGENT_WORKFLOW_PROMPT.md`, `PRODUCTION_COMPLETION_SWARM_PROMPT.md`,
`QUICK_FIX_CHECKLIST.md`, `README_COMPLETE.md`, `README_LARGE_FILES.md`,
`TODO_API_FIXES.md`, `TODO_COMPILATION_FIXES.md`, `completion.go.disabled`
(18-line stub, compiled nowhere), `deployment-automation.sh`, `install_go.ps1`,
`integration-validation.sh`, `setup_dependencies.ps1` (only points outward at
`start_development.ps1`), `setup_go_env.ps1`,
`ubuntu-24-04-architecture-blueprint.md`, `test_qdrant_mcp.py`,
`qdrant_mcp_demo.py` (only imports `qdrant_mcp_utils`, which stays).

## Moved: referenced only by each other (14, four closed groups)

- Claude-Flow launch prompts: `READY_FOR_CLAUDE_FLOW_EXECUTION.md`,
  `CLAUDE_FLOW_ULTIMATE_SWARM_PROMPT.md`, `EXECUTE_THIS_PROMPT.txt`,
  `execute_claude_flow_swarm.sh`.
- Plans: `FEATURE_IMPLEMENTATION_PLANS.md`, `IMPLEMENTATION_PLAN.md`.
- MLE-Star installer family: `mle-star.sh`, `mle-star-command.js`,
  `claude-flow-mle-star.js`, `integrate-mle-star.js`, `register-mle-star.js`,
  `test-mle-star-direct.js`. Nothing outside the group requires them; the
  `src/mle-star/` library and `package.json`'s `test:integration:mle-star` script
  do not use them.
- `code_memory.py` and `novabay_code_memory.py` (the latter is its only importer).

## Deliberately left at the root

- `PROMPT.md`: `ralph.yml` sets `prompt_file: PROMPT.md`.
- Qdrant "code memory" subsystem: `.github/workflows/update-code-memory.yml`
  (nightly + manual) runs `create_collection.py`, `index_key_files_mcp.py`,
  `test_qdrant_connection.py` and imports `qdrant_mcp_utils` (also imported by
  `web_interface/app.py`); `CODE_MEMORY.md` documents the rest, so
  `external_knowledge.py`, `index_external_knowledge.py`, `qdrant_code_memory_demo.py`,
  `set_openai_key.{py,sh,ps1}`, `setup_code_memory.py`,
  `store_repo_in_qdrant_mcp.py` and `code_memory_requirements.txt` stay with it.
  Whether that workflow is still wanted is a product call, not a cleanup call.
- `healthcheck.js`: `deployments/docker/Dockerfile.dwcp-v3` runs `node healthcheck.js`.
- `start_development.sh` / `.ps1`: listed as the dev start command in
  `.serena/memories/suggested_commands.md`.
- `test-auth.sh`: cited by `claudedocs/AUTH_SYSTEM_INTEGRATION_REPORT.md`.
- `AGENT_29_QUICK_REFERENCE.txt`, `BENCHMARK_RESULTS_QUICK_REFERENCE.md`: each is
  named by a non-archive report under `docs/`. Safe to move once those reports are
  annotated or archived.
- `SETUP.md` (named in `docs/SWARM_ANALYSIS_REPORT.md`) and `README-LARGE-FILES.md`
  (linked from `SETUP.md`).
- `QUICK_DEPLOY.md`: an 8-line deprecation stub kept on purpose (bead `novacron-5ny`)
  so old links land somewhere.
- `CRUSH.md` (read by the Crush CLI by name), `.codeboardingignore` (tool config).
- Never-move list: `README.md`, `STATUS.md`, `CLAUDE.md`, `AGENTS.md`, `LICENSE`,
  `Makefile`, `go.mod`, `go.sum`, `package.json`, `pyproject.toml`, `uv.lock`,
  `requirements.txt`, `docker-compose*` / `qdrant-docker-compose.yml`, `.env*`,
  `.gitignore`, `.gitattributes`, `WATCHDOG.yml`, `playwright.config.ts`, `ralph.yml`.

## Known stale mentions (not edited)

Generated files still name some moved files and will refresh on regeneration:
`coverage/**` (MLE-Star scripts), `.codemap/state.json` (Python files), and
`.beads/issues.jsonl` (issue prose).
