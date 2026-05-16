#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

gitnexus_bin="${GITNEXUS_BIN:-gitnexus}"
timeout_bin="${TIMEOUT_BIN:-timeout}"
analyze_timeout="${GITNEXUS_ANALYZE_TIMEOUT:-1200s}"
refresh_commit_message="${GITNEXUS_REFRESH_COMMIT_MESSAGE:-Refresh GitNexus metadata}"

head_before="$(git rev-parse HEAD)"

run_analyze() {
	if command -v "$timeout_bin" >/dev/null 2>&1; then
		"$timeout_bin" "$analyze_timeout" "$gitnexus_bin" analyze
	else
		"$gitnexus_bin" analyze
	fi
}

meta_last_commit() {
	node -e "const fs=require('fs'); const p='.gitnexus/meta.json'; if (!fs.existsSync(p)) process.exit(1); console.log(JSON.parse(fs.readFileSync(p,'utf8')).lastCommit || '')"
}

patch_gitnexus_commit() {
	local commit="$1"
	node - "$commit" "$repo_root" <<'NODE'
const fs = require('fs');
const os = require('os');
const path = require('path');

const [commit, repoRoot] = process.argv.slice(2);
const metaPath = path.join(repoRoot, '.gitnexus', 'meta.json');
const now = new Date().toISOString();

const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
meta.lastCommit = commit;
meta.indexedAt = now;
fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2) + '\n');

const registryPath = path.join(os.homedir(), '.gitnexus', 'registry.json');
if (fs.existsSync(registryPath)) {
  const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
  const entry = registry.find((item) => path.resolve(item.path) === path.resolve(repoRoot));
  if (entry) {
    entry.lastCommit = commit;
    entry.indexedAt = now;
    entry.stats = meta.stats;
    fs.writeFileSync(registryPath, JSON.stringify(registry, null, 2) + '\n');
  }
}
NODE
}

analyze_rc=0
run_analyze || analyze_rc=$?

if [[ "$analyze_rc" -ne 0 ]]; then
	if ! current_meta_commit="$(meta_last_commit 2>/dev/null)" || [[ "$current_meta_commit" != "$head_before" ]]; then
		echo "gitnexus analyze failed before producing an index for $head_before (exit $analyze_rc)" >&2
		exit "$analyze_rc"
	fi
	echo "gitnexus analyze exited $analyze_rc after updating metadata; continuing with status repair" >&2
fi

mapfile -t changed_files < <(git diff --name-only)
allowed_changes=()
unexpected_changes=()
for changed_file in "${changed_files[@]}"; do
	case "$changed_file" in
		AGENTS.md|CLAUDE.md)
			allowed_changes+=("$changed_file")
			;;
		*)
			unexpected_changes+=("$changed_file")
			;;
	esac
done

if ((${#unexpected_changes[@]} > 0)); then
	printf 'gitnexus analyze produced unexpected working-tree changes:\n' >&2
	printf '  %s\n' "${unexpected_changes[@]}" >&2
	exit 1
fi

if ((${#allowed_changes[@]} > 0)); then
	git add -- "${allowed_changes[@]}"
	git commit -m "$refresh_commit_message"
fi

head_after="$(git rev-parse HEAD)"
if [[ "$(meta_last_commit)" != "$head_after" ]]; then
	patch_gitnexus_commit "$head_after"
fi

"$gitnexus_bin" status
