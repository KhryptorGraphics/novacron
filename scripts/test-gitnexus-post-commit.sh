#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
script_under_test="$repo_root/scripts/gitnexus-post-commit.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

export HOME="$tmp_dir/home"
mkdir -p "$HOME/.gitnexus"

test_repo="$tmp_dir/repo"
mkdir -p "$test_repo/.gitnexus" "$tmp_dir/bin"
cd "$test_repo"

git init -q
git config user.name "GitNexus Test"
git config user.email "gitnexus-test@example.invalid"

cat > AGENTS.md <<'EOF'
GitNexus stats: old
EOF
cat > CLAUDE.md <<'EOF'
GitNexus stats: old
EOF
cat > README.md <<'EOF'
test repo
EOF
cat > .gitignore <<'EOF'
.gitnexus/
EOF
git add AGENTS.md CLAUDE.md README.md .gitignore
git commit -q -m "initial"

initial_head="$(git rev-parse HEAD)"
cat > .gitnexus/meta.json <<EOF
{
  "repoPath": "$test_repo",
  "lastCommit": "$initial_head",
  "indexedAt": "2026-01-01T00:00:00.000Z",
  "stats": {
    "files": 3,
    "nodes": 1,
    "edges": 0,
    "communities": 0,
    "processes": 0,
    "embeddings": 0
  }
}
EOF
cat > "$HOME/.gitnexus/registry.json" <<EOF
[
  {
    "name": "repo",
    "path": "$test_repo",
    "storagePath": "$test_repo/.gitnexus",
    "indexedAt": "2026-01-01T00:00:00.000Z",
    "lastCommit": "$initial_head",
    "stats": {
      "files": 3,
      "nodes": 1,
      "edges": 0,
      "communities": 0,
      "processes": 0,
      "embeddings": 0
    }
  }
]
EOF

cat > "$tmp_dir/bin/gitnexus" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
	analyze)
		head="$(git rev-parse HEAD)"
		node - "$head" <<'NODE'
const fs = require('fs');
const head = process.argv[2];
const meta = JSON.parse(fs.readFileSync('.gitnexus/meta.json', 'utf8'));
meta.lastCommit = head;
meta.indexedAt = '2026-01-01T00:00:01.000Z';
fs.writeFileSync('.gitnexus/meta.json', JSON.stringify(meta, null, 2) + '\n');
NODE
		sed -i 's/old/new/' AGENTS.md CLAUDE.md
		;;
	status)
		head="$(git rev-parse HEAD)"
		meta_head="$(node -e "console.log(JSON.parse(require('fs').readFileSync('.gitnexus/meta.json','utf8')).lastCommit)")"
		if [[ "$head" != "$meta_head" ]]; then
			echo "Status: stale"
			exit 1
		fi
		echo "Status: up-to-date"
		;;
	*)
		echo "unexpected gitnexus command: $*" >&2
		exit 2
		;;
esac
EOF
chmod +x "$tmp_dir/bin/gitnexus"

PATH="$tmp_dir/bin:$PATH" GITNEXUS_ANALYZE_TIMEOUT=30s "$script_under_test" >/tmp/gitnexus-post-commit-test.log

if [[ "$(git log --format=%s -1)" != "Refresh GitNexus metadata" ]]; then
	echo "expected refresh commit" >&2
	exit 1
fi

if [[ "$(git status --short)" != "" ]]; then
	git status --short >&2
	exit 1
fi

final_head="$(git rev-parse HEAD)"
meta_head="$(node -e "console.log(JSON.parse(require('fs').readFileSync('.gitnexus/meta.json','utf8')).lastCommit)")"
registry_head="$(node -e "console.log(JSON.parse(require('fs').readFileSync(process.env.HOME+'/.gitnexus/registry.json','utf8'))[0].lastCommit)")"

if [[ "$meta_head" != "$final_head" || "$registry_head" != "$final_head" ]]; then
	echo "expected meta and registry to point at final head" >&2
	exit 1
fi
