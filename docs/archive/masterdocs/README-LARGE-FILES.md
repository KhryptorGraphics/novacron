# Large Files Management

This repository contains large files (>100MB) that are automatically split and reassembled.

## How It Works

### For Repository Maintainers

**Splitting large files before commit:**
```bash
bash scripts/split-large-files.sh
git add .github/large-files/
git commit -m "Add split large files"
```

### For Repository Users

**Automatic reassembly (recommended):**

When you clone or pull the repository, large files are automatically reassembled via Git hooks.

```bash
git clone https://github.com/khryptorgraphics/novacron.git
cd novacron
bash scripts/setup-hooks.sh  # Install hooks
```

**Manual reassembly:**
```bash
bash scripts/join-large-files.sh
```

## Large Files in This Repository

- `src/neural/knowledge_base_20251011_124231.json` (102 MB)
- `docs/neural_analysis_report_20251011_124231.json`

## Technical Details

- **Chunk size:** 50 MB per part
- **Verification:** SHA-256 checksums ensure integrity
- **Storage:** Split parts stored in `.github/large-files/`
- **Automation:** Git hooks handle reassembly automatically

## Troubleshooting

**Files not reassembling automatically:**
```bash
bash scripts/setup-hooks.sh  # Reinstall hooks
bash scripts/join-large-files.sh  # Manual reassembly
```

**Checksum verification fails:**
```bash
# Re-clone the repository
git clone https://github.com/khryptorgraphics/novacron.git
```

## Why Not Git LFS?

This approach uses standard Git features without requiring Git LFS installation or additional service configuration, making it more accessible for all contributors.
