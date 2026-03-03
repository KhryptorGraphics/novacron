# NovaCron Quick Setup Guide

## First Time Setup After Clone

When you clone this repository, run:

```bash
bash scripts/setup-hooks.sh
```

This installs Git hooks that automatically reassemble large files (>100MB) when you:
- Clone the repository
- Switch branches
- Pull changes

## Large Files Management

### Automatic (Recommended)

Large files are automatically handled via Git hooks. No manual intervention needed!

### Manual Commands

**Reassemble large files manually:**
```bash
bash scripts/join-large-files.sh
```

**Split large files before committing (maintainers only):**
```bash
bash scripts/split-large-files.sh
git add .github/large-files/
git commit -m "Update large files"
```

## What Files Are Split?

- `src/neural/knowledge_base_20251011_124231.json` (102 MB)
- `docs/neural_analysis_report_20251011_124231.json`

These files are split into 50MB chunks stored in `.github/large-files/` and automatically reassembled.

## Verification

Check if files are properly reassembled:
```bash
sha256sum -c .github/large-files/knowledge_base_20251011_124231.json.sha256
sha256sum -c .github/large-files/neural_analysis_report_20251011_124231.json.sha256
```

## For More Details

See [README-LARGE-FILES.md](./README-LARGE-FILES.md) for complete documentation.
