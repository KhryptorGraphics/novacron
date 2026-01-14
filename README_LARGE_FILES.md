# Large Files Notice

This repository contains split files to comply with GitHub's file size limitations.

## Automatic Reassembly

After cloning this repository, run the following command to reassemble large files:

```bash
./scripts/reassemble-files.sh
```

This will automatically restore:
- Frontend webpack cache files (needed for production builds)
- Binary executables
- Other large assets

## Manual Reassembly (if needed)

If the automatic script fails, you can manually reassemble files:

```bash
# Navigate to the repository
cd novacron

# Reassemble a specific file
cat .github/split-files/frontend/.next/cache/webpack/client-production/0.pack.part* > frontend/.next/cache/webpack/client-production/0.pack
```

## Why are files split?

GitHub has a 100MB file size limit. Some build artifacts exceed this limit, so they're split into 40MB chunks for safe storage. The reassembly process is automatic and preserves file integrity through MD5 verification.

## Files affected

- `frontend/.next/cache/webpack/client-production/0.pack` (69MB)
- `frontend/.next/cache/webpack/server-production/0.pack` (61MB)
- `acli.exe` (61MB)

Note: Node modules binaries will be automatically installed when you run `npm install` in the frontend directory.