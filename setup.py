#!/usr/bin/env python3
"""
NovaCron Setup Script - Python version
Automatically reassembles large files after cloning
"""

import os
import sys
import subprocess
import hashlib
from pathlib import Path

def reassemble_files():
    """Reassemble split files"""
    print("📦 Reassembling large files...")
    
    splits_dir = Path(".splits")
    if not splits_dir.exists():
        print("ℹ️ No split files found. Repository is ready to use.")
        return True
    
    # Find all .original files
    for original_file in splits_dir.rglob("*.original"):
        with open(original_file, 'r') as f:
            target_path = f.read().strip()
        
        print(f"🔧 Reassembling {target_path}...")
        
        # Create target directory
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Get base name and directory
        base_name = target.name
        dir_path = original_file.parent
        
        # Find and sort part files
        part_files = sorted(dir_path.glob(f"{base_name}.part-*"))
        
        # Concatenate parts
        with open(target, 'wb') as outfile:
            for part_file in part_files:
                with open(part_file, 'rb') as infile:
                    outfile.write(infile.read())
        
        # Verify checksum if available
        md5_file = dir_path / f"{base_name}.md5"
        if md5_file.exists():
            print("🔍 Verifying checksum...")
            with open(md5_file, 'r') as f:
                expected_md5 = f.read().split()[0]
            
            # Calculate actual MD5
            md5_hash = hashlib.md5()
            with open(target, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            actual_md5 = md5_hash.hexdigest()
            
            if expected_md5 == actual_md5:
                print(f"✅ Checksum verified for {target_path}")
            else:
                print(f"❌ Checksum mismatch for {target_path}!")
                return False
    
    # Clean up splits directory
    print("🧹 Cleaning up split files...")
    import shutil
    shutil.rmtree(splits_dir)
    
    print("✨ Reassembly complete!")
    return True

def main():
    """Main setup function"""
    print("╔══════════════════════════════════════╗")
    print("║     NovaCron Automatic Setup         ║")
    print("╚══════════════════════════════════════╝")
    print()
    
    # Reassemble files
    if not reassemble_files():
        print("❌ Setup failed!")
        sys.exit(1)
    
    print()
    print("╔══════════════════════════════════════╗")
    print("║        Setup Complete! 🎉            ║")
    print("╚══════════════════════════════════════╝")
    print()
    print("📝 Next steps:")
    print("  1. Install dependencies:")
    print("     npm install")
    print("     cd frontend && npm install")
    print()
    print("  2. Start development:")
    print("     npm run dev")
    print()

if __name__ == "__main__":
    main()
