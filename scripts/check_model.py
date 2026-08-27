#!/usr/bin/env python3
import sys
from huggingface_hub import model_info, scan_cache_dir

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-AWQ"
try:
    m = model_info(model_id)
    files = [s.rfilename for s in (m.siblings or [])]
    total_gb = sum(getattr(s, "size", 0) or 0 for s in (m.siblings or [])) / 1e9
    print(f"OK: {m.id}")
    print(f"Files: {len(files)}  approx_size={total_gb:.1f} GB")
    for f in files[:8]:
        print(f"  {f}")
    if len(files) > 8:
        print(f"  ... +{len(files) - 8} more")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

cache = scan_cache_dir()
for repo in cache.repos:
    if model_id.replace("/", "--") in repo.repo_id or model_id in repo.repo_id:
        print(f"Cached: {repo.repo_id} ({repo.size_on_disk / 1e9:.1f} GB)")
