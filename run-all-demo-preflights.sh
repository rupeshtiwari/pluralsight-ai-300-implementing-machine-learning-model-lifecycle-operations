#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
for f in demos/m*_demo.py; do
  python3 "$f" >/dev/null
  echo "✔ Preflight PASS: $f"
done
echo "ALL 20 DEMO PREFLIGHTS PASSED!"
