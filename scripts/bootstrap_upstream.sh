#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ ! -e upstream ]] || { echo 'upstream exists; use a clean checkout'; exit 1; }
git clone --filter=blob:none https://github.com/qiancheng0/ToolRL.git upstream
git -C upstream checkout 8cee13ec0ca72f0461da372a93a6fd8140dbb840
git -C upstream apply "$ROOT/notes/local-adaptations.patch"
mkdir -p results logs .cache
echo 'Pinned source and local compatibility/final-checkpoint patches ready.'
