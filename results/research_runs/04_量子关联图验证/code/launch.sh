#!/usr/bin/env bash
set -uo pipefail
cd /home/js_cn/sensing || exit 1
base='双图研究框架/04_量子关联图验证'
profile="${1:?profile}"
if [ "$profile" = smoke ]; then
 venv/bin/python -u "$base/code/run.py" --smoke > "$base/logs/smoke.log" 2>&1
else
 venv/bin/python -u "$base/code/run.py" > "$base/logs/development.log" 2>&1
fi
rc=$?
printf '%s\n' "$rc" > "$base/logs/${profile}_exit.txt"
exit "$rc"
