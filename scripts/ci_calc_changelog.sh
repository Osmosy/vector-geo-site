#!/usr/bin/env bash
# Если менялся код расчёта и менялись пакеты примеров, в diff обязан быть CHANGELOG-CALC.md.
set -euo pipefail

if [ -z "${GITHUB_BASE_REF:-}" ]; then
  echo "нет базовой ветки — проверка пропущена (локальный запуск)"
  exit 0
fi

base="origin/${GITHUB_BASE_REF}"
git fetch --no-tags --depth=50 origin "$GITHUB_BASE_REF" >/dev/null 2>&1 || true
changed=$(git diff --name-only "$base"...HEAD || true)

if echo "$changed" | grep -qE '^src/geosite/.*\.py$' && echo "$changed" | grep -qE '^examples/.*site_pack\.json$'; then
  if ! echo "$changed" | grep -q '^CHANGELOG-CALC\.md$'; then
    echo "код расчёта и примеры изменились, а CHANGELOG-CALC.md — нет" >&2
    exit 1
  fi
fi
echo "проверка журнала расчёта пройдена"
