#!/usr/bin/env bash
set -euo pipefail

forbidden='^(tinystoragedrivers|tinystoragedrivers-(core|document|kv|blob|queue|secrets|connect|sqlite|file|mongodb|s3|rabbitmq|redis)|rusqlite|libsqlite3-sys|mongodb|bson|aws-sdk-s3|lapin|amq-protocol|sqlx|redb|sled|rocksdb)$'

mapfile -t packages < <(
  cargo metadata --format-version 1 --no-deps \
    | jq -r '.packages[] | select(any(.targets[]; any(.kind[]; . == "lib"))) | .name'
)

if [[ "${#packages[@]}" -eq 0 ]]; then
  echo "cargo metadata found no library packages" >&2
  exit 1
fi

for package in "${packages[@]}"; do
  echo "Checking normal/build dependencies for ${package}"
  tree="$(cargo tree --locked -p "${package}" -e normal,build --all-features --prefix none)"
  if matches="$(printf '%s\n' "${tree}" | grep -E "${forbidden}" || true)"; [[ -n "${matches}" ]]; then
    printf 'forbidden storage engine dependency in %s:\n%s\n' "${package}" "${matches}" >&2
    exit 1
  fi
done
