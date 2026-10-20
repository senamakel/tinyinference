#!/usr/bin/env bash
set -euo pipefail

forbidden='^(tinystoragedrivers|tinystoragedrivers-(core|document|kv|blob|queue|secrets|connect|sqlite|file|mongodb|s3|rabbitmq|redis)|rusqlite|libsqlite3-sys|mongodb|bson|aws-sdk-s3|lapin|amq-protocol|sqlx|redb|sled|rocksdb)$'
if [[ -n "${CANCEL_AWARE_CARGO_WRAPPER:-}" ]]; then
  cargo_cmd=("${CANCEL_AWARE_CARGO_WRAPPER}" cargo)
else
  cargo_cmd=(cargo)
fi

matches_forbidden_package() {
  awk '{ print $1 }' <<<"$1" | grep -xE "$forbidden" || true
}

if [[ "${1:-}" == "--self-test" ]]; then
  allowed_tree=$'tinystoragedrivers-api v1.0.0 (git+https://example.test/storage)\ntinystoragedrivers-types v1.0.0 (git+https://example.test/storage)'
  forbidden_tree=$'tinystoragedrivers-core v1.0.0 (git+https://example.test/storage)\ntinystoragedrivers-secrets v1.0.0 (git+https://example.test/storage)'
  if [[ -n "$(matches_forbidden_package "$allowed_tree")" ]]; then
    echo "allowed API/types dependencies were rejected" >&2
    exit 1
  fi
  if [[ "$(matches_forbidden_package "$forbidden_tree")" != $'tinystoragedrivers-core\ntinystoragedrivers-secrets' ]]; then
    echo "Cargo tree package extraction missed a forbidden engine dependency" >&2
    exit 1
  fi
  echo "storage dependency tree matcher self-test passed"
  exit 0
fi

mapfile -t packages < <(
  "${cargo_cmd[@]}" metadata --locked --format-version 1 --no-deps \
    | jq -r '.packages[] | select(any(.targets[]; any(.kind[]; . == "lib"))) | .name'
)

if [[ "${#packages[@]}" -eq 0 ]]; then
  echo "cargo metadata found no library packages" >&2
  exit 1
fi

for package in "${packages[@]}"; do
  echo "Checking normal/build dependencies for ${package}"
  tree="$("${cargo_cmd[@]}" tree --locked -p "${package}" -e normal,build --all-features --prefix none)"
  if matches="$(matches_forbidden_package "${tree}")"; [[ -n "${matches}" ]]; then
    printf 'forbidden storage engine dependency in %s:\n%s\n' "${package}" "${matches}" >&2
    exit 1
  fi
done
