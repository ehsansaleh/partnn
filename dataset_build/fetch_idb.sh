#!/bin/bash
# Download the raw PartMC-MOSAIC scenario tars from the Illinois Data Bank
# (doi:10.13012/B2IDB-2774261_V1, CC0) and split them into per-scenario
# tarballs:
#   <DEST>/scenario_N.tar.gz   (~380 MB each; unpacks to scenario_N/scenario_N_0001_000000TT.nc)
#
# The per-scenario tarballs stay compressed (281 GB total instead of ~1 TB
# unpacked); the h5 build step unpacks them as needed.
#
# Each outer tar is ~28 GB (100 scenarios). It is streamed straight into
# `tar -x`, so it is never saved whole. Run on a machine with internet access
# (inside tmux or screen); for many tars Globus (collection
# 7e8c5183-112b-472b-9f86-751a60dcfb75, path /IDB-2774261/) is the alternative.
#
# Usage:
#   bash fetch_idb.sh DEST [RANGE ...]      (RANGE like 000_099; default: all ten)
set -euo pipefail

DEST=${1:?usage: fetch_idb.sh DEST [RANGE ...]   (RANGE like 000_099)}
shift
declare -A IDS=(
  [000_099]=j331s [100_199]=zvxb9 [200_299]=2l02k [300_399]=rj2vw [400_499]=4aoaq
  [500_599]=gvvvv [600_699]=awa5b [700_799]=n50a6 [800_899]=m1zv0 [900_999]=24sex
)
RANGES=("$@")
[[ ${#RANGES[@]} -gt 0 ]] || RANGES=(000_099 100_199 200_299 300_399 400_499 500_599 600_699 700_799 800_899 900_999)
# The Data Bank rejects requests without a browser-like User-Agent.
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

mkdir -p "${DEST}"
cd "${DEST}"
for RANGE in "${RANGES[@]}"; do
  ID=${IDS[$RANGE]:?unknown range ${RANGE}}
  TAR="scenarios_${RANGE}.tar"
  if [[ -f "${TAR}.done" ]]; then
    echo "== ${RANGE}: already done"; continue
  fi
  echo "== ${RANGE}: downloading and splitting ${TAR} ($(date))"
  # A failed stream restarts the whole range (tar overwrites the partial
  # scenario files); each retry re-requests the Data Bank URL, which hands out
  # a fresh presigned S3 link.
  OK=0
  for TRY in 1 2 3 4 5; do
    if curl -L --fail -sS -A "${UA}" "https://databank.illinois.edu/datafiles/${ID}/download" \
        | tar -xvf - > "${TAR}.list"; then
      OK=1; break
    fi
    echo "   download failed (try ${TRY}); retrying in 30 s"; sleep 30
  done
  [[ ${OK} -eq 1 ]] || { echo "== ${RANGE}: download FAILED"; exit 1; }
  N=$(grep -c '\.tar\.gz$' "${TAR}.list")
  [[ ${N} -eq 100 ]] || { echo "== ${RANGE}: expected 100 scenarios, got ${N}"; exit 1; }
  rm -f "${TAR}" "${TAR}.list"
  touch "${TAR}.done"
  echo "== ${RANGE}: done, ${N} scenarios ($(ls scenario_*.tar.gz | wc -l) total, $(du -sh . | cut -f1))"
done
echo "== all done ($(date))"
