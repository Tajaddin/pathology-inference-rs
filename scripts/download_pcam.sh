#!/usr/bin/env bash
# Download the nine PatchCamelyon files (Zenodo record 2546921, CC0) into DIR and verify their MD5 checksums.
#
#   scripts/download_pcam.sh [DIR]          default data/raw; about 8.1 GB
#
# A file already present with the right checksum is skipped; an interrupted download resumes (curl -C -). A file
# that still fails its checksum after downloading is deleted and the script stops.
# Environment: PCAM_BASE (the Zenodo files URL), RETRIES (5).
set -euo pipefail

dir=${1:-data/raw}
base=${PCAM_BASE:-https://zenodo.org/api/records/2546921/files}
retries=${RETRIES:-5}
mkdir -p "$dir"

checksums=$(cat <<'EOF'
1571f514728f59376b705fc836ff4b63 camelyonpatch_level_2_split_train_x.h5.gz
35c2d7259d906cfc8143347bb8e05be7 camelyonpatch_level_2_split_train_y.h5.gz
5a3dd671e465cfd74b5b822125e65b0a camelyonpatch_level_2_split_train_meta.csv
d5b63470df7cfa627aeec8b9dc0c066e camelyonpatch_level_2_split_valid_x.h5.gz
2b85f58b927af9964a4c15b8f7e8f179 camelyonpatch_level_2_split_valid_y.h5.gz
67589e00a4a37ec317f2d1932c7502ca camelyonpatch_level_2_split_valid_meta.csv
d8c2d60d490dbd479f8199bdfa0cf6ec camelyonpatch_level_2_split_test_x.h5.gz
60a7035772fbdb7f34eb86d4420cf66a camelyonpatch_level_2_split_test_y.h5.gz
3455fd69135b66734e1008f3af684566 camelyonpatch_level_2_split_test_meta.csv
EOF
)
[[ -n ${PCAM_CHECKSUMS:-} ]] && checksums=$(cat "$PCAM_CHECKSUMS")

ok() { echo "$1  $dir/$2" | md5sum -c --status; }

while read -r md5 name; do
  if [[ -s $dir/$name ]] && ok "$md5" "$name"; then
    echo "ok       $name"
    continue
  fi
  echo "download $name"
  curl -fsSL --retry "$retries" --retry-delay 5 --retry-all-errors -C - -o "$dir/$name" "$base/$name/content"
  if ! ok "$md5" "$name"; then
    rm -f "$dir/$name"
    echo "checksum mismatch for $name; deleted, re-run to download it again" >&2
    exit 1
  fi
  echo "verified $name"
done <<<"$checksums"
