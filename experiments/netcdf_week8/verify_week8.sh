#!/usr/bin/env bash
set -euo pipefail

# Run demo and capture just the value lines
out=$(experiments/netcdf_week8/run_week8_demo.sh | awk '
  $1 ~ /^[0-9.-]/ || $1=="20.5" || $1=="1" { print }
')

# Expected flattened values in order of the five runs:
# temp[:]
# temp[1:5]
# grid[:,:]
# grid[:,1:3]
# grid[:,::2]
expected=$(
cat <<'EOT'
20.5 21 19.8 18.9 22.2 23.1
21 19.8 18.9 22.2
1 1.1 1.2 2 2.1 2.2
1.1 1.2 2.1 2.2
1 1.2 2 2.2
EOT
)

# Diff
diff -u <(printf "%s\n" "$expected") <(printf "%s\n" "$out") && echo "VERIFY: OK"
