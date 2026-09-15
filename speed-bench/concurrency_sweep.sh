#!/bin/sh
# One process per cell of the concurrency x context grid.
#
# A single process cannot walk the whole grid honestly: prefill fills the
# unified buffer cache with n-gram pages, the resident weight mapping cannot be
# evicted to make room, and the system starts compressing session memory
# instead.  A fresh process per cell gives every cell the same starting
# conditions and the same memory budget.
#
# usage: speed-bench/concurrency_sweep.sh CSV [extra session_concurrency_bench options]

set -e
CSV=${1:?usage: concurrency_sweep.sh CSV [options]}
shift || true
BENCH=./speed-bench/session_concurrency_bench
CONCURRENCY=${CONCURRENCY:-"1 2 4 8 16"}
CONTEXTS=${CONTEXTS:-"0 4096 16384 32768 65536 131072"}

for ctx in $CONTEXTS; do
    for conc in $CONCURRENCY; do
        echo "=== concurrency $conc, context $ctx ==="
        "$BENCH" --concurrency "$conc" --ctx "$ctx" --csv "$CSV" "$@" || \
            echo "cell concurrency=$conc ctx=$ctx failed with $?"
    done
done
echo
echo "wrote $CSV"
