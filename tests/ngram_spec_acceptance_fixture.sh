#!/bin/sh
set -eu

# N-gram (prompt-lookup) speculative-drafting acceptance fixture.
#
# Mirrors tests/dspark_acceptance_fixture.sh for the --ngram-spec drafter:
# for every case the greedy baseline and the n-gram speculative run must
# produce byte-identical output (losslessness is the core invariant), the
# verifier must report no errors, and a repetitive prompt must accept a
# meaningful number of drafted tokens.  A temperature run checks that the
# drafter stays inert for sampled decoding.

DS4_BIN=${DS4_BIN:-./ds4}
MODEL=${DS4_NGRAM_MODEL:-${DS4_TEST_MODEL:-./ds4flash.gguf}}
NGRAM=${DS4_NGRAM_SPEC_WIDTH:-5}
TOKENS=${DS4_NGRAM_FIXTURE_TOKENS:-64}
TEMPERATURE=${DS4_NGRAM_FIXTURE_TEMPERATURE:-1.0}
TOP_P=${DS4_NGRAM_FIXTURE_TOP_P:-0.95}
SEED=${DS4_NGRAM_FIXTURE_SEED:-12345}
REQUIRE_IDENTICAL=${DS4_NGRAM_FIXTURE_REQUIRE_IDENTICAL:-1}
REPETITIVE_MIN_ACCEPTED=${DS4_NGRAM_FIXTURE_REPETITIVE_MIN_ACCEPTED:-8}

case "$NGRAM" in
""|*[!0-9]*)
    echo "ngram-fixture: invalid DS4_NGRAM_SPEC_WIDTH=$NGRAM" >&2
    exit 1
    ;;
esac

file_bytes() {
    if stat -L -f %z "$1" >/dev/null 2>&1; then
        stat -L -f %z "$1"
    elif stat -Lc %s "$1" >/dev/null 2>&1; then
        stat -Lc %s "$1"
    else
        echo unknown
    fi
}

git_commit_label() {
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ "$commit" != unknown ] && ! git diff --quiet -- . 2>/dev/null; then
        commit="${commit}+dirty"
    fi
    echo "$commit"
}

print_metadata() {
    printf '# commit=%s\n' "$(git_commit_label)"
    printf '# hardware=%s\n' "$(uname -sm 2>/dev/null || echo unknown)"
    printf '# model=%s model_bytes=%s\n' "$MODEL" "$(file_bytes "$MODEL")"
    printf '# ngram_width=%s match_len=%s tokens=%s require_identical=%s repetitive_min_accepted=%s\n' \
        "$NGRAM" "${DS4_NGRAM_SPEC_SIZE:-default}" "$TOKENS" \
        "$REQUIRE_IDENTICAL" "$REPETITIVE_MIN_ACCEPTED"
    printf '# baseline_command=%s -m %s --tokens %s --temp 0 --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "$MODEL" "$TOKENS"
    printf '# ngram_command=DS4_DSPARK_STATS=1 %s --ngram-spec %s -m %s --tokens %s --temp 0 --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "$NGRAM" "$MODEL" "$TOKENS"
}

if [ ! -x "$DS4_BIN" ]; then
    echo "ngram-fixture: skipped, missing executable $DS4_BIN" >&2
    exit 0
fi
if [ ! -f "$MODEL" ]; then
    echo "ngram-fixture: skipped, missing model $MODEL" >&2
    exit 0
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-ngram-fixture.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

run_logged() {
    out=$1
    err=$2
    shift 2
    if "$@" >"$out" 2>"$err"; then
        return 0
    else
        status=$?
        cat "$err" >&2
        return "$status"
    fi
}

# Greedy losslessness + acceptance counters for one prompt.  When
# min_accepted is set, the case also requires that many accepted drafts.
run_case() {
    id=$1
    prompt=$2
    min_accepted=${3:-0}
    base_out="$tmpdir/$id.baseline.out"
    base_err="$tmpdir/$id.baseline.err"
    ngram_out="$tmpdir/$id.ngram.out"
    ngram_err="$tmpdir/$id.ngram.err"

    run_logged "$base_out" "$base_err" "$DS4_BIN" -m "$MODEL" \
        --tokens "$TOKENS" --temp 0 --nothink -p "$prompt"

    DS4_DSPARK_STATS=1 \
    run_logged "$ngram_out" "$ngram_err" "$DS4_BIN" --ngram-spec "$NGRAM" \
        -m "$MODEL" --tokens "$TOKENS" --temp 0 --nothink -p "$prompt"

    if ! cmp -s "$base_out" "$ngram_out"; then
        if [ "$REQUIRE_IDENTICAL" != 0 ]; then
            echo "ngram-fixture: greedy output mismatch for $id" >&2
            echo "baseline:" >&2
            sed 's/^/  /' "$base_out" >&2
            echo "ngram-spec:" >&2
            sed 's/^/  /' "$ngram_out" >&2
            return 1
        fi
    fi

    base_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$base_err" | tail -n 1)
    ngram_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$ngram_err" | tail -n 1)
    stats=$(grep 'ngram-spec stats' "$ngram_err" | tail -n 1 | sed 's/^ds4: ngram-spec stats //')
    if [ -z "$stats" ]; then
        echo "ngram-fixture: missing ngram-spec stats for $id" >&2
        return 1
    fi

    proposed=$(printf '%s\n' "$stats" | sed -n 's/.*proposed=\([0-9][0-9]*\).*/\1/p')
    accepted=$(printf '%s\n' "$stats" | sed -n 's/.*accepted_draft=\([0-9][0-9]*\).*/\1/p')
    errors=$(printf '%s\n' "$stats" | sed -n 's/.*errors=\([0-9][0-9]*\).*/\1/p')
    accept_rate=$(printf '%s\n' "$stats" | sed -n 's/.*accept_rate=\([0-9.]*\)%.*/\1/p')
    proposed=${proposed:-0}
    accepted=${accepted:-0}
    errors=${errors:-0}
    accept_rate=${accept_rate:-0}
    if [ "$errors" -ne 0 ]; then
        echo "ngram-fixture: verifier errors for $id: $stats" >&2
        return 1
    fi
    if [ "$accepted" -gt "$proposed" ]; then
        echo "ngram-fixture: inconsistent acceptance accounting for $id: $stats" >&2
        return 1
    fi
    if [ "$accepted" -lt "$min_accepted" ]; then
        echo "ngram-fixture: $id accepted $accepted drafts, below required $min_accepted: $stats" >&2
        return 1
    fi

    printf '%s\tgreedy_match=yes\tbaseline_tps=%s\tngram_tps=%s\tproposed=%s\taccepted=%s\taccept_rate=%s%%\n' \
        "$id" "${base_tps:-n/a}" "${ngram_tps:-n/a}" \
        "$proposed" "$accepted" "$accept_rate"
}

# Sampled decoding must be untouched by the drafter: same seed, same output.
run_sampled_case() {
    id=$1
    prompt=$2
    plain_out="$tmpdir/$id.sampled-plain.out"
    ngram_out="$tmpdir/$id.sampled-ngram.out"
    plain_err="$tmpdir/$id.sampled-plain.err"
    ngram_err="$tmpdir/$id.sampled-ngram.err"

    run_logged "$plain_out" "$plain_err" "$DS4_BIN" -m "$MODEL" \
        --tokens "$TOKENS" --temp "$TEMPERATURE" --top-p "$TOP_P" \
        --seed "$SEED" --nothink -p "$prompt"

    run_logged "$ngram_out" "$ngram_err" "$DS4_BIN" --ngram-spec "$NGRAM" \
        -m "$MODEL" --tokens "$TOKENS" --temp "$TEMPERATURE" --top-p "$TOP_P" \
        --seed "$SEED" --nothink -p "$prompt"

    if ! cmp -s "$plain_out" "$ngram_out"; then
        echo "ngram-fixture: sampled output mismatch for $id (drafter must stay inert at temperature $TEMPERATURE)" >&2
        return 1
    fi
    printf '%s\tsampled_match=yes\ttemp=%s top_p=%s seed=%s\n' \
        "$id" "$TEMPERATURE" "$TOP_P" "$SEED"
}

repetitive_prompt='Continue this pattern exactly, one line at a time:

The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy'
code_prompt='Complete this C file. Keep the same style.

int add(int a, int b) {
    return a + b;
}

int sub(int a, int b) {
    return a - b;
}

int mul(int a, int b) {
    return'
sparse_prompt='Explain Redis in one sentence.'

print_metadata
echo "id	greedy_match	baseline_tps	ngram_tps	acceptance"
run_case repetitive "$repetitive_prompt" "$REPETITIVE_MIN_ACCEPTED"
run_case code_complete "$code_prompt"
run_case sparse "$sparse_prompt"
run_sampled_case sampled_inert "$sparse_prompt"
