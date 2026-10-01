#!/usr/bin/env python3
"""Execute the real n-gram (prompt-lookup) proposal helper against mock contexts.

Run with Python's standard library and a C compiler; no model or GPU is
needed.  Mirrors the extraction approach of test_dspark_eos_contract.py so
the production drafter, not a copy of it, is what gets checked:
most-recent-occurrence lookup, draft width clamp, boundary behavior, and
the DS4_NGRAM_SPEC_SIZE override.
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'ds4.c'
FUNCTIONS = ('ds4_ngram_env_u32', 'ds4_ngram_match_len', 'ds4_ngram_propose')


def extract_function(source, name):
    matches = list(re.finditer(r'static \w+ ' + re.escape(name) + r'\s*\(', source))
    if len(matches) != 1:
        raise ValueError('require exactly one definition of ' + name)
    start = matches[0].start()
    opening = source.index('{', start)
    token = re.compile(r'/\*.*?\*/|"[^"]*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match.group() == '{':
            depth += 1
        elif match.group() == '}':
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError('unterminated function ' + name)


HARNESS = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int *v; int len, cap; } ds4_tokens;

{env_fn}
{match_fn}
{propose_fn}

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d %s\n", __func__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static int run(int *ctx, int len, int max_draft, int *drafts) {
    ds4_tokens t = { ctx, len, 0 };
    return ds4_ngram_propose(&t, max_draft, drafts);
}

static void test_unseen_suffix_misses(void) {
    int ctx[] = { 5, 6, 7, 8, 9 };
    int drafts[16];
    CHECK(run(ctx, 5, 4, drafts) == 0);
}

static void test_most_recent_occurrence_wins(void) {
    /* suffix "3,4,50" occurs at 0 (followed by 7) and at 7 (followed by 8);
     * the backward scan must pick the newer one. */
    int ctx[] = { 3,4,50,7, 5,5,5, 3,4,50,8, 2,2,2, 3,4,50 };
    int drafts[16];
    int n = run(ctx, 17, 2, drafts);
    CHECK(n == 2);
    CHECK(drafts[0] == 8 && drafts[1] == 2);
    n = run(ctx, 17, 1, drafts);
    CHECK(n == 1);
    CHECK(drafts[0] == 8);
}

static void test_repetition_replays_tail(void) {
    /* The newest "occurrence" is the suffix shifted by one: the only
     * continuation token left in the context is the seed itself. */
    int ctx[] = { 7, 7, 7, 7 };
    int drafts[16];
    int n = run(ctx, 4, 4, drafts);
    CHECK(n == 1);
    CHECK(drafts[0] == 7);
}

static void test_draft_width_clamps(void) {
    /* suffix "5,1,2" at 9..11 matches at 4; the replay offers the five
     * context tokens that follow it, and max_draft caps them. */
    int ctx[] = { 1,2,3,4,5, 1,2,3,4,5, 1,2 };
    int drafts[16];
    CHECK(run(ctx, 12, 2, drafts) == 2);
    int n = run(ctx, 12, 16, drafts);
    CHECK(n == 5);
    CHECK(drafts[0] == 3 && drafts[1] == 4 && drafts[2] == 5 &&
          drafts[3] == 1 && drafts[4] == 2);
}

static void test_short_context_misses(void) {
    int ctx[] = { 1, 2, 3 };
    int drafts[16];
    CHECK(run(ctx, 3, 4, drafts) == 0);
    CHECK(run(ctx, 1, 4, drafts) == 0);
    CHECK(run(ctx, 5, 0, drafts) == 0);
}

static void test_continuation_never_leaves_context(void) {
    /* Every proposed token must be a token that exists in the context. */
    int ctx[64];
    for (int i = 0; i < 64; i++) ctx[i] = (i * 7) % 5;
    int drafts[16];
    int n = run(ctx, 64, 16, drafts);
    CHECK(n > 0);
    for (int i = 0; i < n; i++) {
        bool present = false;
        for (int j = 0; j < 64; j++) present = present || ctx[j] == drafts[i];
        CHECK(present);
    }
}

static void test_match_len_override(void) {
    /* Default n=3: the trigram "9,5,1" never occurred, so no proposal.
     * With DS4_NGRAM_SPEC_SIZE=2 the bigram "5,1" at the start matches and
     * replays "1,9" behind it. */
    int ctx[] = { 5,1,1,9,9,5,1 };
    int drafts[16];
    CHECK(run(ctx, 7, 2, drafts) == 0);
    setenv("DS4_NGRAM_SPEC_SIZE", "2", 1);
    CHECK(ds4_ngram_match_len() == 2);
    int n = run(ctx, 7, 2, drafts);
    CHECK(n == 2);
    CHECK(drafts[0] == 1 && drafts[1] == 9);
    unsetenv("DS4_NGRAM_SPEC_SIZE");
    CHECK(ds4_ngram_match_len() == 3);
}

int main(void) {
    test_unseen_suffix_misses();
    test_most_recent_occurrence_wins();
    test_repetition_replays_tail();
    test_draft_width_clamps();
    test_short_context_misses();
    test_continuation_never_leaves_context();
    test_match_len_override();
    if (failures == 0) {
        printf("ngram-propose: all checks passed\n");
        return 0;
    }
    printf("ngram-propose: %d failure(s)\n", failures);
    return 1;
}
'''


def main():
    source = SOURCE.read_text()
    parts = {}
    for name in FUNCTIONS:
        parts[name] = extract_function(source, name)
    harness = HARNESS.replace('{env_fn}', parts['ds4_ngram_env_u32'])
    harness = harness.replace('{match_fn}', parts['ds4_ngram_match_len'])
    harness = harness.replace('{propose_fn}', parts['ds4_ngram_propose'])
    with tempfile.TemporaryDirectory() as tmp:
        c_path = Path(tmp) / 'harness.c'
        bin_path = Path(tmp) / 'harness'
        c_path.write_text(harness)
        cc = os.environ.get('CC', 'cc')
        subprocess.run([cc, '-O2', '-Wall', '-Wextra', '-std=gnu99',
                        '-o', str(bin_path), str(c_path)],
                       check=True)
        result = subprocess.run([str(bin_path)])
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
