## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.

Numeric runtime tuning controls can use `--candidate-value TEXT` (default `1`), and
`--prefill-chunk N` selects the same chunk size for both variants (default
4096). The control unsets the named variable. Use only controls read at
dispatch time: this harness keeps one engine alive and cannot compare settings
cached during initialization.

`DS4_QWEN4_TIMING=2` reports each prefill chunk's position, token count, and
stage times, including the attention mixer separately from GDN/attention.
These measurements include synchronization overhead and are for attribution;
use uninstrumented alternating runs for throughput claims.
`DS4_QWEN4_MOE_PROFILE=1` further splits tiled MoE into routing, expert-list
construction, routed/shared gate-up, routed/shared down, and reduction,
reporting one line per layer. Its synchronization also makes it diagnostic only.

Qwen routed MoE quantization specialization is enabled by default on M3 Ultra.
`DS4_QWEN4_MOE_MM_SPECIALIZE=0` restores the generic kernels; `=1` opts in on
other devices. To compare the final default against rollback, pass
`--candidate-env DS4_QWEN4_MOE_MM_SPECIALIZE --candidate-value 0` to the harness.

### Concurrency: many requests at once

Serving several requests at once is a different regime from the single-session
walk `ds4-bench` measures. Two harnesses cover it, one per layer.

`session_concurrency_bench` drives the engine API directly, so the numbers
carry no HTTP, tokenizer or scheduler noise: it is the ceiling the server can
aim at, and the signal to optimize batched decode against.

```
make session-concurrency-bench
./speed-bench/session_concurrency_bench \
  -m ds4flash.gguf \
  --concurrency 1,2,4,8,16 \
  --ctx 0,4096,16384,32768,65536,131072 \
  --gen 32 \
  --csv speed-bench/qwen38_concurrency.csv
```

Each cell creates one session per stream, prefills them at staggered corpus
offsets (identical prompts would give every stream the same MoE routing and
flatter any batched expert path), then advances all of them one token per step
through `ds4_sessions_eval_batch`. It reports aggregate decode throughput,
per-stream throughput, step time percentiles, the aggregate prefill rate and
the last stream's time-to-first-token, which is what a queued user waits.

Memory, not time, bounds the wide/long corner of the grid. The harness prints
the per-cell estimate from `ds4_context_memory_estimate_with_prefill`, then
attempts every cell and records the ones whose sessions do not fit as `oom`
instead of refusing them from a guessed budget. The reported footprint is the
process physical footprint, which counts session graphs and caches but not the
file-backed weight mapping.

`--mixed` adds the pattern that decides whether one long prompt stalls every
other user: the decoders keep running while a separate session resumes a
chunked prefill through `ds4_sessions_eval_batch_with_prefill`. `--verify`
checks that batched decode selects the same tokens as one-at-a-time decode;
native grouping may reorder floating point reductions, so the contract is token
agreement, not bit-identical logits.

`serve_concurrency_bench.py` measures the same load through the HTTP server, so
queueing, prompt rendering, prefix-cache decisions and streaming all count:

```
./ds4-server --ctx 32768 --batched-session 8 &
python3 speed-bench/serve_concurrency_bench.py \
  --concurrency 8 --prompt-tokens 4096 --max-tokens 128 --requests 32 \
  --ignore-eos --json /tmp/serve-8.json
```

It reports TTFT, inter-token latency, TPOT and end-to-end latency with
percentiles, alongside request and token throughput. `--concurrency N` keeps N
clients in a closed loop, which is the shape the engine harness mirrors;
`--request-rate R` switches to Poisson arrivals to find where latency degrades.
Prompts carry a fresh nonce so the server cannot answer from a cached prefix;
`--shared-prefix` measures the cached path on purpose.

Walk the whole grid with one process per cell:

```
CONCURRENCY="1 2 4 8 16" CONTEXTS="0 16384 65536 131072" \
  speed-bench/concurrency_sweep.sh speed-bench/qwen38_concurrency.csv --gen 32
```

One process cannot walk the grid honestly. Prefill fills the unified buffer
cache with n-gram pages — each 320-byte row read pulls a 16 KiB page from a
table accessed by hashing, so there is no locality to amortize it — while the
resident weight mapping cannot be evicted to make room. The system then
compresses session memory instead of dropping cache, and the machine swaps
even though the sessions themselves fit. A fresh process per cell gives every
cell the same budget and the same starting conditions.

#### Comparing two variants

Absolute throughput drifts by five to fifteen percent between runs on a machine
this size: each process pays a different price to make tens of GiB of weights
resident against whatever the last run left cached, and the GPU's thermal state
moves over a long sweep. Comparing two processes therefore measures the machine,
not the change. Alternate the variants inside one engine instead:

```
./speed-bench/session_concurrency_bench --ctx 4096 --concurrency 16 \
  --candidate-env DS4_QWEN4_SESSION_BATCH --candidate-value 0 --repeat 3
```

The control unsets the variable and the candidate sets it, on the same prefilled
sessions, in ABBA blocks so drift over a pair cancels instead of favouring
whichever ran first. The reported ratio holds to about two percent across runs
while the absolute figures still move by five; read the ratio, not the numbers
either side of it.

#### Sustained load moves the absolute numbers

Back-to-back cells decay: four consecutive C=16 runs measured 110.1, 109.5,
103.4 and 93.4 tok/s aggregate with free memory and swap identical before each
one, so the cause is sustained load on the GPU rather than memory state. Most
of the heating comes from the prefill wave, not from the measured decode: a
sixteen-stream 4096-token wave keeps the GPU saturated for roughly a minute
before the first timed step.

Within one cell the step time is steady to a few percent, so a single
measurement is internally consistent. It is the drift between cells, over
minutes, that makes absolute figures from different sessions incomparable.
Read the paired ratio for comparisons, and when an absolute number is wanted,
take the first cell after the machine has been idle and say so.
