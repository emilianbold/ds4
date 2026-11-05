#!/usr/bin/env python3
"""Convert a DS4_MOE_RECORD_SELECTED_HOTLIST recording into a hotlist .inc table.

The streaming expert cache can be seeded at session create from a routing
hotlist: a table of (layer, expert) pairs ordered by observed selection
count, which the loader converts into rank-based priorities
(metal_graph_streaming_builtin_hotness in ds4.c). V4.1 and GLM ship
built-in tables generated with this tool.

Recording the routing (the env vars are read by ds4, not by this tool):

    DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt ./ds4-bench ... > /dev/null

Every routed-layer call appends to an in-memory count table (one line per
distinct (layer, expert) pair, six selections per call); the file is written
when the process exits. Add DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 to
accumulate across runs: each run first loads the previous file, so one
hotlist can keep growing over representative workloads
(DS4_MOE_RECORD_SELECTED_HOTLIST_FRESH=1 inverts this for a clean start).
Record with the same engine options as the intended deployment; the corpus
determines the table. For example, merged bench and one-shot runs:

    DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt \
    DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 \
        ./ds4-bench -m MODEL --ssd-streaming \
            --prompt-file speed-bench/promessi_sposi.txt \
            --ctx-start 1024 --ctx-max 1024 --gen-tokens 128

    DS4_MOE_RECORD_SELECTED_HOTLIST=hotlist.txt \
    DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1 \
        ./ds4 -m MODEL --ssd-streaming --ctx 2048 \
            --think-level 0 --temp 0 -n 48 -p "A representative prompt."

The exact corpus recorded for the shipped ds4_streaming_hotlist_v41.inc
(two bench sizes and three one-shot prompts, which reproduce that table
on the same GGUF) is documented in docs/SSD_STREAMING.md.

Then convert:

    python3 gguf-tools/hotlist_to_inc.py hotlist.txt ds4_streaming_hotlist_v41.inc

The output keeps the recorder's order (hits descending, then layer and
expert ascending), which is what the loader's priority derives from; rows
are re-sorted here so hand-merged files also work. See
docs/SSD_STREAMING.md for how the table is wired into a model.

Only Python's standard library is used.
"""

import sys

MAX_LAYER = 80    # DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER
MAX_EXPERT = 512  # DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT


def fail(msg):
    sys.stderr.write("hotlist_to_inc: %s\n" % msg)
    sys.exit(1)


def parse(path):
    """Read the recorder's text format: '# ...' headers then
    'layer expert hits weight' rows."""
    rows = []
    selections = 0
    records = None
    try:
        with open(path) as fp:
            for line in fp:
                fields = line.split()
                if not fields:
                    continue
                if fields[0].startswith("#"):
                    continue
                if len(fields) != 4:
                    fail("expected 'layer expert hits weight' in %s, got: %s"
                         % (path, line.strip()))
                layer, expert, hits = int(fields[0]), int(fields[1]), int(fields[2])
                if not (0 <= layer < MAX_LAYER):
                    fail("layer %d out of range 0..%d in %s" % (layer, MAX_LAYER - 1, path))
                if not (0 <= expert < MAX_EXPERT):
                    fail("expert %d out of range 0..%d in %s" % (expert, MAX_EXPERT - 1, path))
                if hits <= 0:
                    fail("non-positive hits for layer %d expert %d in %s" % (layer, expert, path))
                rows.append((layer, expert, hits))
                selections += hits
    except ValueError:
        fail("non-numeric field in %s" % path)
    except OSError as e:
        fail("cannot read %s: %s" % (path, e))
    return rows, selections


def main():
    if len(sys.argv) not in (3, 4):
        sys.stderr.write(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) == 4 else "ds4_default_streaming_hotlist_v41"

    rows, selections = parse(src)
    if not rows:
        fail("no entries in %s" % src)
    rows.sort(key=lambda r: (-r[2], r[0], r[1]))  # recorder order: hits, layer, expert

    with open(dst, "w") as f:
        f.write("/* Generated from routing recorded with\n"
                " * DS4_MOE_RECORD_SELECTED_HOTLIST (%d selections); every\n"
                " * expert recorded selected at least once, %d (layer, expert)\n"
                " * pairs, hits-descending so the loader's rank-based priority\n"
                " * matches observed frequency. Regenerate: see\n"
                " * gguf-tools/hotlist_to_inc.py and docs/SSD_STREAMING.md. */\n"
                % (selections, len(rows)))
        f.write("static const uint16_t %s[][2] = {\n" % name)
        for i in range(0, len(rows), 6):
            f.write("    " + " ".join("{%d,%d}," % (r[0], r[1]) for r in rows[i:i+6]) + "\n")
        f.write("};\n\n"
                "static const uint32_t %s_count =\n"
                "    (uint32_t)(sizeof(%s) /\n"
                "               sizeof(%s[0]));\n"
                % (name, name, name))
    layers = len(set(r[0] for r in rows))
    print("%s: %d pairs, %d layers, %d selections" % (dst, len(rows), layers, selections))


if __name__ == "__main__":
    main()
