# Official Quality Fixtures

This directory contains curated hosted-model continuation fixtures that are
safe to commit and use in release QA.

- `glm52-openrouter-100`: 100 GLM 5.2 OpenRouter continuations with API
  top-logprob slices.
- `glm53-flash-openrouter-zai-fp8-100`: 100 deterministic GLM 5.3 Flash
  continuations from OpenRouter's pinned Z.AI FP8 endpoint. That endpoint did
  not return logprobs.
- `flash`: 100 DeepSeek V4 Flash 0731 continuations from the official DeepSeek
  API, with API top-logprob slices.
- `pro`: 100 DeepSeek V4 PRO preview continuations with API top-logprob slices.
- `pro-0813`: 100 DeepSeek V4 PRO 0813 continuations with API top-logprob
  slices.
- `flash-0731-openrouter-morph-bf16-temp1-100`: the same 100 prompts as
  `flash`, sent to `deepseek/deepseek-v4-flash-0731` through OpenRouter pinned
  to the Morph bf16 endpoint at temperature 1 with top-20 logprobs. Sampled,
  not greedy: the official DeepSeek endpoint collapses logprobs to `0`/`-9999`
  at temperature 0, so this set is the one with usable next-token
  distributions for `api_overlap` and the other logprob-agreement columns.
  Judge it by NLL and distribution agreement, not greedy prefix length.

Each fixture directory contains:

- `prompts/case_*.txt`: exact user prompts.
- `continuations/case_*.txt`: deterministic hosted-model continuations.
- `responses/case_*.json`: raw hosted responses, including logprob slices.
- `manifest.tsv`: paths consumed by `score_official`.

DeepSeek V4 Flash smoke vectors are also tracked in `tests/test-vectors/` and
are run by `./ds4_test --logprob-vectors`.
