# Measurements used by the ccg_v2 card

Six ladders, one per model, in the format `contextual_drag.route_model.ladder/1`. For every kept
problem and every T = 0..4, a ladder records the drafts shown and the number of correct answers.

**Protocol.** Problems are AIME 2024, AIME 2025, HMMT 2024 and HMMT 2025. Each model answered every
problem 16 times without drafts, and a problem was kept if at least 4 answers were correct and at
least 4 wrong. For each kept problem and each T, the model answered with T correct and 4 - T wrong
drafts of its own, in 8 distinct orders (4 for GPT-OSS-20B on `hmmt24:11` at T = 0, where only 4
orders were generated), with the `dc_same_math` prompt (`prompt_templates/draft_composition_templates.json`),
reasoning on, up to 32,768 tokens, sampling seed 42 and composition seed 20260916.

| File | Eligible problems | sha256 |
|---|---|---|
| `Gemma4_31B.json` | 20 | `8410a26699280c5c…` |
| `Gemma4_E4B.json` | 22 | `7acd5f23d8b0fd4c…` |
| `GPT_OSS_120B.json` | 30 | `f21ff2f51479dcba…` |
| `GPT_OSS_20B.json` | 22 | `1383475970c351ce…` |
| `Qwen3_32B.json` | 22 | `d8ffb6d2ffa7ab9a…` |
| `Qwen3_8B.json` | 18 | `2ced2dc6a7cc031f…` |
