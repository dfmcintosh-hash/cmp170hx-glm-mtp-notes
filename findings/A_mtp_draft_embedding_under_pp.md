<!-- Text A of our vLLM share-back. Not yet filed upstream. Upstream state as checked on 2026-10-02: vLLM main @ 6e517b15, PR #38476 head 3740c02bb1, PR #57171 head f234dd46d8, PR #46994 merged 2026-09-11. -->

# Finding A: MTP draft embedding under pipeline parallelism (vLLM issue text)

**Title:** `[Bug][Spec Decode] MTP draft from a draft-only checkpoint runs with an uninitialised embedding under pipeline parallelism (no error; acceptance collapses)`

### Your current environment
- vLLM 0.27.1, V1 engine. Same logic on `main` @ `6e517b15` by source reading.
- 7 × NVIDIA CMP 170HX (GA100, sm_80), `--pipeline-parallel-size 7`.
- Out-of-tree sparse-MLA backend from #38476 for sm_80. Not involved in this bug.
- Model: GLM-5.3 (`glm_moe_dsa`, MTP layer index 78). A custom mixed-precision weight quantization, with the MTP head stored as a separate **draft-only** checkpoint passed via `--speculative-config '{"method":"mtp","model":"<draft dir>",...}'`. The draft dir holds only `model.layers.78.*` tensors plus `config.json`.

### 🐛 Describe the bug
With PP > 1, the MTP drafter never gets a valid input embedding when the draft checkpoint does not contain `embed_tokens`:

1. **No sharing under PP.** `LLMBaseProposer._maybe_share_embeddings` shares the target's `embed_tokens` only when `get_pp_group().world_size == 1`. Under PP it logs `The draft model's vocab embedding will be loaded separately from the target model.` (`lm_head` is still shared.)
2. **Nothing to load.** The draft checkpoint has no `model.layers.<N>.embed_tokens.weight` and no top-level `model.embed_tokens.weight`. So `DeepSeekMTP` / `DeepseekV32MTP.load_weights` never writes the drafter's `embed_tokens`; #46994 covers only the top-level tied case.
3. **No error.** `DefaultModelLoader.track_weights_loading` adds every parameter of a module whose `quant_method` has `process_weights_after_loading` to `loaded_weights`. `UnquantizedEmbeddingMethod` has that method, so the unloaded embedding passes the "weights not initialized from checkpoint" check.
4. **Result.** The draft embeds `x_{t+1}` with whatever `torch.empty` returned. Outputs stay correct, because verification rejects bad drafts, so the only symptom is low acceptance and lost speedup.

**Measured** (one model, the hardware above):

| | value |
|---|---|
| Offline draft top-1, real target embedding (5,663 held-out positions) | 0.779 |
| Same eval, embedding zeroed | 0.200 |
| Same eval, random-normal embedding | 0.194 |
| Served K=1 acceptance, PP=7, draft-only checkpoint | ~0.22 (0.13–0.31 across runs) |
| Served K=1 acceptance, PP=7, after adding the embedding to the draft dir (context < 2k) | 0.55–0.85 |
| Served K=2, same fix: tokens/step and decode vs. no speculation | 2.58 tokens/step, +49–53 % at 4k context, +41–49 % at 100k |

The zeroed-embedding ablation matches the served acceptance. The real-embedding number matches what the head achieves offline.

### Minimal repro
1. Take any DeepSeek-V3-family / GLM-5-family checkpoint with an MTP layer at index `N`. Write a draft dir containing only the `model.layers.N.*` tensors plus `config.json`.
2. Serve the target with `--pipeline-parallel-size 2` and `--speculative-config '{"method":"mtp","model":"<draft dir>","num_speculative_tokens":1}'`.
3. Observe:
   - the `loaded separately` log line;
   - no load error;
   - `SpecDecoding metrics` acceptance far below what the head reaches with a correct embedding.
4. **Workaround / control:** add the target's `model.embed_tokens.weight` to the draft dir as `model.layers.N.embed_tokens.weight`, plus an index entry. `_rewrite_spec_layer_name` maps it to the drafter's own `model.embed_tokens.weight`. Restart: acceptance recovers. This is a data-only change, with no code patch.

At PP = 1 the MTP branch always shares the target embedding ("Detected MTP model. Sharing target model embedding weights…"), so the same draft dir is fine there (source reading).

### Suggested fix (either)
- **(a) Fail loudly.** In `DeepSeekMTP` / `DeepseekV32MTP.load_weights` (or generically in the proposer under PP), raise when the drafter's `embed_tokens` was not loaded from the checkpoint and cannot be shared. Example message: "draft checkpoint has no embed_tokens; under pipeline parallelism the draft cannot use the target's. Add model.layers.N.embed_tokens.weight to the draft checkpoint."
- **(b) Load it.** Under PP, load the target checkpoint's `model.embed_tokens.weight` into the drafter on the last stage. This is what #57171 does for `glm5next`.

Also worth considering: make the "loaded separately" line a warning when the draft checkpoint has no embedding key.

Related:
- #46994 (fixed the top-level tied-embedding case);
- #57171 (same class for GLM-5.3-Flash `glm5next`).
