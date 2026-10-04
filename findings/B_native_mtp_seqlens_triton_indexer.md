<!-- Text B of our vLLM share-back. Not yet filed upstream. Upstream state as checked on 2026-10-02: vLLM main @ 6e517b15, PR #38476 head 3740c02bb1, PR #57171 head f234dd46d8, PR #46994 merged 2026-09-11. -->

# Finding B: native MTP seq_lens on the Triton sparse-MLA indexer (review comment for vLLM PR #38476)

**Native MTP (`num_speculative_tokens: 1`) on the Triton fallback: `fp8_paged_mqa_logits_triton` reads 2D `seq_lens` as 1D. One-line fix plus a CPU repro.**

Thanks for this backend; we run GLM-5.3 on sm_80 with it. We found one correctness bug in the decode path with native MTP.

### What happens (head `3740c02bb1`)
- **How the 2D tensor arises.** Off SM100, `next_n ∈ [1, 2]` takes the native path (`use_flattening = False`). With `num_speculative_tokens: 1` (`next_n = 2`), `decode_metadata.seq_lens` is 2D `(B, 2)` with `[b, j] = L_b - 2 + j + 1`.
- **Where it breaks.** `sparse_attn_indexer.py` hands that tensor straight to `fp8_paged_mqa_logits_triton`. An earlier revision used `seq_lens.squeeze(dim=1)`, which is a no-op on a size-2 dim, so it gives the same values. The kernel does `context_len = tl.load(context_lens_ptr + batch_id)`: **one length per request**, read from the flat row-major buffer `[L0-1, L0, L1-1, L1, …]`. Its docstring says `context_lens: [B]`.
- **The effects:**
  - Request 0 gets `L0 - 1`, so **every verify row loses its own key**: the row at position `p` sees keys ≤ `p - 1`.
  - Request `b ≥ 1` reads an element of another request's row. For example, request 1 reads `L0`. It is shown too few keys, or keys past its own end.
  - With `clean_logits=False`, as called, whole KV blocks at or after the wrong length are never written. The downstream top-k uses the correct 2D lengths, so it can read uninitialised logits. This last point is from source reading only.
- **Why it is easy to miss.** While the context is ≤ `index_topk` (2048), the top-k keeps every key, so nothing shows.
- **Symptom.** On a 7-stage PP GLM-5.3 serve on sm_80 (greedy, CUDA graphs), the output was coherent and then collapsed into a repetition loop once the context passed 2048 (it started at context 2138). MTP acceptance fell from 0.55–0.85 to ~0.2. K = 0 is fine, and so is `next_n ≥ 3`, which takes the flattening path with 1D lengths.

### Fix
Pass one length per request, exactly as the XPU branch on `main` already does (`seq_lens_xpu = seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens`). Against head `3740c02bb1` (applies cleanly; `ruff format` clean):

```diff
--- a/vllm/model_executor/layers/sparse_attn_indexer.py
+++ b/vllm/model_executor/layers/sparse_attn_indexer.py
@@ -343,11 +343,17 @@
             # `seq_lens`, so size the buffer to the active batch max rather
             # than the configured model max.
             active_max_model_len = attn_metadata_narrowed.max_seq_len
+            # The kernel reads one context length per request (context_lens[batch_id]);
+            # native MTP seq_lens is 2D (B, next_n) with the full length in the last
+            # column, as the XPU branch on main already handles.
+            seq_lens_1d = (
+                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
+            )
             logits = fp8_paged_mqa_logits_triton(
                 padded_q_quant_cast,
                 kv_cache,
                 weights[:num_padded_tokens],
-                seq_lens,
+                seq_lens_1d,
                 decode_metadata.block_table,
                 max_model_len=active_max_model_len,
                 clean_logits=False,
```

The 2D `seq_lens` is kept for the top-k call below, which accepts 2D. For `(B, 1)` the values are unchanged.

**With this change:**
- the same serve stays coherent past 2048 (2,600 greedy tokens, no degeneration);
- acceptance holds at 57.6–100 % across every 10-s window;
- decode is 27.8–30.5 tok/s against 23.6–23.8 tok/s for K = 0.

An alternative would be to make the kernel 2D-aware (load `context_lens[batch_id, next_n_id]`), but the one-liner matches the XPU path.

### CPU repro (Triton interpreter, no GPU)
`CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python repro_native_mtp_seqlens.py`. Exit 0 means the bug is reproduced and the fix is correct.

```python
import os, sys
assert os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
import torch
import vllm.v1.attention.ops.mqa_logits_triton as M
M._fp8_paged_mqa_logits_kernel = M._fp8_paged_mqa_logits_kernel.fn  # interpreter: skip autotune (warps/stages only)
torch.manual_seed(0)
H, D, BS, NEXT_N = 16, 32, 16, 2

def last_visible_keys(lengths, seq_lens_arg):
    B = len(lengths); nblk = max((L + BS - 1) // BS for L in lengths)
    q = torch.randn(B, NEXT_N, H, D).clamp(-4, 4).to(torch.float8_e4m3fn)
    kv = torch.zeros(B * nblk, BS, 1, D + 4, dtype=torch.uint8); flat = kv.view(B * nblk, -1)
    flat[:, :BS * D] = torch.randn(B * nblk, BS * D).clamp(-4, 4).to(torch.float8_e4m3fn).view(torch.uint8)
    flat[:, BS * D:] = torch.ones(B * nblk, BS).view(torch.uint8)  # fp32 scale 1.0
    w = torch.rand(B * NEXT_N, H); bt = torch.arange(B * nblk, dtype=torch.int32).view(B, nblk)
    logits = M.fp8_paged_mqa_logits_triton(q, kv, w, seq_lens_arg, bt, max_model_len=nblk * BS, clean_logits=True)
    out = []
    for b, L in enumerate(lengths):
        for j in range(NEXT_N):
            fin = torch.nonzero(torch.isfinite(logits[b * NEXT_N + j])).flatten()
            out.append((b, j, L - NEXT_N + j, int(fin.max()) if len(fin) else None))
    return out

def report(tag, res):
    for b, j, pos, last in res:
        print(f"  {tag:16s} req {b} row {j}: position {pos:5d}, last visible key {last}{'' if last == pos else '   <-- WRONG'}")
    return all(last == pos for _, _, pos, last in res)

bug, fix = True, True
for lengths in ([40], [2100], [40, 70], [70, 40]):
    s2d = torch.tensor([[L - NEXT_N + j + 1 for j in range(NEXT_N)] for L in lengths], dtype=torch.int32)
    print(f"lengths {lengths}: seq_lens {s2d.tolist()}")
    bug &= not report("2D (as passed)", last_visible_keys(lengths, s2d))
    fix &= report("seq_lens[:, -1]", last_visible_keys(lengths, s2d[:, -1].contiguous()))
print(f"VERDICT: 2D call wrong in every case = {bug}; seq_lens[:, -1] right in every case = {fix}")
sys.exit(0 if bug and fix else 1)
```

**Output** (rc 0). The kernel file is byte-identical to the one at `3740c02bb1`:
```
lengths [40]: seq_lens [[39, 40]]
  2D (as passed)   req 0 row 0: position    38, last visible key 37   <-- WRONG
  2D (as passed)   req 0 row 1: position    39, last visible key 38   <-- WRONG
  seq_lens[:, -1]  req 0 row 0: position    38, last visible key 38
  seq_lens[:, -1]  req 0 row 1: position    39, last visible key 39
lengths [2100]: seq_lens [[2099, 2100]]
  2D (as passed)   req 0 row 0: position  2098, last visible key 2097   <-- WRONG
  2D (as passed)   req 0 row 1: position  2099, last visible key 2098   <-- WRONG
  seq_lens[:, -1]  req 0 row 0: position  2098, last visible key 2098
  seq_lens[:, -1]  req 0 row 1: position  2099, last visible key 2099
lengths [40, 70]: seq_lens [[39, 40], [69, 70]]
  2D (as passed)   req 1 row 0: position    68, last visible key 38   <-- WRONG
  2D (as passed)   req 1 row 1: position    69, last visible key 39   <-- WRONG
  ...
lengths [70, 40]: seq_lens [[69, 70], [39, 40]]
  2D (as passed)   req 1 row 0: position    38, last visible key 68   <-- WRONG
  2D (as passed)   req 1 row 1: position    39, last visible key 69   <-- WRONG
  ...
VERDICT: 2D call wrong in every case = True; seq_lens[:, -1] right in every case = True
```

**Test suggestion:** `tests/kernels/attention/test_mqa_logits_triton.py` only passes 1D `context_lens`. A case with `B = 2, next_n = 2` and the indexer's 2D `seq_lens` through the decode wrapper would catch this.

The repro script is also in [`repro/repro_native_mtp_seqlens.py`](../repro/repro_native_mtp_seqlens.py).
