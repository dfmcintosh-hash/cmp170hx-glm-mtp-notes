# SPDX-License-Identifier: Apache-2.0
# CPU repro (Triton interpreter, no GPU) of finding B: fp8_paged_mqa_logits_triton reads native-MTP 2D seq_lens as 1D.
# Run inside a vLLM checkout of PR #38476: CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python repro_native_mtp_seqlens.py
# Exit 0 = the bug is reproduced with the 2D call AND seq_lens[:, -1] is right in every case.
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
