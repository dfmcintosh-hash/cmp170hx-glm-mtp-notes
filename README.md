# cmp170hx-glm-mtp-notes

Measurements and two bug reports from running GLM-5.3-family MoE models with **speculative decoding (native MTP and DFlash2) under pipeline parallelism over PCIe** on NVIDIA **CMP 170HX** cards.

This is a single home box, not a lab. The numbers are historical observations with their gaps marked, not a benchmark suite.

## The box
- **Cards:** eight NVIDIA CMP 170HX (GA100, **sm_80**, Ampere), with 64 GB visible per card.
  - Runs used 4 (PP4) or 7 (PP7) of them.
  - One GLM-5.3-Flash configuration used a single Blackwell-generation card as the last pipeline stage. It is labeled as such wherever it appears.
- **Links:** mixed PCIe. Most cards sit on Gen2 x4 links and one card is at Gen2 x2. There is no NVLink and peer-to-peer is off. Where a run's link widths were not recorded at run time, the tables say so.
- **Layout:** **pipeline parallelism over PCIe** (PP4 / PP7), because tensor parallelism is a poor fit for narrow links.
- **Power:** CMPs capped at 150–225 W and the Blackwell tail at 275 W, as stated per table.
- **Firmware:** two vBIOS families ("67" and "6D") are present; they differ in memory clock.

## What's inside

| path | what | licence |
|---|---|---|
| [`data/mtp_pp_dataset.md`](data/mtp_pp_dataset.md) | The dataset with every definition, caveat and missing cell (**NR**): GLM-5.3-Flash native MTP K1/K2/K5 on PP4; full GLM-5.3 (our mixed-precision quantization) native MTP K0/K1/K2 on PP7; GLM-5.3-Flash W4A16 + DFlash2 on PP4 (Morrowmake engine 1.5.0 / 1.6.0) | CC BY 4.0 |
| [`data/csv/`](data/csv/) | The same tables as CSV, one file per table, extracted mechanically from the dataset | CC BY 4.0 |
| [`findings/A_mtp_draft_embedding_under_pp.md`](findings/A_mtp_draft_embedding_under_pp.md) | vLLM: an MTP draft loaded from a **draft-only checkpoint** runs with an **uninitialised embedding** under PP > 1. There is no error; served K=1 acceptance was ~0.22, against 0.55–0.85 with the fix | Apache-2.0 |
| [`findings/B_native_mtp_seqlens_triton_indexer.md`](findings/B_native_mtp_seqlens_triton_indexer.md) | vLLM PR #38476 (Triton sparse-MLA for sm_80): native MTP passes **2D `seq_lens`** to a kernel that reads them as 1D. One-line fix | Apache-2.0 |
| [`repro/repro_native_mtp_seqlens.py`](repro/repro_native_mtp_seqlens.py) | CPU-only repro of finding B (Triton interpreter, no GPU) | Apache-2.0 |

**Headline observations:**
- **GLM-5.3-Flash, all-CMP PP4, synthetic forced-length run:** native MTP K5 wins at one stream, and K2 wins at 2, 4 and 8 streams, at both 150 W and 225 W.
- **Full GLM-5.3 on PP7:**
  - K2 decodes at about 33–36 tok/s against about 23.4–23.8 tok/s for K0, once the draft-embedding bug (finding A) is fixed.
  - K1 reaches 27.8–30.5 tok/s with both fixes applied (one observation per workload).

Read the dataset's definitions before comparing any two numbers.

## Honest limits
- **No Hopper or Blackwell reference baselines.** Everything is sm_80 except the one labeled Blackwell-tail configuration. Nothing here says how these models behave on datacenter GPUs.
- **The 1.6.0 DFlash2 depth-7 vs depth-5 comparison is confounded:**
  - Another request overlapped the depth-7 cells, and the engine was compiling kernels on first use.
  - The depth-5 run had acceptance-aware depth switched off.
  - So no causal depth effect is claimed, and a clean re-measure is pending.
- **6D vs 67 vBIOS:** no isolated effect has been measured. Card selection, pipeline stage, link width and context limit vary together in these runs.
- **Small samples:** typically 1–5 runs per cell, with no confidence intervals.
  - Several runs are synthetic: a 20-token prompt with a forced 512-token output, or repeated prompts.
  - Synthetic throughput says nothing about reasoning fidelity, tool correctness or agent productivity.
- **Raw receipts are not published yet.** Source IDs in the tables refer to private raw logs. Publishing sanitized receipts is a possible later step.
- **Findings A and B:**
  - "Not fixed on main" was established by reading the source at the stated commit, not by running main.
  - Each was observed on one model, one quantization, sm_80 and one PP depth.
- **Upstream may have moved** since 2026-10-02. Each finding records the commits it was checked against.

## Not included
- **No model weights, drafter or container images.** In particular, the DFlash2 drafter ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), CC BY-NC-ND 4.0, by incoai) is not redistributed here in any form. Only our measurements made with it, unmodified, are included.
- **Engine and recipe:** [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) (Apache-2.0) and [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) (MIT). Our PP4-on-x4 measurement report for them is posted separately: **MORROWMAKE-ISSUE-LINK-PLACEHOLDER**.

## Credits
- [vLLM](https://github.com/vllm-project/vllm).
- Morrowmake, for the CMP 170HX engine and recipe.
- incoai, for the DFlash2 drafter.
- canada-quant, for the GLM-5.3-Flash W4A16 + MTP checkpoint.
- The GLM-5.3 model authors.

## Licence
- Text and code (README, `findings/`, `repro/`): **Apache-2.0**, see [`LICENSE`](LICENSE).
- Data (`data/`, including the CSVs): **CC BY 4.0**, see [`LICENSE-DATA`](LICENSE-DATA). Please attribute "cmp170hx-glm-mtp-notes".
