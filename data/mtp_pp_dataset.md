# MTP over PCIe pipeline parallelism: historical measurement dataset

Compilation date: 2026-10-02. These are historical observations, not a current performance guarantee. Source IDs identify immutable private receipts; the receipts are not yet distributed publicly. No measurements were newly run for this compilation. **NR** means not recovered or not verified.

## Reading the measurements

Native MTP K is the number of draft tokens; K0 disables speculation. DFlash is a separate draft model and its adaptive depth is not an MTP K. Aggregate completion throughput includes prefill/TTFT unless explicitly called decode. TTFT includes scheduling and queueing. Acceptance denominators differ across experiments: never compare the percentages without their definitions. Repeated cached prompts and forced output lengths are synthetic tests, not agent task success. No confidence intervals are justified by these small samples.

## 1. GLM-5.3-Flash, native MTP, PP4

### Blackwell-tail configuration

Three CMP 170HX stages and one Blackwell tail; layer split 12/11/9/13; CMP caps 225 W, Blackwell cap 275 W. Maximum context 1,048,576 tokens; maximum sequences 8; memory utilization 0.97; phase coalescing 2. Link widths/generation were not pinned in these benchmark receipts: **topology provenance incomplete**. Prompt 20 tokens, forced 512 output tokens, temperature 0, seed 0, ignore EOS, 32-token warmup, two batches per cell. September 15, 2026. The two source groups are separate runs and must not be pooled together. Acceptance = total accepted draft tokens / total drafted tokens. TTFT is the median over all requests.

| Source | Streams | Aggregate completion tok/s | Acceptance % | TTFT ms | Requests |
| --- | --- | --- | --- | --- | --- |
| F-BW-K1 | 1 | 81.63 | 99.22 | NR | 2 |
| F-BW-K1 | 4 | 181.77 | 88.22 | NR | 8 |
| F-BW-K5a | 1 | 115.44 | 85.44 | NR | 2 |
| F-BW-K5a | 4 | 176.86 | 57.60 | NR | 8 |
| F-BW-K5b | 1 | 123.06 | 94.22 | 180.01 | 2 |
| F-BW-K5b | 8 | 255.03 | 50.82 | 292.32 | 16 |

Requested matrix coverage: K1 C1/C4 and K5 C1/C4/C8 recovered; K0 all cells, K2 all cells, K1 C2/C8, and K5 C2 are NR. A matched K2-versus-K5 crossover on this Blackwell-tail configuration is **not established**.

### Separate all-CMP crossover experiment

Four CMP 170HX stages, layer split 14/11/11/9, phase coalescing off. Tail cap fixed 225 W; the other three caps are in the table. Maximum sequences 8; maximum context and link widths not independently pinned in these raw benchmark files. Same 20-token prompt, forced 512 output, temperature 0, seed 0, warmup and repetition policy as above. September 15, 2026. **This is a different hardware configuration**, not a replacement for missing Blackwell-tail cells.

| Source | K | Non-tail cap W | Streams | Aggregate completion tok/s | Acceptance % | TTFT ms | Requests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F-CMP-K2-150 | 2 | 150 | 1 | 89.99 | 100.00 | 176.18 | 2 |
| F-CMP-K2-150 | 2 | 150 | 2 | 145.06 | 93.36 | 202.16 | 4 |
| F-CMP-K2-150 | 2 | 150 | 4 | 187.63 | 82.04 | 246.91 | 8 |
| F-CMP-K2-150 | 2 | 150 | 8 | 286.38 | 78.98 | 279.37 | 16 |
| F-CMP-K2-225 | 2 | 225 | 1 | 89.79 | 100.00 | 180.24 | 2 |
| F-CMP-K2-225 | 2 | 225 | 2 | 140.32 | 90.03 | 196.73 | 4 |
| F-CMP-K2-225 | 2 | 225 | 4 | 184.96 | 77.46 | 245.61 | 8 |
| F-CMP-K2-225 | 2 | 225 | 8 | 274.59 | 79.01 | 308.70 | 16 |
| F-CMP-K5-150 | 5 | 150 | 1 | 106.82 | 100.00 | 183.59 | 2 |
| F-CMP-K5-150 | 5 | 150 | 2 | 124.56 | 71.28 | 203.77 | 4 |
| F-CMP-K5-150 | 5 | 150 | 4 | 147.67 | 51.85 | 250.57 | 8 |
| F-CMP-K5-150 | 5 | 150 | 8 | 225.98 | 50.98 | 279.35 | 16 |
| F-CMP-K5-225 | 5 | 225 | 1 | 106.49 | 100.00 | 181.28 | 2 |
| F-CMP-K5-225 | 5 | 225 | 2 | 134.30 | 76.49 | 206.70 | 4 |
| F-CMP-K5-225 | 5 | 225 | 4 | 158.99 | 58.55 | 249.83 | 8 |
| F-CMP-K5-225 | 5 | 225 | 8 | 210.06 | 55.52 | 278.41 | 16 |

In this synthetic all-CMP experiment K5 wins at one stream and K2 wins at two, four and eight streams at both cap settings. This observation does not establish the crossover for real tool workloads or for the Blackwell-tail stack.

## 2. Full GLM-5.3 mixed248, native MTP, PP7

Seven CMP 170HX stages; W8 draft and mixed248 target, layer split 17/15/14/9/9/9/5; BF16 KV, block size 64; maximum context 131,072; maximum sequences 8; maximum batched tokens 8,192; utilization 0.90; full decode graphs. Historical cap report: 200 W per CMP. Exact measured links and stage identity are not pinned by the summary receipt: **configuration provenance incomplete**. One active request; nominal contexts 4k and 100k; temperatures as shown; five repeated observations per cell, September 29, 2026. Rates are receipt medians of decode tok/s, excluding prefill.

| Source | K | Nominal context | Temperature | Decode tok/s | Joint acceptance positions 0/1 % | Repetitions |
| --- | --- | --- | --- | --- | --- | --- |
| M-PP7-02 | 0 | 4k | 0.6 | 23.75 | — | 5 |
| M-PP7-02 | 0 | 4k | 1.0 | 23.74 | — | 5 |
| M-PP7-02 | 0 | 100k | 0.6 | 23.39 | — | 5 |
| M-PP7-02 | 0 | 100k | 1.0 | 23.38 | — | 5 |
| M-PP7-02 | 2 | 4k | 0.6 | 34.79 | 85.83/64.23 | 5 |
| M-PP7-02 | 2 | 4k | 1.0 | 34.86 | 84.17/66.67 | 5 |
| M-PP7-02 | 2 | 100k | 0.6 | 36.47 | 90.00/80.18 | 5 |
| M-PP7-02 | 2 | 100k | 1.0 | 33.34 | 81.82/66.94 | 5 |

Acceptance positions are joint survival probabilities relative to drafted steps, not independent or conditional token probabilities. Medians of separate metrics need not obey identities computed from their medians.

K1 evidence is a separate diagnostic run (M-PP7-01): at one request and temperature 0, 4,096-token code/reason outputs measured 27.8/30.5 completion tok/s, versus approximately 23.8 for K0; one observation per workload. The K1 configuration used the embedding fix **and** a sequence-length indexing overlay. Its logged acceptance windows ranged 57.6–100%; these are server windows, not matched request-level rates. Context limit and links are insufficiently pinned for a controlled three-arm K0/K1/K2 comparison.

### Draft embedding defect and fix

The PP draft lacked its own target embedding tensor; PP1 sharing masked the defect. The earlier served acceptance range 13–28% (typically 22%) is a historical diagnostic summary with incomplete request/configuration provenance, **excluded from clean performance comparisons**. After-fix K2 acceptance is shown above, but the before/after workloads differ.

A CPU embedding ablation (M-EMBED-01; September 29, 2026; 5,663 positions, no GPU inference) reports real embedding top-1 agreement 77.93%, zero embedding 19.97%, and noise embedding 19.35%. These are teacher-forced top-1 agreement rates, **not served acceptance**. The real-versus-zero/noise result supports the defect diagnosis; it does not supply a matched served speedup. Early K1 logs beyond the indexing threshold and cross-launch greedy-equality failures are excluded as quality evidence.

## 3. GLM-5.3-Flash W4A16 with DFlash2 on PP4

All rates below are historical observations from October 2, 2026. Four CMP 170HX cards, 225 W caps; layer split 13/11/11/10; utilization 0.95. Version 1.5.0 and 1.6.0 refer to the Morrowmake engine. Scheduler settings in all five serve logs: block size 4,608; maximum batched tokens 2,312; prefill chunk with decodes 384; maximum partial prefills 2. KV dtype was not explicitly pinned and is NR. DFlash configuration requests three speculative tokens; version 1.6.0 adaptive depths are separately labeled. **Stock x4 is not verified for all sets**: the mixed sets include one card reported as x2.

| Source | Engine | Stage VBIOS/topology | Context limit | Max sequences | KV token capacity |
| --- | --- | --- | --- | --- | --- |
| D150-A | 1.5.0 | mixed 67/6D/67/6D; one x2 | 262144 | 8 | 2334498 |
| D150-B | 1.5.0 | all 67; reported x4, run-time links unpinned | 262144 | 8 | 2334498 |
| D150-C | 1.5.0 | 6D/67/6D/67; one x2 | 262144 | 8 | 2334498 |
| D160-7 | 1.6.0 | 6D/67/6D/67; one x2 | 524288 | 4 | 3037668 |
| D160-5 | 1.6.0 | 6D/67/6D/67; one x2 | 524288 | 4 | 3205760 |

Cross-reference to the companion x4 report: sets A/B/C correspond to D150-A / D150-C / D150-B respectively. D150-C decode receipts belong to the earlier 262,144 × 8 startup. A later restart on the same cards (D150-C-524; no decode rows in this table) used 524,288 × 4 with KV capacity 3,373,853 tokens.


6D and 67 denote VBIOS families, not measured bandwidth. Link generation and actual prompt token counts are NR where the receipt does not retain them. D160-7 uses default adaptive depth 7; D160-5 overrides adaptive depths to 5,5. **Depth 7 vs 5: confounded, re-measure pending.** Their warmup/autotune/cache state, scheduler behavior and KV capacities differ; no causal depth speedup is claimed.

### Decode

Temperature 0; one warmup per prompt kind; maximum output 400; prefix reuse; five batches per prompt kind and offered concurrency. Per-stream decode = (output tokens − 1)/(last-token time − first-token time); median over all streams. Aggregate = total output / batch wall; median over five batches. TTFT median over all streams. With max sequences 4, offered C8 includes queueing.

| Source | Prompt kind | Offered streams | Stream decode tok/s | Aggregate completion tok/s | TTFT ms | Requests |
| --- | --- | --- | --- | --- | --- | --- |
| D150-A | structured | 1 | 126.81 | 121.94 | 134.74 | 5 |
| D150-A | structured | 4 | 107.45 | 400.47 | 246.44 | 20 |
| D150-A | structured | 8 | 86.83 | 624.73 | 409.42 | 40 |
| D150-A | code | 1 | 110.12 | 104.61 | 195.63 | 5 |
| D150-A | code | 4 | 87.18 | 305.93 | 325.78 | 20 |
| D150-A | code | 8 | 63.69 | 462.41 | 430.55 | 40 |
| D150-A | prose | 1 | 84.55 | 82.30 | 141.02 | 5 |
| D150-A | prose | 4 | 70.39 | 266.18 | 260.51 | 20 |
| D150-A | prose | 8 | 52.27 | 382.61 | 318.45 | 40 |
| D150-B | structured | 1 | 117.55 | 113.26 | 137.81 | 5 |
| D150-B | structured | 4 | 96.62 | 362.22 | 242.35 | 20 |
| D150-B | structured | 8 | 83.76 | 597.78 | 315.30 | 40 |
| D150-B | code | 1 | 97.73 | 93.67 | 183.69 | 5 |
| D150-B | code | 4 | 79.63 | 286.79 | 306.50 | 20 |
| D150-B | code | 8 | 63.92 | 450.19 | 395.60 | 40 |
| D150-B | prose | 1 | 80.91 | 79.08 | 119.99 | 5 |
| D150-B | prose | 4 | 65.97 | 250.43 | 257.70 | 20 |
| D150-B | prose | 8 | 54.52 | 399.37 | 298.47 | 40 |
| D150-C | structured | 1 | 126.70 | 121.92 | 133.50 | 5 |
| D150-C | structured | 4 | 100.45 | 377.71 | 233.72 | 20 |
| D150-C | structured | 8 | 87.16 | 626.80 | 341.98 | 40 |
| D150-C | code | 1 | 105.11 | 100.62 | 179.15 | 5 |
| D150-C | code | 4 | 82.73 | 294.75 | 298.46 | 20 |
| D150-C | code | 8 | 65.46 | 472.39 | 407.93 | 40 |
| D150-C | prose | 1 | 88.92 | 86.89 | 117.41 | 5 |
| D150-C | prose | 4 | 68.84 | 260.85 | 255.66 | 20 |
| D150-C | prose | 8 | 54.03 | 399.28 | 303.78 | 40 |
| D160-7 | structured | 1 | 96.62 | 92.90 | 181.96 | 5 |
| D160-7 | structured | 4 | 93.44 | 344.03 | 251.17 | 20 |
| D160-7 | structured | 8 | 91.48 | 324.39 | 4285.00 | 40 |
| D160-7 | code | 1 | 128.38 | 121.07 | 191.84 | 5 |
| D160-7 | code | 4 | 79.10 | 285.98 | 324.50 | 20 |
| D160-7 | code | 8 | 77.25 | 283.27 | 2744.95 | 40 |
| D160-7 | prose | 1 | 82.56 | 80.26 | 170.76 | 5 |
| D160-7 | prose | 4 | 67.85 | 249.71 | 264.51 | 20 |
| D160-7 | prose | 8 | 64.79 | 243.48 | 3246.45 | 40 |
| D160-5 | structured | 1 | 162.74 | 152.19 | 176.27 | 5 |
| D160-5 | structured | 4 | 95.76 | 357.56 | 249.63 | 20 |
| D160-5 | structured | 8 | 93.07 | 348.90 | 2435.85 | 40 |
| D160-5 | code | 1 | 122.58 | 116.23 | 185.15 | 5 |
| D160-5 | code | 4 | 81.11 | 292.58 | 328.12 | 20 |
| D160-5 | code | 8 | 75.55 | 283.85 | 2720.71 | 40 |
| D160-5 | prose | 1 | 86.67 | 83.63 | 150.94 | 5 |
| D160-5 | prose | 4 | 66.90 | 254.71 | 260.93 | 20 |
| D160-5 | prose | 8 | 64.19 | 245.93 | 3198.68 | 40 |

D160-7 has substantial within-run variability and is flagged as a warmup-sensitive observation; do not present its medians as steady-state expectations. The initial failed D150-C decode receipt (HTTP 404) is excluded; the complete second receipt is used.

### Agentic-shaped four-turn fixture

One four-turn tool-enabled trial per temperature/configuration; maximum output 900 per turn, initial prompt approximately 412 tokens; thinking disabled. Values below are medians recomputed from four **rounded** text-receipt turn rates. This is not four independent agents or a task-success benchmark; prompt identity and full tool transcripts are not retained in these text receipts. TTFT is an approximately rounded fixture value.

| Source | Temperature | Median turn decode tok/s | Approx TTFT s | Trials | Turns |
| --- | --- | --- | --- | --- | --- |
| D150-A | 1.0 | 98.85 | 0.35 | 1 | 4 |
| D150-A | 0.6 | 100.75 | 0.35 | 1 | 4 |
| D150-B | 1.0 | 91.65 | 0.33 | 1 | 4 |
| D150-B | 0.6 | 93.40 | 0.33 | 1 | 4 |
| D150-C | 1.0 | 95.00 | 0.34 | 1 | 4 |
| D150-C | 0.6 | 110.55 | 0.33 | 1 | 4 |
| D160-7 | 1.0 | 118.55 | 0.35 | 1 | 4 |
| D160-7 | 0.6 | 108.35 | 0.35 | 1 | 4 |
| D160-5 | 1.0 | 104.10 | 0.35 | 1 | 4 |
| D160-5 | 0.6 | 120.65 | 0.35 | 1 | 4 |


### Prefill

Only D150-B has a recovered complete prefill JSON: one active request, fresh nonce per repetition, two repetitions per prompt, maximum output one token. Prompt tokens / TTFT is an effective first-token throughput, not an isolated kernel prefill rate. Temperature is not retained in the rows and is flagged NR. Other configuration prefill cells are NR.

| Source | Prompt ID | Measured prompt tokens | Effective first-token tok/s | TTFT s | Repetitions |
| --- | --- | --- | --- | --- | --- |
| D150-B | 0 | 25668–25670 | 4663.69 | 5.60 | 2 |
| D150-B | 1 | 26601–26602 | 5348.59 | 4.97 | 2 |
| D150-B | 2 | 30644–30645 | 5347.87 | 5.73 | 2 |
| D150-B | 3 | 36750–36751 | 5520.81 | 6.66 | 2 |


### 6D-versus-67 interpretation

The mixed-stage sets show different decode and four-turn rates from the all-67 set, but card selection, stage placement, one x2 link, context limit, scheduler capacity and startup state also vary. **No isolated VBIOS effect has been measured**, and no all-four-6D observation is recovered. A matched same-card, same-link, same-context repeat is needed before attributing a gain to VBIOS or HBM bandwidth.

## Publication limits

This dataset deliberately retains flagged diagnostic observations and shows missing cells. Public source IDs currently require the private appendix for audit; publishing sanitized raw receipts and exact public configuration manifests remains a separate step. Synthetic forced-length throughput does not measure reasoning fidelity, tool correctness, long-context quality or end-to-end agent productivity. Cross-hardware numbers and native-MTP/DFlash numbers are not interchangeable. No local endpoint, machine identity, or private path is needed to read these tables.
