# SYMBA7 — JEPA for Symbolic Regression

ML4Sci summer research project (mentor: Sergei Gleyzer, U. Alabama; collaborator: Eric Reinhardt).
Goal: improve **symbolic recovery** in transformer-based symbolic regression — recovering physics
equations (symbolic expressions) from numerical point cloud data. Target venue: NeurIPS ML4PS workshop.

## The core problem

The SymbolicGPT baseline achieves **99.6% teacher-forced token accuracy but only ~20% exact symbolic
recovery** (Malik et al., ML4PS 2025). High local syntactic fidelity, low global compositional
correctness. The decoder and training objective are the suspected bottleneck, NOT the encoder.
Every change we make should be judged against this gap.

## Plan (in order)

1. **Walkthrough (current phase).** Read and understand (a) Malik's implementation and (b) the
   working Colab notebook, before changing anything. Produce a written summary of the pipeline:
   data flow, shapes, tokenization scheme, train/eval split, and where each artifact JSON comes from.
2. **Synthetic data generation.** Add a synthetic equation + data-cloud generator to Malik's
   pipeline to expand training data beyond the Feynman set.
3. **JEPA integration.** Add a JEPA objective (see Research context below).

## Working environment — READ CAREFULLY

- All experiment work happens in **this Colab notebook**:
  https://colab.research.google.com/drive/1FEBGnqNeTg1WVdudTt4vgl5lfi0_UTNK
- The repo is **cloned into the Colab session** (ephemeral VM — re-clone on runtime restart).
- **Use the Colab MCP server (`colab-mcp`) for ALL notebook and runtime operations**: reading/adding/
  editing cells, executing code, inspecting outputs. Do not assume local bash can touch the repo —
  the working copy lives in the Colab VM, not on this machine.
- **Before any work, verify the connection target**: read the first cell (or notebook title) through
  the MCP tools and confirm it is the notebook above, not a scratch `empty.ipynb`. The MCP proxy
  binds to one browser tab; if tools are missing, the tab isn't connected — call
  `open_colab_browser_connection` and wait for the tool list to update.
- Runtime: Colab L4 GPU (Pro). Checkpoints and datasets persist to mounted Google Drive.
  GPU time is metered — confirm with David before launching any run longer than a few minutes.

## Upstream codebase (Malik / SymbolicGPT)

Repo: `https://github.com/ML4SCI/SYMBA` → `SYMBA_REG/SymbolicGPT_Krish_Malik/` (~1.5k lines):

```
data/
  Feynman_csv_edit.csv        ground-truth Feynman equations
  data_cloud.py               data cloud generation (sampling points around equations)
  data_clouds.json            pre-generated clouds
  feynman_parse_trees.json    equations as parse trees
src/
  parser/symbolic_parser.py   equation string -> parse tree (291 lines)
  embeddings/t_net_embeddings.py  T-Net point-cloud encoder (130 lines)
  decoder/decoder.py          GPT-style autoregressive decoder (176 lines)
  decoder/masking_decoder_setup.py  masked-LM training utilities
  decoder/sliding_window.py   sliding-window sparse-attention decoder — main entry point (570 lines)
  library/learned_library.py  concept library of reusable subtrees
  labels/*.json               tokenized supervision (use *_with_full_funcs.json per README)
```

Known gaps (verified by inspection): **no training driver script, no requirements.txt, no synthetic
data generator**. README's recommended run:
`python src/decoder/sliding_window.py --embeddings src/embeddings/tnet_embeddings_new.json --labels src/labels/tokenized_gpt_labels_with_full_funcs.json`

## Research context (established findings — don't re-litigate)

- **Token accuracy ≠ symbolic recovery** is the central diagnostic; report both, always.
- **Evaluation must handle algebraic equivalence**: many "wrong" predictions are equivalent forms.
  Use SymPy simplification + numerical agreement on held-out points as the equivalence check.
- **Synthetic pretraining pitfall (directly relevant to phase 2)**: pretraining on random parse
  trees consistently FAILED in prior experiments — dataset-scale mismatch and distribution gap with
  physics equations. Synthetic data must match the Feynman distribution (operator frequencies,
  tree depths, variable counts, constant ranges), not sample uniformly over grammars.
- **Set Transformer (SAB+PMA) > T-Net** for relational structure: T-Net's global max-pool is
  destructive and blocks cross-point interaction. Relevant if we touch the encoder, but remember
  the encoder is not the prime suspect.
- **JEPA framing**: predict in embedding space, not input space (vs. autoencoders) — more semantic
  representations, better for small datasets. I-JEPA's block-masking over point-cloud "patches" is
  a more natural fit than LM-JEPA's text framing. Tension to manage: JEPA masking targets global
  semantics, but symbolic regression needs multi-scale precision (local curvature distinguishes
  sin vs cos, x^2 vs x^4).
- **Multi-view JEPA (David's idea, candidate novel contribution)**: views = cross-modal (point-cloud
  tokens <-> equation tokens), augmentation (point subsamples), and algebraically equivalent
  equation forms.
- Baseline reference artifact: David's CSCI378 v4 notebook (T-Net encoder + custom transformer
  decoder trained from scratch, BFGS constant fitting, beam search, multi-metric eval).
- Key papers: SymbolicGPT (2106.14131), I-JEPA (2301.08243), LM-JEPA (2509.14252),
  Malik et al. ML4PS 2025, Set Transformer (Lee et al. 2019).

## How to work with David

- David does the substantive work himself. Claude Code's role: walkthroughs, explanations, code
  review, targeted snippets, debugging — not wholesale project completion. Propose, don't impose.
- Direct and iterative; prefers concise, actionable answers. Skip preamble.
- Prefer modular structure: core logic in `.py` files, the notebook as a thin driver.
- Never delete/overwrite checkpoints, datasets, or Drive contents without explicit confirmation.
- Flag suspected bugs precisely (file, line, failure mode) — David will often fix them himself.
