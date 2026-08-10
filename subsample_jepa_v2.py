# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Subsample JEPA — on the corrected v2 pipeline
#
# Rerun of the simple subsample-invariance pretraining experiment using the
# fixed training/evaluation code, so the result is comparable to
# `jepa_pretrain_v2` (`jepa_pretrain_v2_vocab40_matched`) rather than to the
# older `subsample_jepa.ipynb` sweep.
#
# ```text
# Stage 1  subsample-invariance pretraining   (pretrain_epochs)
#    |     E(S_A) ~= sg(E(S_B))  for two subsamples of ONE function's cloud
#    |     trains the T-Net encoder only
#    v
# Stage 2  CE-only fine-tuning                (FINETUNE_EPOCHS)
#    |
#    v
#        greedy decode + symbolic evaluation
# ```
#
# ## The objective
#
# For two random subsamples \(S_A, S_B\) drawn from the same function's point
# cloud, ask only that the encoder map them to the same latent:
#
# \[ E(S_A) \approx \operatorname{sg}(E(S_B)) \]
#
# No symbolic target, no predictor MLP, no `[PRED]` tokens. This is deliberately
# the simplest subsample representation-alignment objective there is: the sole
# claim being tested is that invariance to *which points you happen to observe*
# is a useful prior for the encoder.
#
# ## The comparison
#
# | run | Stage 1 |
# |---|---|
# | `pretrain_0` | skipped — plain CE training, the baseline |
# | `pretrain_10` | 10 epochs of subsample-invariance pretraining |
#
# Paired over seeds `42, 123, 7`.
#
# ## What is shared with the corrected baseline
#
# Everything downstream of Stage 1 is unchanged and deliberately so — splits
# (same `SYNTH_SEED`, same `group_by_tokens`, so the partition is identical),
# Stage-2 CE fine-tuning, checkpoint provenance, greedy decoding, and scoring.
# `pretrain_0` here and `pretrain_0` in the baseline differ only in which
# Stage-1 objective was skipped, so the two experiments are directly comparable.
#
# ## What differs from the old `subsample_jepa.ipynb`
#
# | Area | old sweep | here |
# |---|---|---|
# | Framing | one-stage, CE + λ·subsample jointly | two-stage, pretrain then fine-tune |
# | Swept variable | λ ∈ {0 … 0.3} | pretrain_epochs ∈ {0, 10} |
# | Gradient | both views carry gradient | `sg` on the target view |
# | Raw expressions | 10 000, no dedup | 200 000, one per distinct token sequence |
# | Splits | random | grouped by token sequence (no verbatim leakage) |
# | Vocabulary | 27 tokens (`x**3`, `x**4` → `C`) | 32 tokens, every exponent structural |
# | Checkpoints | written straight to Drive | atomic local write, verified, then Drive backup |
# | Val metrics | token accuracy | + branching-position accuracy |
#
# Diagnostics (zero/shuffled-input ablations, worker-RNG check, checkpoint
# fault injection, tiny-set overfit) live in
# `jepa_pretrain_diagnostics_v2.ipynb`, not here.

# %% [markdown]
# ## 0. Colab / environment setup
#
# The repository is checked out on Drive (that is also where `data/synthetic.pkl`
# lives — it is 194 MB and not tracked by Git). Checkpoints are **not** written
# there: see section 6.
#
# If library code changes while this runtime is live, restart the kernel before
# relying on the new imports.

# %%
import os
import sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

# Edit these two if your Drive layout differs.
DRIVE_BASE = Path('/content/drive/MyDrive/Symba')
REPO_URL = 'https://github.com/zzpDavid2/symbolic-jepa.git'


def sh(cmd: str) -> int:
    """Run a shell command and echo it. Returns the exit code."""
    print(f'$ {cmd}')
    code = os.system(cmd)
    if code != 0:
        print(f'  -> exit {code}')
    return code


DRIVE_MOUNTED = False
if IN_COLAB:
    from google.colab import drive

    try:
        drive.mount('/content/drive')
        DRIVE_MOUNTED = True
    except Exception as e:
        # Drive is a durable-backup convenience, not a training requirement.
        print(f'[backup WARNING] Drive mount failed: {type(e).__name__}: {e}')
        print('Training can still run; checkpoints stay on /content only.')

    REPO_DIR = DRIVE_BASE / 'symbolic-jepa'
    if not REPO_DIR.exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        sh(f'git clone {REPO_URL} "{REPO_DIR}"')
    else:
        sh(f'git -C "{REPO_DIR}" pull --ff-only')

    sh(f'{sys.executable} -m pip install -q sympy scipy')
else:
    REPO_DIR = Path.cwd()

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

import symbolic_jepa  # noqa: E402

print(f'\nEnvironment : {"Colab" if IN_COLAB else "local"}')
print(f'Repo        : {REPO_DIR}')
print(f'symbolic_jepa imported from: {symbolic_jepa.__file__}')
sh(f'git -C "{REPO_DIR}" rev-parse --short HEAD')

# %% [markdown]
# ## 1. Imports and global config
#
# Every experiment-defining constant lives in this one cell. `make_run_config`
# (section 6) serialises them into each checkpoint, so a checkpoint written
# under different settings cannot be reused.

# %%
import datetime
import gc
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from symbolic_jepa import (
    PrefixTokenizer,
    TNet, SymbolicTransformer,
    subsample_consistency_loss,
    build_multiview_synthetic_splits,
    teacher_forced_counts,
    evaluate_predictions,
    view_consistency_diagnostics, generate_diagnostic_embeddings,
)
from symbolic_jepa.datacache import (
    DataCache, cached_synthetic_expressions, cached_prefix_tree,
)
from symbolic_jepa.checkpointing import (
    CheckpointError,
    validate_checkpoint, save_checkpoint_atomic, save_json_atomic,
    load_local_or_restore,
    backup_checkpoint_to_drive, check_best_latest_consistency,
    write_manifest, sync_local_runs_to_drive,
)
from symbolic_jepa.tokenizer import prefix_to_sympy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f'Device: {DEVICE}')

# %%
# ── Experiment identity ────────────────────────────────────────────────
# Subsample-invariance pretraining on the corrected v2 pipeline.
#
# Stage 1 replaces the equation-token JEPA objective with the simplest
# subsample-alignment objective: for two random subsamples S_A, S_B of the SAME
# function's point cloud,
#
#     E(S_A)  ~=  sg( E(S_B) )
#
# No predictor MLP, no symbolic target, no [PRED] tokens. The only thing being
# asked of the encoder is invariance to WHICH points of a function it sees.
#
# Everything downstream — splits, Stage-2 CE fine-tuning, checkpointing,
# decoding, evaluation — is byte-identical to jepa_pretrain_v2_vocab40_matched,
# so `pretrain_0` here and `pretrain_0` there differ only by which Stage-1
# objective was skipped.
EXPERIMENT_VERSION = 'subsample_jepa_v2'

# Bump when DECODING or SCORING changes, independently of training. Trained
# checkpoints stay valid; only metrics.json is invalidated, so affected runs
# are re-scored rather than retrained. Shared with the baseline: identical
# decoder and identical scorer.
EVAL_VERSION = 'v3_expanded_numerics'

# ── Model / optimisation (unchanged from v1) ───────────────────────────
MAX_VARS = 1                  # univariate synthetic data
D_INPUT = MAX_VARS + 1        # (x, y) = 2
N_POINTS = 1000
MAX_SEQ = 64
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.2

LR = 3e-4
WEIGHT_DECAY = 0.1
BATCH = 16
VAL_EVERY = 1
USE_AMP = True

# ── The comparison ─────────────────────────────────────────────────────
# v1 swept {0, 5, 10, 15}; v2 establishes the corrected baseline for the
# 0-vs-10 comparison, paired by seed. Add values back here to widen the sweep.
FINETUNE_EPOCHS = 30
PRETRAIN_EPOCHS_VALUES = [0, 10]
SEEDS = [42, 123, 7]
GLOBAL_SEED = SEEDS[0]        # notebook-level RNG; per-run seeds come from SEEDS

# ── Subsample objective used in Stage 1 ────────────────────────────────
# 'centered' subtracts each view's batch mean before the cosine. Use it. The
# T-Net max-pools ReLU features into the positive orthant, so RAW cosine
# between any two embeddings starts ~0.99 and the raw loss starts ~5e-4 — too
# small to produce gradient. Worse, the raw objective is *minimised* by total
# collapse (all-equal embeddings drive it to 0), whereas the centered one
# *penalises* collapse (drives it to 1).
SUBSAMPLE_LOSS = 'centered'   # 'centered' (recommended) | 'cosine'
# The mode we do NOT optimise, logged alongside for reference: free to compute
# and it shows how the two scales move relative to each other.
SUBSAMPLE_LOSS_OTHER = 'cosine' if SUBSAMPLE_LOSS == 'centered' else 'centered'
PRETRAIN_STOPGRAD = True      # E(S_A) ~= sg(E(S_B)); see the note in section 7

N_VIEWS = 2                   # subsamples per equation per step
TRAIN_VIEW_SEED = 1729        # governs training views, independent of model seed
POOL_MULT = 4                 # draw N_POINTS*POOL_MULT, then subsample views
VIEW_POINTS = None            # points per view (None = N_POINTS)
# Stage 2 offsets its view epochs so fine-tuning never replays the exact clouds
# Stage 1 pretrained on — otherwise pretrain_10 would meet 10 familiar epochs.
FINETUNE_VIEW_EPOCH_OFFSET = 1000

# ── Fixed view-consistency diagnostic ──────────────────────────────────
# Same clouds for every checkpoint and every model, so reported differences
# reflect the encoder rather than the data.
EVAL_SEED = 2718
DIAG_N_EXPRS = 256            # val equations used for the diagnostic
DIAG_N_VIEWS = 4              # views per equation

# ── Data ───────────────────────────────────────────────────────────────
SYNTH_PKL = str(REPO_DIR / 'data' / 'synthetic.pkl')
SYNTH_SEED = 42
MAX_SYNTH = 200_000           # raw strings read (mentor request)
DEDUPE_BY_TOKENS = True       # keep one example per distinct token sequence
USE_DATA_CACHE = True         # reuse the parsed expressions / branch tree
GROUP_BY_TOKENS = True        # no token sequence spans two splits

# ── DataLoader ─────────────────────────────────────────────────────────
# Workers fork from identical NumPy/Python RNG state and would emit identical
# augmented clouds; `seed_worker` (section 2) fixes that. Set to 0 to move
# sampling back into the main process — that also makes Stage 2's augmentation
# stream reset per epoch, so pre0 and pre10 see bit-identical clouds (see the
# trajectory note in section 2).
NUM_WORKERS = 2

# ── Evaluation ─────────────────────────────────────────────────────────
MAX_EVAL_EXAMPLES = 0         # 0 = full test set; a positive N caps greedy decode

# ── Drive backup cadence ───────────────────────────────────────────────
DRIVE_BACKUP_EVERY = 5        # epochs between latest.pt backups

# %% [markdown]
# ### Why `PRETRAIN_STOPGRAD` stays `True`, and why `centered` matters
#
# The objective is stated as \(E(S_A) \approx \operatorname{sg}(E(S_B))\), so
# view 0 carries the gradient and the remaining views are fixed targets. Unlike
# the token-JEPA target in the baseline, `detach()` alone is sufficient here:
# both sides come from the same T-Net, which has no dropout on this path, so
# there is no risk of the "fixed" target moving between the two encodes. The
# baseline's `model.eval()` dance is therefore not needed.
#
# Stage 1 touches **only the T-Net encoder**. The loss is a function of encoder
# outputs alone, so the decoder never receives a gradient — this measures
# *"does subsample-invariance pretraining of the point-cloud encoder help?"*,
# not *"does pretraining the whole model help?"*
#
# **Collapse is the thing to watch.** The trivial solution to "make two views
# agree" is to map everything to one constant vector. Two guards:
#
# * `SUBSAMPLE_LOSS = 'centered'` subtracts each view's batch mean before the
#   cosine. Under total collapse every centered vector is 0, cosine evaluates
#   to 0, and the loss goes to **1** — the centered objective *penalises*
#   collapse. The raw objective is *minimised* by it, which makes raw actively
#   dangerous here in a way it was not in the joint-loss sweep.
# * `pretrain/std_z_num` is logged every epoch. It falling toward 0 is the
#   collapse signature; `val/gap_centered` going to 0 while `same_fn_cos`
#   stays high is the same story seen from the validation side.
#
# The mentor commit rewrote `encode_expression` to prepend a learned
# `null_data_token`. That path is unused here — this objective never encodes
# expressions — but the library implementation is called directly wherever it
# does appear.

# %% [markdown]
# ## 2. Reproducibility / worker seeding
#
# Three separate mechanisms, deliberately:
#
# * `seed_everything(seed)` — full re-seed of torch/numpy/python.
# * `seed_worker` — the mentor's fix. Without it, forked DataLoader workers
#   inherit identical NumPy/Python state and emit identical point clouds. It is
#   epoch-independent: each worker is seeded once from `torch.initial_seed()`.
# * `stage1_epoch_seed` / `stage2_epoch_seed` — the per-epoch reseed of the
#   *main-process* RNG and of the sampler's `torch.Generator`. These depend only
#   on `(seed, epoch)`, never on how much RNG an earlier stage consumed, so
#   epoch *N* is identical whether the run was uninterrupted or resumed straight
#   into epoch *N*, and every `pretrain_epochs` condition at a given seed shares
#   one Stage-2 trajectory.
#
# **Caveat with `NUM_WORKERS > 0`:** worker processes are persistent and are
# seeded once, so their augmentation stream is *not* reset per epoch. A
# `pretrain_epochs=10` run therefore consumes worker-side draws during Stage 1
# and enters Stage 2 on different clouds than the `0` baseline. Shuffle order,
# dropout, and initialisation remain matched. Set `NUM_WORKERS = 0` if you want
# the clouds matched too.

# %%
def seed_everything(seed: int) -> None:
    """Full re-seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own NumPy/Python stream."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def stage1_epoch_seed(seed: int, epoch: int) -> int:
    """RNG seed for one subsample-pretraining epoch."""
    return seed + 200_000 + epoch


def stage2_epoch_seed(seed: int, epoch: int) -> int:
    """RNG seed for one CE fine-tuning epoch."""
    return seed + 100_000 + epoch


def make_train_loader(dataset, seed: int):
    """Training loader: shuffled, multi-worker, explicitly seeded.

    Returns `(loader, generator)`. Re-seed the generator with
    `g.manual_seed(epoch_seed)` before each epoch — the sampler re-reads it on
    every `__iter__`, so the shuffle order stays a pure function of
    `(seed, epoch)` even with persistent workers.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        # MUST stay False. MultiViewPointCloudDataset reads `self.epoch`
        # inside __getitem__, and workers hold a FORKED COPY of the dataset —
        # persistent workers would never observe the epoch bump and every
        # epoch's views would silently freeze at epoch 0.
        persistent_workers=False,
        pin_memory=(DEVICE == 'cuda'),
        worker_init_fn=seed_worker,
        generator=g,
    )
    return loader, g


def make_eval_loader(dataset):
    """Deterministic loader for validation / test: no shuffle, no workers.

    `evaluate_predictions` indexes `dataset.samples[i]` positionally, so the
    order must be the dataset's own.
    """
    return DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)


# Representation diagnostic: same-function vs different-function cosine on a
# FIXED set of views, not on whichever validation batch arrived first.
#
# `generate_diagnostic_embeddings` seeds every cloud from
# (EVAL_SEED, 0, expr_idx, view_idx), so the SAME clouds are encoded for every
# epoch, every pretraining condition and every model seed — a change in the
# reported numbers is a change in the encoder, never in the data. This is the
# direct read on the Stage-1 objective: subsample invariance should raise
# same_fn_cos without dragging diff_fn_cos up with it.
#
# Read the *_centered numbers. The T-Net max-pools ReLU activations into the
# positive orthant, so raw cosine between ANY two embeddings sits near 0.98
# regardless of content; centering strips that shared offset. `gap_centered`
# (same - diff) is the headline: it is high only when the encoder is both
# subsample-invariant AND still discriminating between functions, so it cannot
# be gamed by collapse the way same_fn_cos alone can.


def view_consistency(model, val_ds, n_exprs: int = DIAG_N_EXPRS,
                     n_views: int = DIAG_N_VIEWS):
    """same/diff-function cosine on the fixed diagnostic views."""
    z = generate_diagnostic_embeddings(
        val_ds, model.encoder, n_exprs=n_exprs, n_views=n_views,
        eval_seed=EVAL_SEED, device=DEVICE, batch_size=BATCH,
    )
    return view_consistency_diagnostics(z, n_views)


seed_everything(GLOBAL_SEED)
print(f'Global seed: {GLOBAL_SEED} | DataLoader workers: {NUM_WORKERS}')

# %% [markdown]
# ## 3. Tokenizer and data loading
#
# `max_expressions=200_000` caps the **raw strings read** from the pickle;
# `dedupe_by_tokens=True` then keeps one expression per distinct prefix token
# sequence. The kept count is therefore much smaller than 200 000 and is *not*
# comparable to v1's 10 000 — v1 had no deduplication, so an unknown fraction of
# it was repeated sequences.
#
# **The first run is slow (~10-15 min, with a progress bar); later runs are
# not.** The cost is SymPy — every raw string is `sympify`d and converted to
# prefix notation at ~3 ms each, and deduplication happens *after* parsing, so
# it saves none of that. The parsed result is therefore cached: a fresh runtime
# restores ~27k expressions in ~90 s instead of re-parsing 200 000 strings, and
# the diagnostics notebook reuses the same entry.
#
# The expressions cannot be cached as prefix strings — `prefix_to_sympy` maps
# every numeric literal back to a *fittable* `c_i` symbol, so the original
# coefficients would be lost and the point clouds would be wrong. The SymPy
# objects themselves are pickled instead.
#
# Set `USE_DATA_CACHE = False` to force a rebuild. Lowering `MAX_SYNTH` for a
# smoke test is safe: it keys a separate entry and cannot collide.

# %%
# Derived-artifact cache. Parsing the pickle is the slow step and its result
# depends only on the settings above, so it is cached content-addressed: the
# filename embeds a hash of the source pickle, every argument that changes
# which expressions survive, the tokenizer vocabulary, and the source of the
# parsing modules. A changed config or a changed parser misses and rebuilds
# under a new name — a stale entry cannot be read.
if IN_COLAB:
    LOCAL_CACHE_ROOT = Path('/content/symbolic_jepa_cache')
    DRIVE_CACHE_ROOT = (DRIVE_BASE / 'symbolic_jepa_cache') if DRIVE_MOUNTED else None
else:
    LOCAL_CACHE_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_cache'
    DRIVE_CACHE_ROOT = None

CACHE = DataCache(LOCAL_CACHE_ROOT, DRIVE_CACHE_ROOT, enabled=USE_DATA_CACHE)
print(f'data cache local : {LOCAL_CACHE_ROOT}')
print(f'data cache drive : {DRIVE_CACHE_ROOT}')

# %%
tokenizer = PrefixTokenizer(max_vars=MAX_VARS)
print(f'Vocab size: {len(tokenizer)}')

print(f'Loading synthetic expressions from {SYNTH_PKL} ...')
synth_exprs = cached_synthetic_expressions(
    SYNTH_PKL, tokenizer,
    max_seq_len=MAX_SEQ,
    max_vars=MAX_VARS,
    max_expressions=MAX_SYNTH,
    dedupe_by_tokens=DEDUPE_BY_TOKENS,
    cache=CACHE,
    progress=True,
)

for expr in synth_exprs[:5]:
    print(f'  {expr.prefix}')

# %% [markdown]
# ## 4. Corrected dataset splits and branching prefix tree
#
# `group_by_tokens=True` keeps every expression sharing a prefix token sequence
# inside one split, so no held-out sequence appears verbatim in train. The
# library prints its own leakage diagnostic — that is the authoritative one and
# is not duplicated here.
#
# `BRANCH_TREE` is built **once**, from the finalised training split only. It
# maps each token prefix to the number of distinct continuations seen in
# training; `teacher_forced_counts(..., branch_tree=BRANCH_TREE)` then scores
# only the positions where more than one next token was ever possible.
# Ordinary teacher-forced accuracy is dominated by deterministic symbolic
# syntax, so it can sit near 100% while the model gets every genuine decision
# wrong. Both are reported, always.
#
# The tree is cached too, keyed on a hash of the training token sequences
# themselves — so the cached tree is provably the one those sequences produce,
# and training and diagnostics cannot drift apart. It is only a few seconds to
# build; the cache is there for that guarantee, not for the time.
#
# Not cached: the point-cloud probe inside `build_synthetic_splits`, which
# samples every expression once (~2 ms each, ~1 min total). It needs the live
# `Expression` objects anyway, so caching it would save little.

# %%
# Train is multi-view; val/test stay deterministic single-view and cached.
# Same split logic, same SYNTH_SEED and same group_by_tokens as
# build_synthetic_splits, so the partition matches the baseline run exactly
# and the two experiments are directly comparable.
synth_train, synth_val, synth_test = build_multiview_synthetic_splits(
    synth_exprs, tokenizer,
    n_points=N_POINTS, max_seq_len=MAX_SEQ, max_vars=MAX_VARS,
    seed=SYNTH_SEED,
    n_views=N_VIEWS, train_view_seed=TRAIN_VIEW_SEED,
    pool_mult=POOL_MULT, view_points=VIEW_POINTS,
    group_by_tokens=GROUP_BY_TOKENS,
    progress=True,
)

BRANCH_TREE = cached_prefix_tree(synth_train.token_keys, cache=CACHE,
                                 progress=True)

_train_keys = set(synth_train.token_keys)
print(f'\nDataset summary ({EXPERIMENT_VERSION})')
print(f'  raw expression strings considered : {MAX_SYNTH}')
print(f'  expressions kept after dedup      : {len(synth_exprs)}')
print(f'  train / val / test                : '
      f'{len(synth_train)} / {len(synth_val)} / {len(synth_test)}')
print(f'  unique token sequences (train)    : '
      f'{synth_train.unique_sequence_count()}')
print(f'  unique token sequences (val/test) : '
      f'{synth_val.unique_sequence_count()} / '
      f'{synth_test.unique_sequence_count()}')
for _name, _ds in (('val', synth_val), ('test', synth_test)):
    _dup = sum(1 for k in _ds.token_keys if k in _train_keys)
    print(f'  leakage {_name:<4} (also in train)      : {_dup}/{len(_ds)}')
print(f'  branch tree prefixes              : {len(BRANCH_TREE)}')
print(f'  branching prefixes (>1 next tok)  : '
      f'{sum(1 for v in BRANCH_TREE.values() if v > 1)}')

# %% [markdown]
# ## 5. Model construction

# %%
def build_model(dropout: float = DROPOUT):
    """Fresh encoder + decoder on DEVICE.

    No predictor: the subsample objective aligns two encoder outputs directly,
    so there is nothing to predict *with*. The return signature keeps a third
    slot for symmetry with the baseline driver, always None.
    """
    encoder = TNet(d_input=D_INPUT, d_model=D_MODEL)
    model = SymbolicTransformer(
        encoder=encoder, vocab_size=len(tokenizer),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=4 * D_MODEL, max_seq_len=MAX_SEQ,
        dropout=dropout, pad_id=tokenizer.pad_id,
    ).to(DEVICE)

    return model, encoder, None


def make_amp_ctx():
    if USE_AMP and DEVICE == 'cuda':
        return lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    if USE_AMP and DEVICE == 'mps':
        return lambda: torch.autocast('mps', dtype=torch.float16)
    return lambda: torch.amp.autocast('cpu', enabled=False)


amp_ctx = make_amp_ctx()

# %% [markdown]
# ## 6. Robust checkpoint helpers and paths
#
# ```text
# training
#    -> local Colab filesystem (/content/...)      <- authoritative
#    -> atomic save: temp file, fsync, load back, validate, os.replace
#    -> best-effort backup to Google Drive          <- durable only
# ```
#
# **A Drive error after a successful local save is a warning, never a stop.**
# Local problems print `[checkpoint ERROR]`; Drive problems print
# `[backup WARNING]`. Nothing is ever written straight to `latest.pt`,
# `best.pt`, `metrics.json`, or the manifest — every write goes temp → verify →
# `os.replace()`. In-flight files carry a `.tmp-` / `.uploading-` /
# `.restore-` marker and are never treated as resumable.
#
# ### Surviving a lost runtime
#
# ```text
# Normal live training : /content v2 checkpoint = authoritative
# Runtime destroyed    : /content disappears
# New runtime          : restore a validated v2 Drive backup -> local /content,
#                        then train from the local copy again
# During training      : periodically copy already-validated local
#                        checkpoints -> Drive
# ```
#
# Resume order is strictly `local -> (Drive restored into local) -> fresh`.
# Training never reads from a Drive path. A corrupt or foreign local file is
# reported loudly and renamed aside (never deleted) before Drive is consulted.
#
# ### Per-run layout
#
# ```text
# jepa_pretrain_v2/
#     pretrain_0/  seed_42/  pretrain/    latest.pt
#                            supervised/  latest.pt  best.pt
#                                         metrics.json  manifest.json
#                                         history.json
#     pretrain_10/ seed_42/  ...
# ```
#
# The `pretrain/` and `supervised/` namespaces are separate *and* their
# `run_config` carries a `stage` field, so a pretraining checkpoint can never be
# mistaken for a supervised one. `run_config` also carries `pretrain_epochs`,
# and every checkpoint carries `seed`, so a `pretrain_epochs=0` run cannot
# restore a `10` run and one seed cannot restore another.

# %%
if IN_COLAB:
    LOCAL_CHECKPOINT_ROOT = Path('/content/symbolic_jepa_checkpoints') / EXPERIMENT_VERSION
    LOCAL_LOG_ROOT = Path('/content/symbolic_jepa_runs') / EXPERIMENT_VERSION
    DRIVE_CHECKPOINT_ROOT = (
        (DRIVE_BASE / 'symbolic_jepa_checkpoints') / EXPERIMENT_VERSION
        if DRIVE_MOUNTED else None
    )
else:
    LOCAL_CHECKPOINT_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_checkpoints' / EXPERIMENT_VERSION
    LOCAL_LOG_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_runs' / EXPERIMENT_VERSION
    DRIVE_CHECKPOINT_ROOT = None

LOCAL_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_LOG_ROOT.mkdir(parents=True, exist_ok=True)

# Files inside one run directory. Nothing outside this map is ever written.
RUN_FILES = {
    'pre_latest': 'pretrain/latest.pt',
    'sup_latest': 'supervised/latest.pt',
    'sup_best': 'supervised/best.pt',
    'metrics': 'supervised/metrics.json',
    'manifest': 'supervised/manifest.json',
    'history': 'supervised/history.json',
}


def run_paths(pretrain_epochs: int, seed: int) -> dict:
    """THE single source of truth for v2 run paths.

    Every checkpoint, metric, manifest and TensorBoard directory derives from
    this — never from a hand-built f-string.
    """
    rel = Path(f'pretrain_{pretrain_epochs}') / f'seed_{seed}'
    paths = {'rel': rel, 'tb': LOCAL_LOG_ROOT / rel,
             'local_dir': LOCAL_CHECKPOINT_ROOT / rel}
    for key, sub in RUN_FILES.items():
        paths[key] = LOCAL_CHECKPOINT_ROOT / rel / sub
        paths[f'drive_{key}'] = (
            None if DRIVE_CHECKPOINT_ROOT is None
            else DRIVE_CHECKPOINT_ROOT / rel / sub
        )
    return paths


print(f'EXPERIMENT_VERSION    : {EXPERIMENT_VERSION}')
print(f'LOCAL_CHECKPOINT_ROOT : {LOCAL_CHECKPOINT_ROOT}   (authoritative)')
print(f'DRIVE_CHECKPOINT_ROOT : {DRIVE_CHECKPOINT_ROOT}   (backup only)')
print(f'LOCAL_LOG_ROOT        : {LOCAL_LOG_ROOT}')
print(f'DRIVE_BACKUP_EVERY    : {DRIVE_BACKUP_EVERY} epochs')
if DRIVE_CHECKPOINT_ROOT is None:
    print('\n[backup WARNING] No Drive root configured — checkpoints are '
          'LOCAL ONLY and will be lost if this runtime dies.')

# %%
def make_run_config(pretrain_epochs: int, stage: str) -> dict:
    """Every setting that materially changes what a run means.

    Stored in each checkpoint and in metrics.json, and compared before anything
    cached is reused. `experiment_version` and `stage` are what make v1
    checkpoints and cross-stage mix-ups structurally impossible.

    `vocab_size`/`vocab_hash` make a tokenizer change structurally impossible
    to miss even if EXPERIMENT_VERSION is not bumped: nothing else here is
    derived from the tokenizer, so a vocabulary edit would otherwise leave
    every field identical and let a stale metrics.json be returned verbatim.
    The hash rather than the size alone catches reorderings and same-count
    token swaps.
    """
    assert stage in ('pretrain', 'supervised')
    return {
        'experiment_version': EXPERIMENT_VERSION,
        'stage': stage,
        'pretrain_epochs': pretrain_epochs,
        'finetune_epochs': FINETUNE_EPOCHS,
        'pretrain_stopgrad': PRETRAIN_STOPGRAD,
        'subsample_loss': SUBSAMPLE_LOSS,
        'n_views': N_VIEWS,
        'train_view_seed': TRAIN_VIEW_SEED,
        'pool_mult': POOL_MULT,
        'view_points': VIEW_POINTS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'batch': BATCH,
        'dropout': DROPOUT,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'max_seq': MAX_SEQ,
        'n_points': N_POINTS,
        'max_vars': MAX_VARS,
        'vocab_size': len(tokenizer),
        'vocab_hash': hashlib.sha1(
            '\x00'.join(tokenizer.vocab).encode()).hexdigest()[:12],
        'synth_seed': SYNTH_SEED,
        'max_synth': MAX_SYNTH,
        'dedupe_by_tokens': DEDUPE_BY_TOKENS,
        'group_by_tokens': GROUP_BY_TOKENS,
    }


# Fields the resume path actually reads back, beyond the structural minimum.
LATEST_EXTRA = ('optimizer', 'scheduler', 'history', 'best_val', 'best_epoch')
PRETRAIN_EXTRA = ('optimizer',)


def make_validator(pretrain_epochs, seed, kind, stage, path=None):
    """A one-argument validator for save/load helpers."""
    cfg = make_run_config(pretrain_epochs, stage)
    extra = {'latest': LATEST_EXTRA, 'pretrain': PRETRAIN_EXTRA}.get(kind, ())
    max_epoch = pretrain_epochs if stage == 'pretrain' else FINETUNE_EPOCHS

    def _validate(ck):
        validate_checkpoint(
            ck, expected_seed=seed, expected_run_config=cfg, kind=kind,
            extra_required=extra, max_epoch=max_epoch, path=path,
        )

    return _validate


def backup_and_record(local_path, drive_path, manifest_path, verbose=True) -> bool:
    """Best-effort Drive backup, recorded in the manifest. Never raises."""
    if drive_path is None:
        write_manifest(manifest_path, {
            'last_drive_backup_status': 'skipped (no Drive root)',
        })
        return False
    ok = backup_checkpoint_to_drive(local_path, drive_path, verbose=verbose)
    write_manifest(manifest_path, {
        'last_drive_backup_time': datetime.datetime.now().isoformat(timespec='seconds'),
        'last_drive_backup_status': 'ok' if ok else 'failed',
        'last_drive_backup_file': Path(local_path).name,
    })
    return ok


# %% [markdown]
# ## 7. JEPA pretraining stage (Stage 1)
#
# Gradient flow is JEPA → encoder only (with `PRETRAIN_STOPGRAD=True`). There is
# deliberately **no** decoder forward pass and **no** CE term: the loss is the
# bare JEPA objective, and there is no lambda — the swept variable is
# `pretrain_epochs`. `pretrain_epochs == 0` skips this stage entirely and is the
# control.
#
# Stage 1 is checkpointed under its own `pretrain/` namespace so a lost runtime
# does not throw away ten epochs of encoder pretraining. It is skipped when a
# valid supervised checkpoint already exists — Stage 1's weights are inside it.

# %%
def run_pretraining(model, predictor, params, pretrain_epochs, seed,
                    paths, writer, train_loader, train_gen, history,
                    train_ds=None):
    """Stage 1: subsample-invariance pretraining. Resumable, local-first.

    Objective, for two random subsamples S_A, S_B of one function's cloud:

        E(S_A)  ~=  sg( E(S_B) )

    Only the encoder is trained — the decoder never sees a gradient here,
    because the loss is a function of encoder outputs alone.
    """
    if pretrain_epochs == 0:
        return

    # Stage 1 gets its OWN optimizer; Stage 2 builds a fresh one so no Adam
    # moments or LR schedule carry across the objective switch.
    pre_opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    validate = make_validator(pretrain_epochs, seed, 'pretrain', 'pretrain',
                              path=paths['pre_latest'])

    start_epoch = 1
    ck, source = load_local_or_restore(
        paths['pre_latest'], paths['drive_pre_latest'], validate=validate)
    if ck is not None:
        model.load_state_dict(ck['model'])
        pre_opt.load_state_dict(ck['optimizer'])
        for key, values in ck.get('history', {}).items():
            history[key] = list(values)
        start_epoch = ck['epoch'] + 1
        if start_epoch > pretrain_epochs:
            print(f'  Stage 1 already complete ({ck["epoch"]} epochs, '
                  f'source={source})')
            return
        print(f'  Stage 1 resuming at epoch {start_epoch} (source={source})')

    print(f'\n=== Stage 1: subsample-invariance pretraining '
          f'({pretrain_epochs} epochs, loss = subsample only) ===')
    print(f'  E(S_A) ~= {"sg(E(S_B))" if PRETRAIN_STOPGRAD else "E(S_B)"} | '
          f'mode={SUBSAMPLE_LOSS} | n_views={N_VIEWS} | '
          f'view_seed={TRAIN_VIEW_SEED} | trains the encoder only')

    for pe in range(start_epoch, pretrain_epochs + 1):
        # Resume-invariant: depends only on (seed, pretrain epoch).
        es = stage1_epoch_seed(seed, pe)
        seed_everything(es)
        train_gen.manual_seed(es)

        # Fresh, reproducible views for this epoch. Read inside __getitem__,
        # so it must be set before the loader is iterated (workers fork here).
        if train_ds is not None:
            train_ds.epoch = pe

        model.train()
        pre_sum = 0.0
        pre_n = 0
        std_num_sum = 0.0
        other_sum = 0.0

        pbar = tqdm(train_loader, leave=False,
                    desc=f'{paths["rel"]} P{pe}/{pretrain_epochs}')
        for batch in pbar:
            # (batch, n_views, view_points, d) — subsamples of ONE pool per
            # equation, so the views are genuinely partial observations of the
            # same function rather than independent full draws.
            views = batch['points_views'].to(DEVICE, non_blocking=True)

            pre_opt.zero_grad()
            with amp_ctx():
                z_views = [model.encoder(views[:, v]) for v in range(N_VIEWS)]

                if PRETRAIN_STOPGRAD:
                    # E(S_A) ~= sg(E(S_B)): view 0 carries the gradient, the
                    # rest are fixed targets. Detaching is enough here — unlike
                    # the token-JEPA target there is no dropout on this path to
                    # make the "fixed" side move between the two encodes.
                    z_in = [z_views[0]] + [z.detach() for z in z_views[1:]]
                else:
                    z_in = z_views

                loss_pre = subsample_consistency_loss(z_in, mode=SUBSAMPLE_LOSS)

                with torch.no_grad():
                    loss_other = subsample_consistency_loss(
                        [z.detach() for z in z_views],
                        mode=SUBSAMPLE_LOSS_OTHER)

            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pre_opt.step()

            pre_sum += loss_pre.item()
            other_sum += loss_other.item()
            pre_n += 1
            # Averaged over the epoch, not read off the last batch: a per-batch
            # reading jitters with which equations land last and would mask a
            # real collapse trend. std -> 0 is the collapse signature.
            std_num_sum += z_views[0].detach().float().std(dim=0).mean().item()
            pbar.set_postfix({'sub': f'{loss_pre.item():.4f}'})
        pbar.close()
        del pbar

        pre_avg = pre_sum / max(pre_n, 1)
        other_avg = other_sum / max(pre_n, 1)
        std_num = std_num_sum / max(pre_n, 1)
        history.setdefault('pretrain_subsample', []).append(pre_avg)
        history.setdefault('pretrain_subsample_other', []).append(other_avg)
        history.setdefault('pretrain_std_num', []).append(std_num)
        writer.add_scalar('pretrain/subsample_loss', pre_avg, pe)
        writer.add_scalar('pretrain/subsample_loss_other', other_avg, pe)
        writer.add_scalar('pretrain/std_z_num', std_num, pe)
        print(f'  P{pe}/{pretrain_epochs} | subsample={pre_avg:.4f} '
              f'({SUBSAMPLE_LOSS_OTHER}={other_avg:.4f}) | '
              f'std(z_num)={std_num:.4f}')

        pre_ck = {
            'model': model.state_dict(),
            'optimizer': pre_opt.state_dict(),
            'epoch': pe,
            'seed': seed,
            'run_config': make_run_config(pretrain_epochs, 'pretrain'),
            'history': {k: v for k, v in history.items()
                        if k.startswith('pretrain_')},
        }
        save_checkpoint_atomic(pre_ck, paths['pre_latest'], validate=validate,
                               label='pretrain/latest.pt')
        write_manifest(paths['manifest'], {
            'experiment_version': EXPERIMENT_VERSION,
            'seed': seed,
            'pretrain_epochs': pretrain_epochs,
            'pretrain_latest_epoch': pe,
            'run_config': make_run_config(pretrain_epochs, 'supervised'),
        })
        if pe % DRIVE_BACKUP_EVERY == 0 or pe == pretrain_epochs:
            backup_and_record(paths['pre_latest'], paths['drive_pre_latest'],
                              paths['manifest'])

    del pre_opt


# %% [markdown]
# ## 8. Encoder transfer
#
# There is no weight copying step: Stage 1 and Stage 2 share one `model`
# object, so the pretrained encoder *is* the Stage-2 initialisation. What is
# deliberately dropped at the boundary is the optimizer and LR schedule —
# a fresh `AdamW` + `CosineAnnealingLR` is built for Stage 2 so no Adam moments
# carry across the objective switch.
#
# ## 9. Supervised training stage (Stage 2) + 10. Validation metrics
#
# Validation aggregates over **tokens**, not batches: the loss is weighted by
# `out['n_tokens']` and accuracy accumulates raw `(correct, total)` counts from
# `teacher_forced_counts`. Averaging per-batch would over-weight short batches.
# The same counts are accumulated a second time with `branch_tree=BRANCH_TREE`
# to give branching-position accuracy. These aggregates — not any per-batch
# mean — are what feed best-model selection, the checkpoints, the history,
# TensorBoard, and `metrics.json`.

# %%
def validate_epoch(model, predictor, val_loader, val_ds=None):
    """One full validation pass with token-weighted aggregation.

    Returns loss, ordinary token accuracy, branching-position accuracy, their
    denominators, and the encoder view-consistency diagnostics.

    Loss and accuracy always use the FULL val_loader.  The
    `z_sym` / `z_pred` embeddings behind the representation diagnostics come
    from the FIXED diagnostic views of *val_ds* (see `view_consistency`),
    not from whichever validation batch arrived first.  Passing None skips
    them.
    """
    model.eval()

    val_loss_sum = 0.0
    val_tokens_total = 0
    acc_correct = acc_total = 0.0
    branch_correct = branch_total = 0.0
    n_batches = 0

    with torch.no_grad(), amp_ctx():
        for batch in val_loader:
            points = batch['points'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attn_mask = batch['attn_mask'].to(DEVICE, non_blocking=True)
            out = model(points, input_ids, attn_mask=attn_mask)

            n_tok = int(out['n_tokens'])
            val_loss_sum += float(out['loss'].item()) * n_tok
            val_tokens_total += n_tok

            c, t = teacher_forced_counts(
                out['logits'], input_ids, tokenizer.pad_id)
            acc_correct += c
            acc_total += t

            cb, tb = teacher_forced_counts(
                out['logits'], input_ids, tokenizer.pad_id,
                branch_tree=BRANCH_TREE)
            branch_correct += cb
            branch_total += tb

            n_batches += 1

    vc = view_consistency(model, val_ds) if val_ds is not None else {}

    return {
        'val': val_loss_sum / max(val_tokens_total, 1),
        'val_acc': acc_correct / max(acc_total, 1),
        'val_branch_acc': branch_correct / max(branch_total, 1),
        'n_tokens': val_tokens_total,
        'n_branch_tokens': int(branch_total),
        'view_consistency': vc,
    }


# %%
def train_one(pretrain_epochs, seed, synth_train, synth_val):
    """Train one (pretrain_epochs, seed) run. No eval — training + checkpoints."""
    paths = run_paths(pretrain_epochs, seed)
    run_tag = str(paths['rel'])
    sup_cfg = make_run_config(pretrain_epochs, 'supervised')

    # Completion marker: metrics.json is a few KB, so an interrupted sync can
    # leave the ~200 MB latest.pt stale while metrics.json survives. Its
    # run_config is still checked, so a real config change is not skipped.
    if paths['metrics'].exists():
        try:
            with open(paths['metrics']) as f:
                cached = json.load(f)
        except Exception as e:
            print(f'\n{run_tag}: metrics.json unreadable ({type(e).__name__}); '
                  f'falling back to checkpoint state')
            cached = None
        if cached is not None:
            if cached.get('run_config') != sup_cfg:
                raise CheckpointError(
                    f'metrics.json run_config does not match this run\n'
                    f'  {paths["metrics"]}')
            print(f'\n{run_tag}: already trained and evaluated, SKIPPING')
            return

    paths['local_dir'].joinpath('supervised').mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    print(f'\n{"=" * 70}')
    print(f'{run_tag} | pretrain={pretrain_epochs}ep | '
          f'finetune={FINETUNE_EPOCHS}ep | subsample={SUBSAMPLE_LOSS} | '
          f'stopgrad={PRETRAIN_STOPGRAD} | {EXPERIMENT_VERSION}')
    print(f'{"=" * 70}')

    model, encoder, predictor = build_model()
    params = list(model.parameters())

    train_loader, train_gen = make_train_loader(synth_train, seed)
    val_loader = make_eval_loader(synth_val)

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS)

    latest_validate = make_validator(pretrain_epochs, seed, 'latest',
                                     'supervised', path=paths['sup_latest'])
    best_validate = make_validator(pretrain_epochs, seed, 'best',
                                   'supervised', path=paths['sup_best'])

    start_epoch = 1
    best_val = float('inf')
    best_val_acc = 0.0
    best_val_branch_acc = 0.0
    best_view_consistency = {}
    best_epoch = 0
    history = {'train': [], 'val': []}

    # local -> Drive-restored-into-local -> fresh. Never trains from Drive.
    sup_ck, sup_source = load_local_or_restore(
        paths['sup_latest'], paths['drive_sup_latest'], validate=latest_validate)

    if sup_ck is not None:
        model.load_state_dict(sup_ck['model'])
        optimizer.load_state_dict(sup_ck['optimizer'])
        scheduler.load_state_dict(sup_ck['scheduler'])
        start_epoch = sup_ck['epoch'] + 1
        best_val = sup_ck['best_val']
        best_val_acc = sup_ck.get('best_val_acc', 0.0)
        best_val_branch_acc = sup_ck.get('best_val_branch_acc', 0.0)
        best_epoch = sup_ck['best_epoch']
        history = sup_ck['history']

        # Bring best.pt back alongside latest.pt (it is restored from Drive
        # too, if the local tree was lost). An old best.pt paired with a newer
        # unrelated latest.pt would be silently overwritten on the first
        # improvement — refuse instead.
        best_ck, _ = load_local_or_restore(
            paths['sup_best'], paths['drive_sup_best'],
            validate=best_validate, verbose=False)
        if best_ck is not None:
            check_best_latest_consistency(best_ck, sup_ck,
                                          path=paths['local_dir'])
            del best_ck

        print(f'  Stage 2 resuming at epoch {start_epoch} '
              f'(best val {best_val:.4f}, source={sup_source})')
        if start_epoch > FINETUNE_EPOCHS:
            print(f'{run_tag}: training complete, SKIPPING')
            writer = None
        else:
            writer = SummaryWriter(log_dir=str(paths['tb']))
    else:
        print(f'  Starting fresh ({EXPERIMENT_VERSION}; no v1 fallback exists)')
        writer = SummaryWriter(log_dir=str(paths['tb']))

    if writer is None:
        del model, encoder, optimizer, scheduler
        del train_loader, val_loader
        gc.collect()
        return

    # ── Stage 1 ────────────────────────────────────────────────────────
    # Skipped on resume: the pretrained weights are already inside sup_ck.
    if sup_ck is None:
        run_pretraining(model, predictor, params, pretrain_epochs, seed,
                        paths, writer, train_loader, train_gen, history,
                        train_ds=synth_train)
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FINETUNE_EPOCHS)
    del sup_ck

    # ── Stage 2 ────────────────────────────────────────────────────────
    # Rebuild the training loader so every pretraining condition ENTERS Stage 2
    # with the same worker RNG state.
    #
    # `persistent_workers=True` keeps one set of workers alive for the whole
    # run and `seed_worker` only fires at worker *creation*, so the workers'
    # NumPy streams advance continuously. `train_gen.manual_seed(es)` pins the
    # shuffle order but not those streams — point clouds are drawn worker-side
    # from the global NumPy RNG (`resample=True` passes `rng=None`). A
    # 10-epoch Stage 1 therefore leaves its workers 10 epochs advanced, and
    # `pretrain_10` would meet Stage 2 on different clouds than `pretrain_0` —
    # confounding exactly the comparison this experiment exists to make.
    #
    # Done unconditionally, not just after Stage 1, so the loader is created at
    # the same point with the same seed in both arms. Nothing else about the
    # training setup changes.
    del train_loader, train_gen
    gc.collect()                      # joins the old workers before respawning
    train_loader, train_gen = make_train_loader(synth_train, seed)

    print(f'\n=== Stage 2: CE fine-tuning ({FINETUNE_EPOCHS} epochs, '
          f'loss = CE only, epoch seed = {seed} + 100000 + epoch) ===')
    print(f'    train loader rebuilt for Stage 2 '
          f'(workers={NUM_WORKERS}, seed={seed}) so both arms start matched')

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        es = stage2_epoch_seed(seed, epoch)
        seed_everything(es)
        train_gen.manual_seed(es)
        # Stage 2 consumes view 0 of the multi-view dataset. Its clouds are a
        # pure function of (TRAIN_VIEW_SEED, epoch, idx, 0) — no worker RNG is
        # involved — so both pretraining arms see BIT-IDENTICAL Stage-2 data
        # by construction, a stronger guarantee than the loader rebuild alone.
        synth_train.epoch = FINETUNE_VIEW_EPOCH_OFFSET + epoch

        model.train()
        train_loss_gen = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, leave=False,
                    desc=f'{run_tag} E{epoch}/{FINETUNE_EPOCHS}')
        for batch in pbar:
            points = batch['points'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attn_mask = batch['attn_mask'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with amp_ctx():
                out = model(points, input_ids, attn_mask=attn_mask)
                # Stage 2 is CE only. The JEPA term is not gated to zero, it is
                # simply absent — there is no lambda in this experiment.
                loss = out['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_gen += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('train/loss_step', loss.item(),
                              (epoch - 1) * len(train_loader) + n_batches)
        pbar.close()
        del pbar

        scheduler.step()
        train_avg = train_loss_gen / max(n_batches, 1)
        history['train'].append(train_avg)
        writer.add_scalar('train/loss_epoch', train_avg, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        is_best = False
        if epoch % VAL_EVERY == 0 or epoch == FINETUNE_EPOCHS:
            v = validate_epoch(model, predictor, val_loader, synth_val)
            history['val'].append(v['val'])
            history.setdefault('val_acc', []).append(v['val_acc'])
            history.setdefault('val_branch_acc', []).append(v['val_branch_acc'])

            writer.add_scalar('val/loss', v['val'], epoch)
            writer.add_scalar('val/token_accuracy', v['val_acc'], epoch)
            writer.add_scalar('val/token_accuracy_branching',
                              v['val_branch_acc'], epoch)

            vc = v['view_consistency']
            for _k, _val in vc.items():
                writer.add_scalar(f'val/{_k}', _val, epoch)
                history.setdefault(f'val_{_k}', []).append(_val)

            is_best = v['val'] < best_val
            if is_best:
                best_val = v['val']
                best_val_acc = v['val_acc']
                best_val_branch_acc = v['val_branch_acc']
                best_view_consistency = vc
                best_epoch = epoch

            print(f'  E{epoch}/{FINETUNE_EPOCHS} | train={train_avg:.4f} | '
                  f'val={v["val"]:.4f}{" * best" if is_best else ""} | '
                  f'acc={v["val_acc"] * 100:.1f}% | '
                  f'branch_acc={v["val_branch_acc"] * 100:.1f}% '
                  f'(n={v["n_branch_tokens"]}) | '
                  f'gap_c={vc.get("gap_centered", float("nan")):.4f}')

            if is_best:
                # best.pt is the authoritative best-model weight snapshot and
                # carries enough provenance to be validated on its own.
                # best_val_acc means "val_acc AT the epoch of minimum val
                # loss", not "maximum val_acc".
                save_checkpoint_atomic({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'val': best_val,
                    'val_acc': best_val_acc,
                    'val_branch_acc': best_val_branch_acc,
                    'seed': seed,
                    'run_config': sup_cfg,
                }, paths['sup_best'], validate=best_validate,
                    label='supervised/best.pt')

        # latest.pt / history remain authoritative for training metadata.
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val': best_val,
            'best_val_acc': best_val_acc,
            'best_val_branch_acc': best_val_branch_acc,
            'best_view_consistency': best_view_consistency,
            'best_epoch': best_epoch,
            'history': history,
            'seed': seed,
            'run_config': sup_cfg,
        }
        save_checkpoint_atomic(ckpt, paths['sup_latest'],
                               validate=latest_validate,
                               label='supervised/latest.pt')
        save_json_atomic(history, paths['history'])
        write_manifest(paths['manifest'], {
            'experiment_version': EXPERIMENT_VERSION,
            'seed': seed,
            'pretrain_epochs': pretrain_epochs,
            'latest_epoch': epoch,
            'best_epoch': best_epoch,
            # None until the first validation, so the manifest stays strict JSON.
            'best_val': best_val if np.isfinite(best_val) else None,
            'best_val_acc': best_val_acc,
            'best_val_branch_acc': best_val_branch_acc,
            'best_view_consistency': best_view_consistency,
            'run_config': sup_cfg,
        })

        # ── Drive backup: best-effort, after the local save is verified ──
        if is_best:
            backup_and_record(paths['sup_best'], paths['drive_sup_best'],
                              paths['manifest'])
        if epoch % DRIVE_BACKUP_EVERY == 0 or epoch == FINETUNE_EPOCHS:
            backup_and_record(paths['sup_latest'], paths['drive_sup_latest'],
                              paths['manifest'])
            backup_and_record(paths['history'], paths['drive_history'],
                              paths['manifest'], verbose=False)
            backup_and_record(paths['manifest'], paths['drive_manifest'],
                              paths['manifest'], verbose=False)

    writer.close()
    del train_loader, val_loader
    del model, encoder, optimizer, scheduler
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


# %% [markdown]
# ## 11. Final evaluation
#
# Greedy decoding on the held-out test split, scored with the corrected
# `evaluate_predictions` (exact match, algebraic equivalence via SymPy plus
# held-out R² after BFGS constant fitting).
#
# The weights come from `best.pt`, and **that file** is validated — seed,
# experiment version, and full run configuration — not merely `latest.pt`.
# `best['val']` and `latest['best_val']` are written in the same epoch, so they
# are the same float; a mismatch beyond float round-tripping means the two files
# come from different sessions and is a hard error.

# %%
def eval_one(pretrain_epochs, seed, synth_test):
    """Load best.pt and run greedy eval on the test set. Returns a metrics dict."""
    paths = run_paths(pretrain_epochs, seed)
    run_tag = str(paths['rel'])
    sup_cfg = make_run_config(pretrain_epochs, 'supervised')

    # Reuse a cached evaluation only if it came from THIS training config AND
    # THIS evaluator. A decoding change leaves the weights valid but the old
    # metrics wrong, so the two are versioned separately.
    if paths['metrics'].exists():
        with open(paths['metrics']) as f:
            cached = json.load(f)
        if cached.get('eval_version') != EVAL_VERSION:
            print(f'  {run_tag}: eval cache version mismatch '
                  f'(cached={cached.get("eval_version")!r}, '
                  f'current={EVAL_VERSION!r}); re-evaluating.')
        elif cached.get('run_config') != sup_cfg:
            raise CheckpointError(
                f'metrics.json run_config does not match this run\n'
                f'  {paths["metrics"]}')
        else:
            return cached

    latest_validate = make_validator(pretrain_epochs, seed, 'latest',
                                     'supervised', path=paths['sup_latest'])
    best_validate = make_validator(pretrain_epochs, seed, 'best',
                                   'supervised', path=paths['sup_best'])

    latest_ck, _ = load_local_or_restore(
        paths['sup_latest'], paths['drive_sup_latest'],
        validate=latest_validate, verbose=False)
    if latest_ck is None:
        raise CheckpointError(f'{run_tag}: no valid latest.pt to evaluate')

    best_ck, _ = load_local_or_restore(
        paths['sup_best'], paths['drive_sup_best'],
        validate=best_validate, verbose=False)
    if best_ck is None:
        raise CheckpointError(
            f'{run_tag}: no valid best.pt. Refusing to score latest.pt in its '
            f'place — that is a different model, and silently substituting it '
            f'would change what the number means.')

    check_best_latest_consistency(best_ck, latest_ck, path=paths['local_dir'])

    model, encoder, _ = build_model(dropout=0.0)
    model.load_state_dict(best_ck['model'])
    model.eval()

    eval_ds = synth_test
    if MAX_EVAL_EXAMPLES and MAX_EVAL_EXAMPLES < len(synth_test):
        # Prefix subset only: evaluate_predictions indexes dataset.samples[i].
        eval_ds = Subset(synth_test, range(MAX_EVAL_EXAMPLES))
    eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0)

    greedy_preds = []
    for batch in tqdm(eval_loader, desc=f'{run_tag} decode', leave=False):
        points = batch['points'].to(DEVICE)
        input_ids = batch['input_ids']
        preds = model.generate(points, tokenizer, max_new_tokens=MAX_SEQ)
        for j, pred_str in enumerate(preds):
            greedy_preds.append(
                (tokenizer.decode(input_ids[j].tolist()), pred_str))

    results = evaluate_predictions(greedy_preds, synth_test, tokenizer)

    metrics = {
        'experiment_version': EXPERIMENT_VERSION,
        'seed': seed,
        'run_tag': run_tag,
        'pretrain_epochs': pretrain_epochs,
        'run_config': sup_cfg,
        'eval_version': EVAL_VERSION,
        'best_epoch': latest_ck['best_epoch'],
        'best_val_loss': latest_ck['best_val'],
        'best_val_acc': latest_ck.get('best_val_acc', 0.0),
        'best_val_branch_acc': latest_ck.get('best_val_branch_acc', 0.0),
        # Encoder view-consistency at the best epoch: the direct read on what
        # Stage 1 was optimising. gap_centered is the headline.
        'best_view_consistency': latest_ck.get('best_view_consistency', {}),
        'greedy_exact_match': results['exact_match'],
        'greedy_token_acc': results['token_accuracy'],
        'greedy_algebraic_equiv': results['algebraic_equiv'],
        'greedy_r2_above_0.9': results['r2_above_0.9'],
        'mean_r2': results['mean_r2'],
        'median_r2': results['median_r2'],
        'n_parseable': results['n_parseable'],
        'n_r2_computed': results['n_r2_computed'],
        # Why R2 was not computable, by kind (constant_fit_failed,
        # pred_unparseable, ...). Sums to n_total.
        'r2_status_counts': results['r2_status_counts'],
        'n_total': results['n_total'],
        'details': results['details'],
    }

    save_json_atomic(metrics, paths['metrics'])
    backup_and_record(paths['metrics'], paths['drive_metrics'],
                      paths['manifest'], verbose=False)

    del model, encoder, eval_loader, best_ck, latest_ck
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


# %% [markdown]
# ## Run the full experiment
#
# The next cell launches the configured multi-seed experiment
# (`PRETRAIN_EPOCHS_VALUES × SEEDS` = 6 runs at the defaults) and can run for
# hours. All active checkpoints are written to the local
# `/content/symbolic_jepa_checkpoints/jepa_pretrain_v2` root and copied to
# Drive separately; a Drive failure prints a warning and training continues.
#
# Nothing above this cell starts training.

# %% tags=["long-running"]
print(f'Phase 1: training {len(PRETRAIN_EPOCHS_VALUES) * len(SEEDS)} runs '
      f'({EXPERIMENT_VERSION})')
for _pe in PRETRAIN_EPOCHS_VALUES:
    for _seed in SEEDS:
        train_one(_pe, _seed, synth_train, synth_val)

# %% tags=["long-running"]
# An evaluation crash (SymPy timeout, OOM, bug) is an INFRASTRUCTURE failure,
# not a model that scored 0. Writing zeros would drag that condition's averages
# down and make it look like a worse model, so successes and failures are kept
# apart and averages use successes only.
print(f'\n{"=" * 70}')
print(f'Phase 2: evaluating on the test set ({len(synth_test)} equations)')
print(f'{"=" * 70}')

_RUNS = [(p, s) for p in PRETRAIN_EPOCHS_VALUES for s in SEEDS]
successful_metrics = []
failed_runs = []

for _i, (_pe, _seed) in enumerate(_RUNS):
    _tag = f'pretrain_{_pe}/seed_{_seed}'
    _t0 = time.time()
    try:
        _m = eval_one(_pe, _seed, synth_test)
        print(f'  [{_i + 1}/{len(_RUNS)}] {_tag}: '
              f'exact={_m["greedy_exact_match"] * 100:.1f}% | '
              f'equiv={_m["greedy_algebraic_equiv"] * 100:.1f}% | '
              f'R²>.9={_m["greedy_r2_above_0.9"] * 100:.1f}% | '
              f'{time.time() - _t0:.0f}s')
        successful_metrics.append(_m)
    except Exception as _e:
        print(f'  [{_i + 1}/{len(_RUNS)}] {_tag}: FAILED after '
              f'{time.time() - _t0:.0f}s — {type(_e).__name__}: {_e}')
        failed_runs.append({'pretrain_epochs': _pe, 'seed': _seed,
                            'run_tag': _tag,
                            'error': f'{type(_e).__name__}: {_e}'})
    gc.collect()

# %% [markdown]
# ## 12. Run summary, manual Drive sync
#
# The paired `pretrain_epochs = 0` vs `10` comparison is by seed. Branching
# accuracy (`val_branch_acc`) is reported alongside ordinary token accuracy:
# a run can hold ~99% token accuracy while branching accuracy stays near
# chance, and that gap is the diagnostic this project cares about.

# %%
print(f'\n{"=" * 78}')
print(f'{EXPERIMENT_VERSION} — n_test={len(synth_test)}')
print(f'{"=" * 78}')
print(f'\n{"pre_ep":>7} {"seed":>6} {"val_loss":>10} {"val_acc":>9} '
      f'{"val_branch":>11} {"exact":>8} {"equiv":>8} {"R²>.9":>8} '
      f'{"gap_c":>8}')
print('-' * 78)
for _m in successful_metrics:
    print(f'{_m["pretrain_epochs"]:>7} {_m["seed"]:>6} '
          f'{_m["best_val_loss"]:>10.4f} '
          f'{_m["best_val_acc"] * 100:>8.1f}% '
          f'{_m["best_val_branch_acc"] * 100:>10.1f}% '
          f'{_m["greedy_exact_match"] * 100:>7.1f}% '
          f'{_m["greedy_algebraic_equiv"] * 100:>7.1f}% '
          f'{_m["greedy_r2_above_0.9"] * 100:>7.1f}% '
          f'{_m.get("best_view_consistency", {}).get("gap_centered", float("nan")):>8.4f}')

print(f'\n{"pre_ep":>7} {"avg_exact":>10} {"avg_equiv":>10} {"avg_R²>.9":>10} '
      f'{"avg_val_acc":>12} {"avg_branch":>11} {"avg_gap_c":>11} '
      f'{"n_ok":>6} {"n_fail":>7}')
print('-' * 78)
for _pe in PRETRAIN_EPOCHS_VALUES:
    _runs = [m for m in successful_metrics if m['pretrain_epochs'] == _pe]
    _nf = sum(1 for f in failed_runs if f['pretrain_epochs'] == _pe)
    if not _runs:
        print(f'{_pe:>7} {"—":>10} {"—":>10} {"—":>10} {"—":>12} {"—":>11} '
              f'{"—":>11} {0:>6} {_nf:>7}')
        continue
    print(f'{_pe:>7} '
          f'{np.mean([m["greedy_exact_match"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["greedy_algebraic_equiv"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["greedy_r2_above_0.9"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["best_val_acc"] for m in _runs]) * 100:>11.1f}% '
          f'{np.mean([m["best_val_branch_acc"] for m in _runs]) * 100:>10.1f}% '
          f'{np.mean([m.get("best_view_consistency", {}).get("gap_centered", float("nan")) for m in _runs]):>11.4f} '
          f'{len(_runs):>6} {_nf:>7}')

if failed_runs:
    print(f'\n{len(failed_runs)} FAILED EVALUATION(S) — excluded from averages')
    for _f in failed_runs:
        print(f'  {_f["run_tag"]}: {_f["error"]}')
    save_json_atomic(failed_runs, LOCAL_CHECKPOINT_ROOT / 'failed_runs.json')
else:
    print('\nAll evaluations succeeded.')

# %% [markdown]
# ### Manual Drive sync
#
# Run this **before deliberately shutting the runtime down**. It copies only
# completed, validated v2 artifacts (`latest.pt`, `best.pt`, `metrics.json`,
# `manifest.json`, `history.json`), skips in-flight and quarantined files, and
# never deletes anything locally or on Drive. v1 directories are untouched — it
# only ever walks the v2 local root.

# %%
sync_local_runs_to_drive(LOCAL_CHECKPOINT_ROOT, DRIVE_CHECKPOINT_ROOT)

# %% [markdown]
# ### Learning curves

# %%
import matplotlib.pyplot as plt


def load_history(pretrain_epochs, seed):
    p = run_paths(pretrain_epochs, seed)['history']
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


_fig, _ax = plt.subplots(1, 6, figsize=(30, 4))
_cmap = plt.get_cmap('viridis')
_col = {p: _cmap(i / max(len(PRETRAIN_EPOCHS_VALUES) - 1, 1))
        for i, p in enumerate(PRETRAIN_EPOCHS_VALUES)}

for _p in PRETRAIN_EPOCHS_VALUES:
    for _s in SEEDS:
        _h = load_history(_p, _s)
        if not _h:
            continue
        _lbl = f'pre{_p} s={_s}' + (' (baseline)' if _p == 0 else '')
        for _i, _key, _scale in ((0, 'train', 1), (1, 'val', 1),
                                 (2, 'val_acc', 100), (3, 'val_branch_acc', 100),
                                 (4, 'val_gap_centered', 1),
                                 (5, 'pretrain_subsample', 1)):
            _series = _h.get(_key) or []
            if _series:
                _ax[_i].plot(range(1, len(_series) + 1),
                             [v * _scale for v in _series],
                             color=_col[_p], alpha=.6, label=_lbl)

for _i, _title in enumerate(['Stage 2: train CE', 'Stage 2: val loss',
                             'val token acc %', 'val BRANCHING acc %',
                             'same - diff cosine (centered)',
                             'Stage 1: subsample loss']):
    _ax[_i].set_title(_title)
    _ax[_i].set_xlabel('pretrain epoch' if _i == 5 else 'finetune epoch')
    _ax[_i].grid(alpha=.3)
    _ax[_i].legend(fontsize=6)
plt.tight_layout()
plt.show()

# %%
# %load_ext tensorboard
# %tensorboard --logdir {LOCAL_LOG_ROOT}

# %% [markdown]
# ### Inspect predictions

# %%
def to_infix(prefix_str):
    """Prefix string -> infix via SymPy. Returns (infix_str, error_msg)."""
    try:
        expr, _ = prefix_to_sympy(prefix_str)
        return str(expr), None
    except Exception as e:
        return None, str(e)


def inspect_run(pretrain_epochs, seed, n_show=10):
    """Show per-equation predictions from a run's metrics.json."""
    p = run_paths(pretrain_epochs, seed)['metrics']
    if not p.exists():
        print(f'pretrain_{pretrain_epochs}/seed_{seed}: no metrics.json')
        return
    with open(p) as f:
        metrics = json.load(f)

    print(f'\n{"=" * 80}')
    print(f'pretrain_{pretrain_epochs}/seed_{seed} | '
          f'exact={metrics["greedy_exact_match"] * 100:.1f}% | '
          f'equiv={metrics["greedy_algebraic_equiv"] * 100:.1f}% | '
          f'val_branch_acc={metrics["best_val_branch_acc"] * 100:.1f}%')
    print(f'{"=" * 80}')

    details = metrics.get('details', [])
    for i, d in enumerate(details[:n_show]):
        gt_infix, _ = to_infix(d['gt'])
        pred_infix, pred_err = to_infix(d['pred'])
        r2 = f'{d["r2"]:.4f}' if d['r2'] is not None else 'N/A'
        status = ('EXACT' if d['exact'] else 'EQUIV' if d.get('equiv')
                  else 'PARSEABLE' if d.get('parseable') else 'UNPARSEABLE')
        print(f'\n── [{i}] {status} | R²={r2} ──')
        print(f'  GT:   {gt_infix}')
        if pred_err is None:
            print(f'  Pred: {pred_infix}')
        else:
            print(f'  Pred: PARSE FAILED ({pred_err})')
            print(f'        {d["pred"]}')

    print(f'\n  Total {len(details)} | '
          f'exact {sum(d["exact"] for d in details)} | '
          f'equiv {sum(1 for d in details if d.get("equiv"))} | '
          f'unparseable {sum(1 for d in details if not d.get("parseable", True))}')


for _p in PRETRAIN_EPOCHS_VALUES:
    inspect_run(_p, SEEDS[0], n_show=10)
