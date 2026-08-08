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
# # JEPA pretraining — v2 corrected baseline
#
# Same two-stage experiment as `jepa_pretrain.ipynb`, re-based on the corrected
# library (mentor commit `c313563`) and on local-first robust checkpointing.
#
# ```text
# Stage 1  JEPA-only pretraining   (pretrain_epochs)
#    |     encoder learns to predict a fixed symbolic target
#    v
# Stage 2  CE-only fine-tuning     (FINETUNE_EPOCHS)
#    |
#    v
#        greedy decode + symbolic evaluation
# ```
#
# Swept variable: `pretrain_epochs ∈ {0, 10}`, paired by seed. **`0` is the
# baseline** — Stage 1 is skipped and the run is plain CE training.
#
# ## What changed from v1, and why
#
# | Area | v1 | v2 |
# |---|---|---|
# | Raw expressions read | 10 000, no dedup | 200 000, one example per distinct token sequence |
# | Splits | random over expressions | grouped by token sequence (no verbatim leakage) |
# | `encode_expression` | tokens at positions 1..seq, no data slot | null data token at slot 0, positions 0..seq — matches `forward` |
# | Worker RNG | `num_workers=0` | `seed_worker` + explicitly seeded `torch.Generator` |
# | Val loss / accuracy | token-weighted already | token-weighted, **plus** branching-position accuracy |
# | Checkpoints | written straight to Drive | atomic local `/content` write, verified, then best-effort Drive backup |
#
# v2 shares **no** checkpoint path with v1 and never resumes from, migrates, or
# overwrites a v1 checkpoint. `run_config.experiment_version` makes that
# structurally impossible, not merely conventional.
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
    JEPAPredictor, IdentityPredictor, jepa_loss,
    build_synthetic_splits, load_synthetic_pkl,
    build_prefix_tree, teacher_forced_counts,
    evaluate_predictions,
    sym_spread, pred_spread, retrieval_top1, common_mode,
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
# Stamped into every run_config. A v1 checkpoint carries no such field and a
# future v3 carries a different one, so neither can validate as a v2 run.
EXPERIMENT_VERSION = 'jepa_pretrain_v2'

# Bump when DECODING or SCORING changes, independently of training. Trained
# checkpoints stay valid; only metrics.json is invalidated, so affected runs
# are re-scored rather than retrained.
EVAL_VERSION = 'v2_strict_eos'

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

# ── JEPA objective used in Stage 1 (unchanged from v1) ─────────────────
JEPA_LOSS = 'cosine'          # 'cosine' (ordinary) | 'centered'
JEPA_PREDICTOR = 'identity'   # 'identity' | 'mlp'
PRETRAIN_STOPGRAD = True      # see the note below — False collapses Stage 1

# ── Data ───────────────────────────────────────────────────────────────
SYNTH_PKL = str(REPO_DIR / 'data' / 'synthetic.pkl')
SYNTH_SEED = 42
MAX_SYNTH = 200_000           # raw strings read (mentor request)
DEDUPE_BY_TOKENS = True       # keep one example per distinct token sequence
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
# ### Why `PRETRAIN_STOPGRAD` must stay `True`
#
# Stage 1 optimises the JEPA loss alone. With gradients reaching both sides
# (`encoder` and `tok_embed`/`pos_embed`/`transformer`/`norm`, with `head.weight`
# tied to `tok_embed.weight`), the pair is free to converge on one constant
# vector: `cos = 1`, loss = 0, nothing learned. Measured over 6 JEPA-only epochs
# (d_model=128, 458 equations), `stopgrad=False` drove JEPA loss 1.028 → 0.0037
# while `std(z_sym)` fell 0.716 → 0.078 and off-diagonal `cos(z_sym)` rose
# 0.473 → 0.994 — the full collapse signature.
#
# With the stop-gradient, Stage 1 trains **only the T-Net encoder** against a
# fixed, randomly-initialised symbolic target. The decoder is not pretrained,
# so this measures *"does JEPA-pretraining the point-cloud encoder help?"*, not
# *"does pretraining the whole model help?"*
#
# `torch.no_grad()` alone is **not** sufficient — it blocks gradients but leaves
# dropout active, so the "fixed" target would be re-sampled every step. Stage 1
# therefore puts the model in `eval()` while computing the target.
#
# The mentor commit rewrote `encode_expression` to prepend a learned
# `null_data_token` and index the last real token as `attn_mask.sum()` rather
# than `attn_mask.sum() - 1`. v2 calls the library implementation directly; the
# old hand-rolled token alignment is **not** reproduced here.

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
    """RNG seed for one JEPA-pretraining epoch."""
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
        persistent_workers=NUM_WORKERS > 0,
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
# This cell parses and tokenizes 200 000 SymPy strings and can take tens of
# minutes. Lower `MAX_SYNTH` for a structural smoke test.

# %%
tokenizer = PrefixTokenizer(max_vars=MAX_VARS)
print(f'Vocab size: {len(tokenizer)}')

print(f'Loading synthetic expressions from {SYNTH_PKL} ...')
synth_exprs = load_synthetic_pkl(
    SYNTH_PKL,
    max_seq_len=MAX_SEQ,
    tokenizer=tokenizer,
    max_expressions=MAX_SYNTH,
    dedupe_by_tokens=DEDUPE_BY_TOKENS,
)
print(f'Kept {len(synth_exprs)} expressions')

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

# %%
synth_train, synth_val, synth_test = build_synthetic_splits(
    synth_exprs, tokenizer,
    n_points=N_POINTS, max_seq_len=MAX_SEQ, max_vars=MAX_VARS,
    seed=SYNTH_SEED,
    cache_eval=True,          # val/test clouds are deterministic — cache them
    group_by_tokens=GROUP_BY_TOKENS,
)

BRANCH_TREE = build_prefix_tree(synth_train.token_keys)

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
    """Fresh encoder + decoder + JEPA predictor on DEVICE."""
    encoder = TNet(d_input=D_INPUT, d_model=D_MODEL)
    model = SymbolicTransformer(
        encoder=encoder, vocab_size=len(tokenizer),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=4 * D_MODEL, max_seq_len=MAX_SEQ,
        dropout=dropout, pad_id=tokenizer.pad_id,
    ).to(DEVICE)

    # Always built: 'identity' is parameter-free, and both Stage 1 and the
    # validation diagnostics need it.
    predictor = (JEPAPredictor(D_MODEL).to(DEVICE) if JEPA_PREDICTOR == 'mlp'
                 else IdentityPredictor().to(DEVICE))
    return model, encoder, predictor


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
    """
    assert stage in ('pretrain', 'supervised')
    return {
        'experiment_version': EXPERIMENT_VERSION,
        'stage': stage,
        'pretrain_epochs': pretrain_epochs,
        'finetune_epochs': FINETUNE_EPOCHS,
        'pretrain_stopgrad': PRETRAIN_STOPGRAD,
        'jepa_loss': JEPA_LOSS,
        'jepa_predictor': JEPA_PREDICTOR,
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
                    paths, writer, train_loader, train_gen, history):
    """Stage 1: JEPA-only pretraining. Resumable, local-first."""
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
        if 'predictor' in ck and isinstance(predictor, JEPAPredictor):
            predictor.load_state_dict(ck['predictor'])
        for key, values in ck.get('history', {}).items():
            history[key] = list(values)
        start_epoch = ck['epoch'] + 1
        if start_epoch > pretrain_epochs:
            print(f'  Stage 1 already complete ({ck["epoch"]} epochs, '
                  f'source={source})')
            return
        print(f'  Stage 1 resuming at epoch {start_epoch} (source={source})')

    print(f'\n=== Stage 1: JEPA pretraining '
          f'({pretrain_epochs} epochs, loss = JEPA only) ===')
    print(f'  stopgrad={PRETRAIN_STOPGRAD}  trains='
          f'{"encoder only" if PRETRAIN_STOPGRAD else "encoder + decoder"}')

    for pe in range(start_epoch, pretrain_epochs + 1):
        # Resume-invariant: depends only on (seed, pretrain epoch).
        es = stage1_epoch_seed(seed, pe)
        seed_everything(es)
        train_gen.manual_seed(es)

        model.train()
        predictor.train()
        pre_sum = 0.0
        pre_n = 0
        std_num_sum = std_sym_sum = 0.0

        pbar = tqdm(train_loader, leave=False,
                    desc=f'{paths["rel"]} P{pe}/{pretrain_epochs}')
        for batch in pbar:
            points = batch['points'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attn_mask = batch['attn_mask'].to(DEVICE, non_blocking=True)

            pre_opt.zero_grad()
            with amp_ctx():
                if PRETRAIN_STOPGRAD:
                    # eval() first: no_grad blocks gradients but leaves dropout
                    # active, which would make the "fixed" target move.
                    model.eval()
                    with torch.no_grad():
                        z_sym = model.encode_expression(
                            input_ids, attn_mask=attn_mask)
                    model.train()
                else:
                    z_sym = model.encode_expression(
                        input_ids, attn_mask=attn_mask)
                z_num = model.encoder(points)
                loss_pre = jepa_loss(predictor(z_num), z_sym, mode=JEPA_LOSS)

            loss_pre.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            pre_opt.step()

            pre_sum += loss_pre.item()
            pre_n += 1
            # Averaged over the epoch, not read off the last batch: a per-batch
            # reading jitters with which equations land last and would mask a
            # real collapse trend.
            std_num_sum += z_num.detach().float().std(dim=0).mean().item()
            std_sym_sum += z_sym.detach().float().std(dim=0).mean().item()
            pbar.set_postfix({'jepa': f'{loss_pre.item():.4f}'})
        pbar.close()
        del pbar

        pre_avg = pre_sum / max(pre_n, 1)
        std_num = std_num_sum / max(pre_n, 1)
        std_sym = std_sym_sum / max(pre_n, 1)
        history.setdefault('pretrain_jepa', []).append(pre_avg)
        history.setdefault('pretrain_std_num', []).append(std_num)
        history.setdefault('pretrain_std_sym', []).append(std_sym)
        writer.add_scalar('pretrain/jepa_loss', pre_avg, pe)
        writer.add_scalar('pretrain/std_z_num', std_num, pe)
        writer.add_scalar('pretrain/std_z_sym', std_sym, pe)
        print(f'  P{pe}/{pretrain_epochs} | jepa={pre_avg:.4f} | '
              f'std(z_num)={std_num:.4f} std(z_sym)={std_sym:.4f}')

        pre_ck = {
            'model': model.state_dict(),
            'optimizer': pre_opt.state_dict(),
            'epoch': pe,
            'seed': seed,
            'run_config': make_run_config(pretrain_epochs, 'pretrain'),
            'history': {k: v for k, v in history.items()
                        if k.startswith('pretrain_')},
        }
        if isinstance(predictor, JEPAPredictor):
            pre_ck['predictor'] = predictor.state_dict()

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
def validate_epoch(model, predictor, val_loader):
    """One full validation pass with token-weighted aggregation.

    Returns loss, ordinary token accuracy, branching-position accuracy, their
    denominators, and the JEPA alignment diagnostics.
    """
    model.eval()
    predictor.eval()

    val_loss_sum = 0.0
    val_tokens_total = 0
    acc_correct = acc_total = 0.0
    branch_correct = branch_total = 0.0
    align_sum = 0.0
    n_batches = 0
    diag_z_sym = diag_z_pred = None

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

            z_sym_v = model.encode_expression(input_ids, attn_mask=attn_mask)
            z_pred_v = predictor(out['z_num'])
            align_sum += jepa_loss(z_pred_v, z_sym_v, mode=JEPA_LOSS).item()
            if diag_z_sym is None:
                diag_z_sym, diag_z_pred = z_sym_v, z_pred_v
            n_batches += 1

    return {
        'val': val_loss_sum / max(val_tokens_total, 1),
        'val_acc': acc_correct / max(acc_total, 1),
        'val_branch_acc': branch_correct / max(branch_total, 1),
        'n_tokens': val_tokens_total,
        'n_branch_tokens': int(branch_total),
        'jepa': align_sum / max(n_batches, 1),
        'z_sym': diag_z_sym,
        'z_pred': diag_z_pred,
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
          f'finetune={FINETUNE_EPOCHS}ep | jepa_loss={JEPA_LOSS} | '
          f'stopgrad={PRETRAIN_STOPGRAD} | {EXPERIMENT_VERSION}')
    print(f'{"=" * 70}')

    model, encoder, predictor = build_model()
    params = list(model.parameters()) + list(predictor.parameters())

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
    best_epoch = 0
    history = {'train': [], 'val': []}

    # local -> Drive-restored-into-local -> fresh. Never trains from Drive.
    sup_ck, sup_source = load_local_or_restore(
        paths['sup_latest'], paths['drive_sup_latest'], validate=latest_validate)

    if sup_ck is not None:
        model.load_state_dict(sup_ck['model'])
        optimizer.load_state_dict(sup_ck['optimizer'])
        scheduler.load_state_dict(sup_ck['scheduler'])
        if 'predictor' in sup_ck and isinstance(predictor, JEPAPredictor):
            predictor.load_state_dict(sup_ck['predictor'])
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
        del model, encoder, predictor, optimizer, scheduler
        del train_loader, val_loader
        gc.collect()
        return

    # ── Stage 1 ────────────────────────────────────────────────────────
    # Skipped on resume: the pretrained weights are already inside sup_ck.
    if sup_ck is None:
        run_pretraining(model, predictor, params, pretrain_epochs, seed,
                        paths, writer, train_loader, train_gen, history)
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=FINETUNE_EPOCHS)
    del sup_ck

    # ── Stage 2 ────────────────────────────────────────────────────────
    print(f'\n=== Stage 2: CE fine-tuning ({FINETUNE_EPOCHS} epochs, '
          f'loss = CE only, epoch seed = {seed} + 100000 + epoch) ===')

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        es = stage2_epoch_seed(seed, epoch)
        seed_everything(es)
        train_gen.manual_seed(es)

        model.train()
        predictor.train()
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
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
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
            v = validate_epoch(model, predictor, val_loader)
            history['val'].append(v['val'])
            history.setdefault('val_acc', []).append(v['val_acc'])
            history.setdefault('val_branch_acc', []).append(v['val_branch_acc'])
            history.setdefault('val_jepa_raw', []).append(v['jepa'])

            writer.add_scalar('val/loss', v['val'], epoch)
            writer.add_scalar('val/token_accuracy', v['val_acc'], epoch)
            writer.add_scalar('val/token_accuracy_branching',
                              v['val_branch_acc'], epoch)
            writer.add_scalar('val/jepa_raw', v['jepa'], epoch)

            ss = sym_spread(v['z_sym'])
            ps = pred_spread(v['z_pred'])
            rt = retrieval_top1(v['z_pred'], v['z_sym'])
            cm = common_mode(v['z_sym'])
            writer.add_scalar('val/sym_spread_raw', ss['raw'], epoch)
            writer.add_scalar('val/sym_spread_cent', ss['centered'], epoch)
            writer.add_scalar('val/pred_spread', ps['raw'], epoch)
            writer.add_scalar('val/retrieval_top1', rt['centered'], epoch)
            writer.add_scalar('val/common_mode_ratio',
                              cm['mean_norm_ratio'], epoch)

            is_best = v['val'] < best_val
            if is_best:
                best_val = v['val']
                best_val_acc = v['val_acc']
                best_val_branch_acc = v['val_branch_acc']
                best_epoch = epoch

            print(f'  E{epoch}/{FINETUNE_EPOCHS} | train={train_avg:.4f} | '
                  f'val={v["val"]:.4f}{" * best" if is_best else ""} | '
                  f'acc={v["val_acc"] * 100:.1f}% | '
                  f'branch_acc={v["val_branch_acc"] * 100:.1f}% '
                  f'(n={v["n_branch_tokens"]}) | '
                  f'jepa={v["jepa"]:.4f} | retr={rt["centered"]:.2f}')

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
            'best_epoch': best_epoch,
            'history': history,
            'seed': seed,
            'run_config': sup_cfg,
        }
        if isinstance(predictor, JEPAPredictor):
            ckpt['predictor'] = predictor.state_dict()

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
    del train_loader, val_loader, model, encoder, predictor, optimizer, scheduler
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
        'greedy_exact_match': results['exact_match'],
        'greedy_token_acc': results['token_accuracy'],
        'greedy_algebraic_equiv': results['algebraic_equiv'],
        'greedy_r2_above_0.9': results['r2_above_0.9'],
        'mean_r2': results['mean_r2'],
        'median_r2': results['median_r2'],
        'n_parseable': results['n_parseable'],
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
      f'{"val_branch":>11} {"exact":>8} {"equiv":>8} {"R²>.9":>8}')
print('-' * 78)
for _m in successful_metrics:
    print(f'{_m["pretrain_epochs"]:>7} {_m["seed"]:>6} '
          f'{_m["best_val_loss"]:>10.4f} '
          f'{_m["best_val_acc"] * 100:>8.1f}% '
          f'{_m["best_val_branch_acc"] * 100:>10.1f}% '
          f'{_m["greedy_exact_match"] * 100:>7.1f}% '
          f'{_m["greedy_algebraic_equiv"] * 100:>7.1f}% '
          f'{_m["greedy_r2_above_0.9"] * 100:>7.1f}%')

print(f'\n{"pre_ep":>7} {"avg_exact":>10} {"avg_equiv":>10} {"avg_R²>.9":>10} '
      f'{"avg_val_acc":>12} {"avg_branch":>11} {"n_ok":>6} {"n_fail":>7}')
print('-' * 78)
for _pe in PRETRAIN_EPOCHS_VALUES:
    _runs = [m for m in successful_metrics if m['pretrain_epochs'] == _pe]
    _nf = sum(1 for f in failed_runs if f['pretrain_epochs'] == _pe)
    if not _runs:
        print(f'{_pe:>7} {"—":>10} {"—":>10} {"—":>10} {"—":>12} {"—":>11} '
              f'{0:>6} {_nf:>7}')
        continue
    print(f'{_pe:>7} '
          f'{np.mean([m["greedy_exact_match"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["greedy_algebraic_equiv"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["greedy_r2_above_0.9"] for m in _runs]) * 100:>9.1f}% '
          f'{np.mean([m["best_val_acc"] for m in _runs]) * 100:>11.1f}% '
          f'{np.mean([m["best_val_branch_acc"] for m in _runs]) * 100:>10.1f}% '
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


_fig, _ax = plt.subplots(1, 4, figsize=(20, 4))
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
                                 (2, 'val_acc', 100), (3, 'val_branch_acc', 100)):
            _series = _h.get(_key) or []
            if _series:
                _ax[_i].plot(range(1, len(_series) + 1),
                             [v * _scale for v in _series],
                             color=_col[_p], alpha=.6, label=_lbl)

for _i, _title in enumerate(['Stage 2: train CE', 'Stage 2: val loss',
                             'val token acc %', 'val BRANCHING acc %']):
    _ax[_i].set_title(_title)
    _ax[_i].set_xlabel('finetune epoch')
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
