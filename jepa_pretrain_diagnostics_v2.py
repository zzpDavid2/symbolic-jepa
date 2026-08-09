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
# # JEPA pretraining v2 — diagnostics
#
# Sanity checks, ablations, and checkpoint-infrastructure tests for
# `jepa_pretrain_v2.ipynb`. Kept separate so the training notebook stays a
# clean experiment.
#
# **This notebook never writes into the v2 experiment checkpoint tree.** It
# reads trained checkpoints (validated, read-only) and writes its own outputs
# under a separate diagnostics root.
#
# | Section | What it answers | Needs a trained checkpoint? |
# |---|---|---|
# | 4 | Which v2 checkpoint am I looking at? | yes |
# | 5 | Mentor's real-vs-zero-input check | yes |
# | 6 | real / zero / shuffled × token / branching accuracy | yes |
# | 7 | Do DataLoader workers still emit identical clouds? | no |
# | 8 | Do the checkpoint helpers actually hold their invariants? | no |
# | 9 | Can the model memorise a tiny set? (optional, off) | no |
# | 10 | Is an old v1 checkpoint intact? (optional, read-only) | no |
#
# Sections 7–10 run without any trained model, so the notebook is executable
# top-to-bottom in a fresh runtime.

# %% [markdown]
# ## 0. Colab / environment setup

# %%
import os
import sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

DRIVE_BASE = Path('/content/drive/MyDrive/Symba')
REPO_URL = 'https://github.com/zzpDavid2/symbolic-jepa.git'


def sh(cmd: str) -> int:
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
        print(f'[backup WARNING] Drive mount failed: {type(e).__name__}: {e}')

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
print(f'symbolic_jepa imported from: {symbolic_jepa.__file__}')

# %% [markdown]
# ## 1. Imports and configuration
#
# The constants below **must match `jepa_pretrain_v2.ipynb`**. They are not
# imported from it (a notebook is not a module), but they do not have to be
# trusted either: every checkpoint load in section 4 runs the same
# `validate_checkpoint` the training notebook uses, so any drift in this cell
# fails loudly with a config diff instead of producing a quietly wrong
# diagnostic.

# %%
import json
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from symbolic_jepa import (
    PrefixTokenizer,
    TNet, SymbolicTransformer,
    JEPAPredictor, IdentityPredictor,
    build_synthetic_splits,
    teacher_forced_counts,
)
from symbolic_jepa.datacache import (
    DataCache, cached_synthetic_expressions, cached_prefix_tree,
)
from symbolic_jepa.checkpointing import (
    CheckpointError,
    validate_checkpoint, save_checkpoint_atomic, save_json_atomic,
    load_checkpoint, load_local_or_restore,
    backup_checkpoint_to_drive, check_best_latest_consistency,
    is_temp_name,
)

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
# ══ Must mirror jepa_pretrain_v2.ipynb ═════════════════════════════════
EXPERIMENT_VERSION = 'jepa_pretrain_v2'
EVAL_VERSION = 'v2_strict_eos'

MAX_VARS = 1
D_INPUT = MAX_VARS + 1
N_POINTS = 1000
MAX_SEQ = 64
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
DROPOUT = 0.2

LR = 3e-4
WEIGHT_DECAY = 0.1
BATCH = 16

FINETUNE_EPOCHS = 30
PRETRAIN_EPOCHS_VALUES = [0, 10]
SEEDS = [42, 123, 7]
GLOBAL_SEED = SEEDS[0]

JEPA_LOSS = 'cosine'
JEPA_PREDICTOR = 'identity'
PRETRAIN_STOPGRAD = True

SYNTH_PKL = str(REPO_DIR / 'data' / 'synthetic.pkl')
SYNTH_SEED = 42
MAX_SYNTH = 200_000
DEDUPE_BY_TOKENS = True
USE_DATA_CACHE = True
GROUP_BY_TOKENS = True

NUM_WORKERS = 2
# ═══════════════════════════════════════════════════════════════════════

# ── Which trained run to diagnose ──────────────────────────────────────
DIAG_PRETRAIN_EPOCHS = 10
DIAG_SEED = 42

# Cap the ablation pass; 0 = whole validation split. The ablation runs three
# forward passes per batch, so a cap is useful for a quick look.
DIAG_MAX_VAL_BATCHES = 0

# ── Roots ──────────────────────────────────────────────────────────────
# Read-only source of trained checkpoints (owned by the training notebook).
if IN_COLAB:
    LOCAL_CHECKPOINT_ROOT = Path('/content/symbolic_jepa_checkpoints') / EXPERIMENT_VERSION
    DRIVE_CHECKPOINT_ROOT = (
        (DRIVE_BASE / 'symbolic_jepa_checkpoints') / EXPERIMENT_VERSION
        if DRIVE_MOUNTED else None
    )
    DIAGNOSTICS_ROOT = Path('/content/symbolic_jepa_diagnostics') / EXPERIMENT_VERSION
    DRIVE_DIAGNOSTICS_ROOT = (
        (DRIVE_BASE / 'symbolic_jepa_diagnostics') / EXPERIMENT_VERSION
        if DRIVE_MOUNTED else None
    )
else:
    LOCAL_CHECKPOINT_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_checkpoints' / EXPERIMENT_VERSION
    DRIVE_CHECKPOINT_ROOT = None
    DIAGNOSTICS_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_diagnostics' / EXPERIMENT_VERSION
    DRIVE_DIAGNOSTICS_ROOT = None

DIAGNOSTICS_ROOT.mkdir(parents=True, exist_ok=True)
# Scratch space for the checkpoint self-tests — never the real run directory.
SELFTEST_ROOT = DIAGNOSTICS_ROOT / 'selftests'

print(f'checkpoints (read-only) : {LOCAL_CHECKPOINT_ROOT}')
print(f'drive backups (read-only): {DRIVE_CHECKPOINT_ROOT}')
print(f'diagnostics outputs     : {DIAGNOSTICS_ROOT}')
print(f'self-test scratch       : {SELFTEST_ROOT}')

# %%
RUN_FILES = {
    'pre_latest': 'pretrain/latest.pt',
    'sup_latest': 'supervised/latest.pt',
    'sup_best': 'supervised/best.pt',
    'metrics': 'supervised/metrics.json',
    'manifest': 'supervised/manifest.json',
    'history': 'supervised/history.json',
}


def run_paths(pretrain_epochs: int, seed: int) -> dict:
    rel = Path(f'pretrain_{pretrain_epochs}') / f'seed_{seed}'
    paths = {'rel': rel, 'local_dir': LOCAL_CHECKPOINT_ROOT / rel}
    for key, sub in RUN_FILES.items():
        paths[key] = LOCAL_CHECKPOINT_ROOT / rel / sub
        paths[f'drive_{key}'] = (
            None if DRIVE_CHECKPOINT_ROOT is None
            else DRIVE_CHECKPOINT_ROOT / rel / sub
        )
    return paths


def make_run_config(pretrain_epochs: int, stage: str) -> dict:
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


LATEST_EXTRA = ('optimizer', 'scheduler', 'history', 'best_val', 'best_epoch')


def make_validator(pretrain_epochs, seed, kind, stage, path=None):
    cfg = make_run_config(pretrain_epochs, stage)
    extra = LATEST_EXTRA if kind == 'latest' else ()
    max_epoch = pretrain_epochs if stage == 'pretrain' else FINETUNE_EPOCHS

    def _validate(ck):
        validate_checkpoint(
            ck, expected_seed=seed, expected_run_config=cfg, kind=kind,
            extra_required=extra, max_epoch=max_epoch, path=path,
        )

    return _validate


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


seed_everything(GLOBAL_SEED)

# %% [markdown]
# ## 2. Data, splits, and the branching prefix tree
#
# Identical construction to the training notebook: same tokenizer, same
# `max_expressions` / dedup, same grouped split, same `SYNTH_SEED`. That is what
# makes the ablation apples-to-apples with the reported validation numbers.
#
# This cell shares the training notebook's derived-artifact cache, so once
# `jepa_pretrain_v2.ipynb` has parsed this configuration the expressions and the
# branch tree load in seconds instead of being re-parsed (~10-15 min). The
# point-cloud probe still runs (~1 min). The branch-tree entry is keyed on the
# training token sequences themselves, so this notebook provably gets the same
# tree training used. Lower `MAX_SYNTH` only if you also lower it in the
# training notebook — otherwise the splits diverge and section 4 will refuse
# the checkpoint.

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

synth_exprs = cached_synthetic_expressions(
    SYNTH_PKL, tokenizer,
    max_seq_len=MAX_SEQ,
    max_vars=MAX_VARS,
    max_expressions=MAX_SYNTH,
    dedupe_by_tokens=DEDUPE_BY_TOKENS,
    cache=CACHE,
    progress=True,
)

synth_train, synth_val, synth_test = build_synthetic_splits(
    synth_exprs, tokenizer,
    n_points=N_POINTS, max_seq_len=MAX_SEQ, max_vars=MAX_VARS,
    seed=SYNTH_SEED,
    cache_eval=True,
    group_by_tokens=GROUP_BY_TOKENS,
    progress=True,
)

BRANCH_TREE = cached_prefix_tree(synth_train.token_keys, cache=CACHE,
                                 progress=True)
print(f'\nbranch tree: {len(BRANCH_TREE)} prefixes, '
      f'{sum(1 for v in BRANCH_TREE.values() if v > 1)} branching')

val_loader = DataLoader(synth_val, batch_size=BATCH, shuffle=False,
                        num_workers=0)

# %% [markdown]
# ## 3. Model construction

# %%
def build_model(dropout: float = DROPOUT):
    encoder = TNet(d_input=D_INPUT, d_model=D_MODEL)
    model = SymbolicTransformer(
        encoder=encoder, vocab_size=len(tokenizer),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=4 * D_MODEL, max_seq_len=MAX_SEQ,
        dropout=dropout, pad_id=tokenizer.pad_id,
    ).to(DEVICE)
    predictor = (JEPAPredictor(D_MODEL).to(DEVICE) if JEPA_PREDICTOR == 'mlp'
                 else IdentityPredictor().to(DEVICE))
    return model, encoder, predictor


# %% [markdown]
# ## 4. Load one trained v2 checkpoint (validated, read-only)
#
# `best.pt` is loaded through the same validator the training notebook uses, so
# the checkpoint's seed, experiment version and full run configuration are
# checked against the constants in this notebook before anything is measured.
# Nothing here writes to the run directory.

# %%
DIAG_PATHS = run_paths(DIAG_PRETRAIN_EPOCHS, DIAG_SEED)
MODEL = None
BEST_CK = None

try:
    BEST_CK = load_checkpoint(
        DIAG_PATHS['sup_best'],
        validate=make_validator(DIAG_PRETRAIN_EPOCHS, DIAG_SEED, 'best',
                                'supervised', path=DIAG_PATHS['sup_best']),
    )
except CheckpointError as e:
    print(f'[diagnostics] no usable checkpoint yet:\n{e}\n')
    print('Sections 5 and 6 will be skipped; sections 7-10 still run.')

if BEST_CK is not None:
    MODEL, _ENCODER, _PREDICTOR = build_model(dropout=0.0)
    MODEL.load_state_dict(BEST_CK['model'])
    MODEL.eval()

    if DIAG_PATHS['sup_latest'].exists():
        _latest = load_checkpoint(
            DIAG_PATHS['sup_latest'],
            validate=make_validator(DIAG_PRETRAIN_EPOCHS, DIAG_SEED, 'latest',
                                    'supervised',
                                    path=DIAG_PATHS['sup_latest']),
        )
        check_best_latest_consistency(BEST_CK, _latest,
                                      path=DIAG_PATHS['local_dir'])
        print(f'best.pt / latest.pt consistent '
              f'(best epoch {BEST_CK["epoch"]}, latest epoch {_latest["epoch"]})')
        del _latest

    print(f'\nDiagnosing {DIAG_PATHS["rel"]}')
    print(f'  best epoch          : {BEST_CK["epoch"]}')
    print(f'  val loss            : {BEST_CK["val"]:.4f}')
    print(f'  val token acc       : {BEST_CK["val_acc"] * 100:.2f}%')
    print(f'  val branching acc   : '
          f'{BEST_CK.get("val_branch_acc", float("nan")) * 100:.2f}%')

# %% [markdown]
# ## 5. The mentor's zero-input sanity check
#
# > If `acc_zero` is close to your normal number, the encoder is contributing
# > nothing.
#
# Reported exactly as requested so the number is directly comparable with the
# mentor's expectation. The richer table follows in section 6.

# %% tags=["diagnostic"]
if MODEL is None:
    print('Skipped: no trained checkpoint loaded (see section 4).')
else:
    MODEL.eval()
    real = zero = tot_r = tot_z = 0.0

    with torch.no_grad():
        for batch in val_loader:
            pts = batch['points'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            m = batch['attn_mask'].to(DEVICE)

            c, t = teacher_forced_counts(
                MODEL(pts, ids, attn_mask=m)['logits'], ids, tokenizer.pad_id)
            real += c
            tot_r += t

            c, t = teacher_forced_counts(
                MODEL(torch.zeros_like(pts), ids, attn_mask=m)['logits'],
                ids, tokenizer.pad_id)
            zero += c
            tot_z += t

    print(f'real={100 * real / tot_r:.1f}%  '
          f'zeroed-input={100 * zero / tot_z:.1f}%')

# %% [markdown]
# ## 6. Real vs zero vs shuffled input, on both metrics
#
# Three conditions over the same validation batches:
#
# * **real** — the matching point cloud.
# * **zero** — an all-zero cloud. Out of distribution: after the encoder's
#   BatchNorm/max-pool this is one fixed vector, so it mostly measures "what
#   does the decoder do with a constant data token".
# * **shuffled** — a *valid* cloud belonging to a different expression, taken as
#   `torch.roll(pts, shifts=1, dims=0)`. A deterministic cyclic shift, not a
#   random permutation: a permutation can leave fixed points, which would
#   silently pair some rows with their own cloud. This condition stays fully
#   in-distribution at the point-cloud level and is therefore the more
#   informative ablation.
#
# A batch of size 1 cannot be shuffled (rolling it returns the same cloud), so
# such a batch is **skipped for the shuffled condition only** and its tokens are
# excluded from that row's denominators. Real and zero still use it, which is
# why the `n_tokens` column can differ by row.
#
# ### How to read the result
#
# * zero/shuffled close to real on **overall** accuracy but much worse on
#   **branching** accuracy → the encoder contributes mainly at genuinely
#   ambiguous symbolic decisions, and overall accuracy is being carried by
#   deterministic syntax.
# * zero/shuffled close to real even on **branching** accuracy → strong evidence
#   the decoder is largely ignoring the numerical encoder.

# %% tags=["diagnostic"]
def ablation_counts(model, loader, branch_tree, max_batches=0):
    """(token, branch) counts for real / zero / shuffled point clouds."""
    stats = {c: {'tok_c': 0.0, 'tok_t': 0.0, 'br_c': 0.0, 'br_t': 0.0}
             for c in ('real', 'zero', 'shuffled')}
    n_skipped_singleton = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            pts = batch['points'].to(DEVICE)
            ids = batch['input_ids'].to(DEVICE)
            m = batch['attn_mask'].to(DEVICE)

            conditions = {'real': pts, 'zero': torch.zeros_like(pts)}
            if pts.shape[0] > 1:
                conditions['shuffled'] = torch.roll(pts, shifts=1, dims=0)
            else:
                n_skipped_singleton += 1

            for name, cloud in conditions.items():
                logits = model(cloud, ids, attn_mask=m)['logits']
                c, t = teacher_forced_counts(logits, ids, tokenizer.pad_id)
                stats[name]['tok_c'] += c
                stats[name]['tok_t'] += t
                cb, tb = teacher_forced_counts(logits, ids, tokenizer.pad_id,
                                               branch_tree=branch_tree)
                stats[name]['br_c'] += cb
                stats[name]['br_t'] += tb

    return stats, n_skipped_singleton


if MODEL is None:
    print('Skipped: no trained checkpoint loaded (see section 4).')
else:
    _t0 = time.time()
    _stats, _skipped = ablation_counts(MODEL, val_loader, BRANCH_TREE,
                                       max_batches=DIAG_MAX_VAL_BATCHES)
    print(f'ablation over {"all" if not DIAG_MAX_VAL_BATCHES else DIAG_MAX_VAL_BATCHES} '
          f'val batches in {time.time() - _t0:.0f}s '
          f'({_skipped} singleton batch(es) skipped for shuffled)\n')

    _acc = {k: (v['tok_c'] / max(v['tok_t'], 1),
                v['br_c'] / max(v['br_t'], 1)) for k, v in _stats.items()}
    _base_tok, _base_br = _acc['real']

    print(f'{"condition":<12}{"token_acc":>11}{"branch_acc":>12}'
          f'{"n_tokens":>11}{"n_branch":>10}'
          f'{"delta_token":>13}{"delta_branch":>14}')
    print('-' * 83)
    for _name in ('real', 'zero', 'shuffled'):
        _tok, _br = _acc[_name]
        _s = _stats[_name]
        _dt = '' if _name == 'real' else f'{(_tok - _base_tok) * 100:>+12.2f}%'
        _db = '' if _name == 'real' else f'{(_br - _base_br) * 100:>+13.2f}%'
        print(f'{_name:<12}{_tok * 100:>10.2f}%{_br * 100:>11.2f}%'
              f'{int(_s["tok_t"]):>11}{int(_s["br_t"]):>10}{_dt:>13}{_db:>14}')

    print('\nInterpretation: compare the two delta columns. A large '
          'delta_branch with a small\ndelta_token means the encoder matters '
          'only where the syntax leaves a real choice.\nBoth deltas near zero '
          'means the decoder is ignoring the point cloud.')

# %% [markdown]
# ## 7. DataLoader worker RNG sanity check
#
# The mentor's observation: forked workers inherit identical NumPy/Python RNG
# state and emit identical clouds. This check uses a tiny standalone dataset
# that simply returns a draw from each RNG — the production dataset is not
# modified to expose worker IDs.
#
# Two loaders are compared: one **without** `worker_init_fn` (the old behaviour)
# and one **with** `seed_worker`. Distinct draws per worker in the second case
# is the property we need. Fast, no training, nothing written.
#
# On platforms that spawn rather than fork workers (macOS/Windows default), the
# unpatched loader may already produce distinct streams; the patched one must
# still produce distinct streams, which is what is asserted.

# %% tags=["diagnostic"]
class _RNGProbe(Dataset):
    """Returns whatever the worker's NumPy/Python RNG produces next."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        info = torch.utils.data.get_worker_info()
        return {
            'worker': -1 if info is None else info.id,
            'numpy': float(np.random.rand()),
            'python': random.random(),
        }


def _worker_streams(worker_init_fn, n_workers=2, n_items=8):
    """Map worker id -> the (numpy, python) draws it produced."""
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    loader = DataLoader(
        _RNGProbe(n_items), batch_size=1, shuffle=False,
        num_workers=n_workers, worker_init_fn=worker_init_fn, generator=g,
    )
    streams: dict[int, list[tuple[float, float]]] = {}
    for item in loader:
        wid = int(item['worker'][0])
        streams.setdefault(wid, []).append(
            (float(item['numpy'][0]), float(item['python'][0])))
    return streams


if NUM_WORKERS < 1:
    print('NUM_WORKERS = 0 — sampling happens in the main process, so there '
          'is no worker RNG to check.')
else:
    _patched_first = None
    for _label, _init in (('without seed_worker', None),
                          ('with seed_worker', seed_worker)):
        try:
            _streams = _worker_streams(_init)
        except Exception as _e:
            # Under the 'spawn' start method a class defined in the notebook
            # cannot be sent to a worker. Colab forks, so this is a local-dev
            # limitation, not a finding about the RNG fix.
            print(f'{_label:<22} could not start workers '
                  f'({type(_e).__name__}: {_e})')
            print('  Multiprocessing start method: '
                  f'{torch.multiprocessing.get_start_method()} — this check '
                  'needs "fork" (the Colab default).')
            break

        _first = {w: s[0] for w, s in sorted(_streams.items())}
        _distinct = len(set(_first.values())) == len(_first)
        print(f'{_label:<22} workers={sorted(_first)}  '
              f'distinct_first_draw={_distinct}')
        for _w, _draw in _first.items():
            print(f'    worker {_w}: numpy={_draw[0]:.6f} python={_draw[1]:.6f}')
        if _init is seed_worker:
            _patched_first = _first

    if _patched_first is None:
        print('\nSKIPPED: worker RNG could not be probed in this environment.')
    else:
        _firsts = list(_patched_first.values())
        assert len(set(_firsts)) == len(_firsts), (
            'seed_worker did not de-correlate the worker RNG streams')
        print('\nPASS: with seed_worker, every worker starts from a distinct '
              'NumPy/Python state.')

# %% [markdown]
# ## 8. Checkpoint infrastructure self-tests
#
# Exercises `symbolic_jepa/checkpointing.py` inside a temporary directory under
# the diagnostics root. The real experiment tree is never touched, and the
# "Drive" side of the backup/restore tests is another temporary local directory
# — no real Drive data is read, written, or unmounted.
#
# Covered: (A) atomic save + load-back, (B) an invalid candidate must not
# replace a valid final file, (C) a backup failure is non-fatal, (D) restore
# from backup, (E) provenance mismatches are rejected, (F) `best.pt` /
# `latest.pt` consistency.

# %% tags=["checkpoint-test"]
def _toy_run_config(pretrain_epochs=10, stage='supervised', version=EXPERIMENT_VERSION):
    cfg = make_run_config(pretrain_epochs, stage)
    cfg['experiment_version'] = version
    return cfg


def _toy_latest(epoch=3, seed=42, best_val=0.25, pretrain_epochs=10):
    return {
        'model': {'w': torch.zeros(2)},
        'optimizer': {'state': {}},
        'scheduler': {'last_epoch': epoch},
        'history': {'train': [1.0], 'val': [0.25]},
        'epoch': epoch,
        'best_val': best_val,
        'best_val_acc': 0.5,
        'best_epoch': epoch,
        'seed': seed,
        'run_config': _toy_run_config(pretrain_epochs),
    }


def _toy_best(epoch=3, seed=42, val=0.25, pretrain_epochs=10):
    return {
        'model': {'w': torch.zeros(2)},
        'epoch': epoch,
        'val': val,
        'val_acc': 0.5,
        'val_branch_acc': 0.3,
        'seed': seed,
        'run_config': _toy_run_config(pretrain_epochs),
    }


_LATEST_V = make_validator(10, 42, 'latest', 'supervised')
_BEST_V = make_validator(10, 42, 'best', 'supervised')

_results = []


def _check(name, fn):
    try:
        fn()
    except Exception as e:
        _results.append((name, f'FAIL — {type(e).__name__}: {e}'))
    else:
        _results.append((name, 'pass'))


if SELFTEST_ROOT.exists():
    shutil.rmtree(SELFTEST_ROOT)
SELFTEST_ROOT.mkdir(parents=True, exist_ok=True)


# ── A. Atomic local save / load-back ───────────────────────────────────
def _test_a():
    d = SELFTEST_ROOT / 'A'
    final = d / 'latest.pt'
    assert not final.exists()
    save_checkpoint_atomic(_toy_latest(), final, validate=_LATEST_V,
                           verbose=False)
    assert final.exists(), 'final file was not installed'
    ck = load_checkpoint(final, validate=_LATEST_V)
    assert ck['epoch'] == 3 and ck['seed'] == 42
    leftovers = [p.name for p in d.iterdir() if is_temp_name(p.name)]
    assert not leftovers, f'temp files left behind: {leftovers}'

    # best.pt has its own required-field set (val / val_acc).
    best_path = d / 'best.pt'
    save_checkpoint_atomic(_toy_best(), best_path, validate=_BEST_V,
                           verbose=False)
    load_checkpoint(best_path, validate=_BEST_V)


_check('A. atomic save / load-back / no temp residue', _test_a)


# ── B. Invalid candidate must not replace a valid final ────────────────
def _test_b():
    d = SELFTEST_ROOT / 'B'
    final = d / 'latest.pt'
    save_checkpoint_atomic(_toy_latest(epoch=3), final, validate=_LATEST_V,
                           verbose=False)
    before = final.read_bytes()

    malformed = _toy_latest(epoch=3)
    del malformed['scheduler']          # required by the resume path
    try:
        save_checkpoint_atomic(malformed, final, validate=_LATEST_V,
                               verbose=False)
    except CheckpointError:
        pass
    else:
        raise AssertionError('malformed checkpoint was accepted')

    assert final.read_bytes() == before, 'the valid final file was modified'
    load_checkpoint(final, validate=_LATEST_V)
    assert not [p for p in d.iterdir() if is_temp_name(p.name)], \
        'a rejected temp file was left behind'


_check('B. invalid candidate does not replace valid final', _test_b)


# ── C. Backup failure is non-fatal ─────────────────────────────────────
def _test_c():
    d = SELFTEST_ROOT / 'C'
    final = d / 'latest.pt'
    save_checkpoint_atomic(_toy_latest(), final, validate=_LATEST_V,
                           verbose=False)

    # A destination that cannot be created: an existing *file* used as a
    # parent directory. No real Drive path is involved.
    blocker = d / 'blocked'
    blocker.write_text('not a directory')
    ok = backup_checkpoint_to_drive(final, blocker / 'sub' / 'latest.pt')

    assert ok is False, 'a failing backup reported success'
    load_checkpoint(final, validate=_LATEST_V)   # caller can carry on
    print('    ^ the warning above is the expected, non-fatal backup failure')


_check('C. backup failure warns and does not stop the caller', _test_c)


# ── D. Restore from backup ─────────────────────────────────────────────
def _test_d():
    d = SELFTEST_ROOT / 'D'
    local = d / 'local' / 'latest.pt'
    remote = d / 'fake_drive' / 'latest.pt'

    save_checkpoint_atomic(_toy_latest(epoch=7), local, validate=_LATEST_V,
                           verbose=False)
    assert backup_checkpoint_to_drive(local, remote, verbose=False)

    # Simulate a destroyed runtime: the local tree is gone, the backup is not.
    shutil.rmtree(local.parent)
    assert not local.exists()

    ck, source = load_local_or_restore(local, remote, validate=_LATEST_V,
                                       verbose=False)
    assert source == 'drive', f'expected a Drive restore, got {source!r}'
    assert ck['epoch'] == 7
    assert local.exists(), 'restore did not install the file locally'
    load_checkpoint(local, validate=_LATEST_V)


_check('D. restore backup -> local temp -> validate -> install', _test_d)


# ── E. Provenance mismatch is rejected ─────────────────────────────────
def _test_e():
    cases = {
        'wrong seed': _toy_latest(seed=999),
        'wrong pretrain_epochs': _toy_latest(pretrain_epochs=0),
        'wrong experiment_version': {
            **_toy_latest(),
            'run_config': _toy_run_config(version='jepa_pretrain_v1'),
        },
        'pretrain checkpoint offered as supervised': {
            **_toy_latest(),
            'run_config': _toy_run_config(stage='pretrain'),
        },
    }
    for label, ck in cases.items():
        try:
            _LATEST_V(ck)
        except CheckpointError:
            continue
        raise AssertionError(f'{label} was accepted')

    # And the matching checkpoint still passes.
    _LATEST_V(_toy_latest())


_check('E. wrong seed / pretrain_epochs / version / stage rejected', _test_e)


# ── F. best.pt / latest.pt consistency ─────────────────────────────────
def _test_f():
    latest = _toy_latest(epoch=9, best_val=0.2168)
    check_best_latest_consistency(_toy_best(epoch=4, val=0.2168), latest)

    for label, best in (
        ('val mismatch', _toy_best(epoch=4, val=0.3100)),
        ('best newer than latest', _toy_best(epoch=12, val=0.2168)),
    ):
        try:
            check_best_latest_consistency(best, latest)
        except CheckpointError:
            continue
        raise AssertionError(f'{label} was accepted')


_check('F. best/latest val and epoch consistency', _test_f)

# %% tags=["checkpoint-test"]
print(f'\nCheckpoint self-tests ({SELFTEST_ROOT}):\n')
for _name, _outcome in _results:
    print(f'  [{"PASS" if _outcome == "pass" else "FAIL"}] {_name}')
    if _outcome != 'pass':
        print(f'         {_outcome}')

_n_fail = sum(1 for _, o in _results if o != 'pass')
print(f'\n{len(_results) - _n_fail}/{len(_results)} passed.')
save_json_atomic({'results': _results,
                  'experiment_version': EXPERIMENT_VERSION},
                 DIAGNOSTICS_ROOT / 'checkpoint_selftests.json')

# %% [markdown]
# ## 9. Optional: tiny-set overfit diagnostic
#
# Can the architecture memorise a very small training set at all? If
# teacher-forced *training* accuracy cannot approach ~100% on 32-128 unique
# token sequences, the problem is optimisation or capacity, not the JEPA
# objective or the data scale.
#
# Disabled by default and **never** a prerequisite for the v2 experiment. It
# trains a fresh baseline model with no pretraining and writes nothing into the
# experiment checkpoint tree.

# %% tags=["optional", "diagnostic", "long-running"]
RUN_TINY_OVERFIT = False

TINY_N = 64          # unique token sequences
TINY_EPOCHS = 200
TINY_BATCH = 16
TINY_LR = 3e-4

if not RUN_TINY_OVERFIT:
    print('RUN_TINY_OVERFIT = False — skipped. Set it to True to run.')
else:
    seed_everything(GLOBAL_SEED)

    # Distinct token sequences only: repeated sequences would make the task
    # easier than it looks.
    _seen, _idx = set(), []
    for _i, _key in enumerate(synth_train.token_keys):
        if _key in _seen:
            continue
        _seen.add(_key)
        _idx.append(_i)
        if len(_idx) >= TINY_N:
            break
    tiny_ds = Subset(synth_train, _idx)
    print(f'tiny set: {len(tiny_ds)} unique token sequences')

    tiny_model, _, _ = build_model()
    tiny_opt = torch.optim.AdamW(tiny_model.parameters(), lr=TINY_LR,
                                 weight_decay=WEIGHT_DECAY)
    _g = torch.Generator()
    _g.manual_seed(GLOBAL_SEED)
    tiny_loader = DataLoader(tiny_ds, batch_size=TINY_BATCH, shuffle=True,
                             num_workers=0, generator=_g)

    tiny_history = []
    for _ep in range(1, TINY_EPOCHS + 1):
        tiny_model.train()
        _loss_sum = 0.0
        _n = 0
        for _b in tiny_loader:
            _p = _b['points'].to(DEVICE)
            _i = _b['input_ids'].to(DEVICE)
            _m = _b['attn_mask'].to(DEVICE)
            tiny_opt.zero_grad()
            _out = tiny_model(_p, _i, attn_mask=_m)
            _out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 1.0)
            tiny_opt.step()
            _loss_sum += _out['loss'].item()
            _n += 1

        if _ep % 10 == 0 or _ep == TINY_EPOCHS:
            tiny_model.eval()
            _c = _t = _cb = _tb = 0.0
            with torch.no_grad():
                for _b in DataLoader(tiny_ds, batch_size=TINY_BATCH,
                                     shuffle=False, num_workers=0):
                    _p = _b['points'].to(DEVICE)
                    _i = _b['input_ids'].to(DEVICE)
                    _m = _b['attn_mask'].to(DEVICE)
                    _lg = tiny_model(_p, _i, attn_mask=_m)['logits']
                    _a, _bb = teacher_forced_counts(_lg, _i, tokenizer.pad_id)
                    _c += _a
                    _t += _bb
                    _a, _bb = teacher_forced_counts(_lg, _i, tokenizer.pad_id,
                                                    branch_tree=BRANCH_TREE)
                    _cb += _a
                    _tb += _bb
            _rec = {'epoch': _ep, 'loss': _loss_sum / max(_n, 1),
                    'train_token_acc': _c / max(_t, 1),
                    'train_branch_acc': _cb / max(_tb, 1)}
            tiny_history.append(_rec)
            print(f'  E{_ep:>4} | loss={_rec["loss"]:.4f} | '
                  f'train_token_acc={_rec["train_token_acc"] * 100:.1f}% | '
                  f'train_branch_acc={_rec["train_branch_acc"] * 100:.1f}%')

    save_json_atomic({'config': {'tiny_n': len(tiny_ds), 'epochs': TINY_EPOCHS,
                                 'lr': TINY_LR, 'batch': TINY_BATCH},
                      'history': tiny_history},
                     DIAGNOSTICS_ROOT / 'tiny_overfit.json')
    print(f'\nWritten to {DIAGNOSTICS_ROOT / "tiny_overfit.json"} '
          f'(experiment checkpoints untouched)')
    del tiny_model, tiny_opt, tiny_loader
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# %% [markdown]
# ## 10. Optional: read-only audit of an old checkpoint
#
# Old Drive checkpoints may have been written while Drive I/O was failing. This
# cell reports what a file contains. It is strictly read-only: nothing is
# deleted, rewritten, copied into v2, or trusted for v2 training.
#
# Put paths in `AUDIT_PATHS` to use it — it does nothing by default.

# %% tags=["optional", "diagnostic"]
AUDIT_PATHS: list[str] = [
    # e.g. '/content/drive/MyDrive/Symba/symbolic-jepa/checkpoints/'
    #      'jepa_pretrain/pretrain_v1/pre10_seed42/latest.pt',
]


def audit_checkpoint(path):
    """Report on a checkpoint without modifying it."""
    p = Path(path)
    print(f'\n{"=" * 74}\n{p}\n{"=" * 74}')
    if not p.exists():
        print('  exists            : NO')
        return
    print('  exists            : yes')
    print(f'  size              : {p.stat().st_size / 1e6:.1f} MB')
    if is_temp_name(p.name):
        print('  NOTE              : in-flight/quarantined name — never resumable')

    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'  torch.load        : FAILED ({type(e).__name__}: {e})')
        return
    print('  torch.load        : ok')

    if not isinstance(ck, dict):
        print(f'  contents          : {type(ck).__name__}, not a dict')
        return

    print(f'  keys              : {sorted(ck)}')
    for _k in ('epoch', 'seed', 'best_val', 'best_val_acc', 'val', 'val_acc'):
        if _k in ck:
            print(f'  {_k:<18}: {ck[_k]}')
    _model = ck.get('model')
    print(f'  model tensors     : '
          f'{len(_model) if isinstance(_model, dict) else "MISSING"}')

    _cfg = ck.get('run_config')
    if _cfg is None:
        print('  run_config        : MISSING (legacy file — provenance unknown)')
    else:
        print(f'  run_config.version: '
              f'{_cfg.get("experiment_version", "<absent — pre-v2 file>")}')
        print(f'  run_config        : {json.dumps(_cfg, sort_keys=True)[:300]}')

    sibling = p.with_name('best.pt' if p.name == 'latest.pt' else 'latest.pt')
    if sibling.exists():
        try:
            other = torch.load(sibling, map_location='cpu', weights_only=False)
            latest, best = ((ck, other) if p.name == 'latest.pt'
                            else (other, ck))
            check_best_latest_consistency(best, latest, path=p.parent)
            print(f'  vs {sibling.name:<14}: consistent')
        except CheckpointError as e:
            print(f'  vs {sibling.name:<14}: INCONSISTENT — {e}')
        except Exception as e:
            print(f'  vs {sibling.name:<14}: unreadable ({type(e).__name__})')

    print('\n  (read-only: nothing was deleted, rewritten, or copied into v2)')


if not AUDIT_PATHS:
    print('AUDIT_PATHS is empty — nothing audited. Add old checkpoint paths '
          'above to inspect them read-only.')
for _path in AUDIT_PATHS:
    audit_checkpoint(_path)
