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
# # Upstream SymbolicGPT on our dataset — independent baseline
#
# One question, and nothing else:
#
# > How well does **upstream SymbolicGPT** do on **exactly our dataset**, without
# > JEPA and without any of our architectural or training modifications?
#
# So the model, the point encoder, the conditioning, the trainer, the loss, and the
# autoregressive sampler all come from `mojivalipour/symbolicgpt` at a pinned
# commit and are used unmodified. What we supply is the data and the scoring:
#
# * **data** — the same synthetic expressions, the same seed-42 train/val/test
#   partition, the same 1000-point clouds and normalisation as
#   `subsample_jepa_v2` / `jepa_pretrain_v2`;
# * **scoring** — our `evaluate_predictions` (exact match, algebraic equivalence,
#   held-out R² after BFGS constant fitting), so the numbers land on the same axis
#   as our own runs rather than on upstream's relative-error metric.
#
# The two sides meet in one place, `symbolic_jepa/symbolicgpt_adapter.py`, which
# converts our prefix token strings to the infix skeleton strings SymbolicGPT is
# character-level over, converts predictions back, and serves our point clouds in
# upstream's channel-first layout. The conversion is exactly invertible and the
# notebook asserts that over the whole dataset before training starts.
#
# **What is deliberately *not* here:** no JEPA, no auxiliary loss, no pretraining,
# no beam search, no tuned hyperparameters. Where upstream's settings had to
# change to fit our data at all, section 3 says so explicitly.

# %% [markdown]
# ## 0. Environment setup
#
# Two checkouts are involved: our repository (library code + the data pickle) and
# the upstream SymbolicGPT repository, pinned to `UPSTREAM_COMMIT`. Upstream is
# cloned to local disk — it is small and re-cloning on a fresh runtime is free.
#
# Edit `DRIVE_BASE` if your Drive layout differs.

# %%
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules

DRIVE_BASE = Path('/content/drive/MyDrive/Symba')
REPO_URL = 'https://github.com/zzpDavid2/symbolic-jepa.git'

# Upstream SymbolicGPT, pinned. Printed again after cloning so the notebook's
# output records which code actually ran.
UPSTREAM_URL = 'https://github.com/mojivalipour/symbolicgpt.git'
UPSTREAM_COMMIT = '6aef07c285d4d61f161490d42c057f2f55d045a9'


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
        print(f'[backup WARNING] Drive mount failed: {type(e).__name__}: {e}')
        print('Training can still run; artifacts stay on /content only.')

    REPO_DIR = DRIVE_BASE / 'symbolic-jepa'
    if not REPO_DIR.exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        sh(f'git clone {REPO_URL} "{REPO_DIR}"')
    else:
        sh(f'git -C "{REPO_DIR}" pull --ff-only')

    LOCAL_ROOT = Path('/content')
    sh(f'{sys.executable} -m pip install -q sympy scipy')
else:
    REPO_DIR = Path.cwd()
    LOCAL_ROOT = REPO_DIR / 'local_runs'

LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

import symbolic_jepa  # noqa: E402

print(f'\nEnvironment : {"Colab" if IN_COLAB else "local"}')
print(f'Repo        : {REPO_DIR}')
print(f'symbolic_jepa imported from: {symbolic_jepa.__file__}')
sh(f'git -C "{REPO_DIR}" rev-parse --short HEAD')

# %% [markdown]
# ### Clone upstream SymbolicGPT at the pinned commit
#
# `models.py`, `trainer.py` and `utils.py` are imported from this checkout, so
# their printed paths below are the proof that the baseline runs upstream code and
# not ours. Our repository has no top-level modules by those names, so there is
# nothing for them to collide with.
#
# **One upstream incompatibility, one shim.** `utils.py` line 11 does
# `from numpy import *`, which shadows the builtin `min`/`max` inside that module.
# `top_k_top_p_filtering` then calls `min(top_k, logits.size(-1))`, which modern
# NumPy reads as `np.min(value, axis=...)` and rejects with
# `AxisError: axis 31 is out of bounds`. Every call to upstream's sampler dies
# there. Rebinding the two names to the builtins in that module restores the
# behaviour the code was written against without touching any logic; it is the
# smallest change that makes upstream generation run at all.

# %%
UPSTREAM_DIR = LOCAL_ROOT / 'symbolicgpt'

if not (UPSTREAM_DIR / '.git').exists():
    sh(f'git clone {UPSTREAM_URL} "{UPSTREAM_DIR}"')
sh(f'git -C "{UPSTREAM_DIR}" fetch --quiet origin')
sh(f'git -C "{UPSTREAM_DIR}" checkout --quiet {UPSTREAM_COMMIT}')

UPSTREAM_SHA = subprocess.run(
    ['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'],
    capture_output=True, text=True, check=True).stdout.strip()
assert UPSTREAM_SHA == UPSTREAM_COMMIT, (
    f'upstream checkout is {UPSTREAM_SHA}, expected {UPSTREAM_COMMIT}')

sys.path.insert(0, str(UPSTREAM_DIR))

import models as upstream_models          # noqa: E402
import trainer as upstream_trainer        # noqa: E402
import utils as upstream_utils            # noqa: E402

# The `from numpy import *` shim described above.
import builtins  # noqa: E402
upstream_utils.min, upstream_utils.max = builtins.min, builtins.max

print(f'\nupstream commit : {UPSTREAM_SHA}')
print(f'models.py       : {upstream_models.__file__}')
print(f'trainer.py      : {upstream_trainer.__file__}')
print(f'utils.py        : {upstream_utils.__file__}')

# %% [markdown]
# ## 1. Imports, configuration, seed
#
# Everything that defines the experiment lives in the config cell below.
#
# * The **data** block is copied verbatim from `subsample_jepa_v2` — changing any
#   of it means this is no longer a baseline on our dataset.
# * The **upstream hyperparameter** block is upstream's own, taken from
#   `symbolicGPT.py` with `numVars` set to our single variable. Do not tune it;
#   the point of the run is what upstream's settings do here.
# * `BLOCK_SIZE = 0` means "derive from the data" (section 3). Upstream's literal
#   `blockSize = 32` cannot hold our equations and is the one hyperparameter that
#   must change.

# %%
import csv
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from symbolic_jepa import PrefixTokenizer, build_synthetic_splits
from symbolic_jepa.datacache import DataCache, cached_synthetic_expressions
from symbolic_jepa.evaluation import evaluate_predictions
from symbolic_jepa.symbolicgpt_adapter import (
    SymbolicGPTDataset, build_char_vocab, check_roundtrip, decode_prediction,
    infix_to_prefix, prefix_to_infix,
)

from models import GPT, GPTConfig, PointNetConfig          # noqa: E402
from trainer import Trainer, TrainerConfig                 # noqa: E402
from utils import sample_from_model, set_seed              # noqa: E402

# %%
EXPERIMENT_VERSION = 'symbolicgpt_baseline_v1'

# ── Reproducibility ────────────────────────────────────────────────────
# One knob. Re-run the notebook with 123, 7, ... for the 8-seed comparison;
# every artifact path below carries the seed, so runs cannot overwrite one
# another.
SEED = 42

# ── Our dataset (verbatim from subsample_jepa_v2 — do not tune) ─────────
MAX_VARS = 1                  # univariate synthetic data
N_POINTS = 1000               # points per cloud
MAX_SEQ = 64                  # prefix tokens; caps which expressions are kept
SYNTH_SEED = 42               # governs the train/val/test partition
MAX_SYNTH = 200_000           # raw strings read from the pickle
DEDUPE_BY_TOKENS = True       # one example per distinct token sequence
GROUP_BY_TOKENS = True        # no token sequence spans two splits
USE_DATA_CACHE = True         # reuse the parsed-expression cache
SYNTH_PKL = str(REPO_DIR / 'data' / 'synthetic.pkl')

# ── Upstream SymbolicGPT hyperparameters (symbolicGPT.py, numVars=1) ────
EMBEDDING_SIZE = 512
N_LAYER = 8
N_HEAD = 8
METHOD = 'EMB_SUM'            # point embedding added to token+position embedding
VARIABLE_EMBEDDING = 'NOT_VAR'
NUM_VARS = MAX_VARS           # our data is univariate
NUM_YS = 1
NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 6e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_NORM_CLIP = 1.0
LR_DECAY = True
WARMUP_TOKENS = 512 * 20
NUM_WORKERS = 0               # upstream's own default; see the note in section 2
BLOCK_SIZE = 0                # 0 = derive from our equations (section 3)

# ── Decoding ───────────────────────────────────────────────────────────
# Upstream's own sampler either way. Its script used nucleus sampling
# (temperature 1.0, top_p 0.7); greedy is the default here because our
# baselines are scored under greedy decoding and a stochastic decoder would
# make the comparison a different experiment. Set DECODE_SAMPLE = True and
# DECODE_TOP_P = 0.7 to reproduce upstream's script exactly.
DECODE_SAMPLE = False
DECODE_TEMPERATURE = 1.0
DECODE_TOP_K = 0.0
DECODE_TOP_P = 0.0
MAX_EVAL_EXAMPLES = 0         # 0 = full test set; a positive N caps the decode

N_EXAMPLES_TO_PRINT = 10

# ── Artifacts ──────────────────────────────────────────────────────────
RUN_DIR = LOCAL_ROOT / EXPERIMENT_VERSION / f'seed_{SEED}'
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = RUN_DIR / 'best.pt'
DRIVE_RUN_DIR = ((DRIVE_BASE / EXPERIMENT_VERSION / f'seed_{SEED}')
                 if (IN_COLAB and DRIVE_MOUNTED) else None)

# Upstream's Trainer takes the string 'gpu' and decides between CUDA and CPU on
# its own; DEVICE mirrors that decision so evaluation runs where training ran.
TRAINER_DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device   : {DEVICE}')
print(f'Run dir  : {RUN_DIR}')
print(f'Drive dir: {DRIVE_RUN_DIR}')

# %%
# Upstream's own seeding routine: random, numpy, torch, torch.cuda.
set_seed(SEED)
print(f'Seeded with {SEED} via upstream utils.set_seed')

# %% [markdown]
# ## 2. Our dataset
#
# What "our dataset" means, read off `subsample_jepa_v2` / `jepa_pretrain_v2`:
#
# | | |
# |---|---|
# | source | `data/synthetic.pkl` — expression strings from `SYMBA_Reg_Data_Gen` |
# | kept | first `MAX_SYNTH` strings, parsed to SymPy, one per distinct prefix token sequence, prefix length ≤ `MAX_SEQ` |
# | variables | one, `x`, sampled uniformly on [-π, π] |
# | point cloud | `N_POINTS` rows of `(x, y)`; non-finite rows dropped; per-column z-score; padded to `(N_POINTS, MAX_VARS + 1)` |
# | train clouds | resampled every `__getitem__` (augmentation) |
# | val/test clouds | deterministic, seeded by a fixed data seed + sample index — independent of the model seed |
# | split | 80/10/10 by `SYNTH_SEED`, grouped so no token sequence spans two splits |
# | expression format | prefix token strings over the 32-token vocabulary (`MAX_VARS = 1`) |
#
# `cached_synthetic_expressions` and `build_synthetic_splits` are the same calls
# the JEPA notebooks make, with the same arguments, so the partition is the same
# partition. `build_multiview_synthetic_splits` (used by `subsample_jepa_v2`)
# derives its split from the identical `_split_indices(len(exprs), SYNTH_SEED,
# 0.8, 0.1, groups)` call and applies the identical filtering, so the *examples*
# in each split match; only how many point clouds the training set draws per item
# differs (view 0 of a 4× pool there, a fresh 1000-point draw here — same
# distribution, same normalisation).
#
# The expression cache is content-addressed on the pickle, every argument that
# changes which expressions survive, the tokenizer vocabulary and the parser
# source, so this notebook hits the same entry the other notebooks built. A first
# run without a warm cache spends ~10–15 minutes parsing the pickle.
#
# Sanity check when this finishes: the split sizes and unique-sequence counts
# printed below must match what `subsample_jepa_v2` prints.

# %%
if IN_COLAB:
    LOCAL_CACHE_ROOT = Path('/content/symbolic_jepa_cache')
    DRIVE_CACHE_ROOT = (DRIVE_BASE / 'symbolic_jepa_cache') if DRIVE_MOUNTED else None
else:
    LOCAL_CACHE_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_cache'
    DRIVE_CACHE_ROOT = None

CACHE = DataCache(LOCAL_CACHE_ROOT, DRIVE_CACHE_ROOT, enabled=USE_DATA_CACHE)

tokenizer = PrefixTokenizer(max_vars=MAX_VARS)
print(f'Prefix vocab size: {len(tokenizer)}')

synth_exprs = cached_synthetic_expressions(
    SYNTH_PKL, tokenizer,
    max_seq_len=MAX_SEQ,
    max_vars=MAX_VARS,
    max_expressions=MAX_SYNTH,
    dedupe_by_tokens=DEDUPE_BY_TOKENS,
    cache=CACHE,
    progress=True,
)

# %%
synth_train, synth_val, synth_test = build_synthetic_splits(
    synth_exprs, tokenizer,
    n_points=N_POINTS, max_seq_len=MAX_SEQ, max_vars=MAX_VARS,
    seed=SYNTH_SEED,
    cache_eval=True,
    group_by_tokens=GROUP_BY_TOKENS,
    progress=True,
)

print(f'\ntrain / val / test : '
      f'{len(synth_train)} / {len(synth_val)} / {len(synth_test)}')

# %% [markdown]
# ## 3. Dataset adapter
#
# SymbolicGPT is character-level over an infix skeleton string such as
# `((C*x1)+sin(x1))`, where `C` marks a fittable constant, `<` is SOS, `>` is EOS
# and `_` is padding. Our expressions are prefix token sequences. The adapter
# converts between the two with `prefix_to_infix` / `infix_to_prefix`:
#
# * every binary application is emitted fully parenthesised, so the string parses
#   back without precedence rules;
# * structural numerics keep their identity (`neg2` → `(-2)`, `half` → `0.5`),
#   which matters because `C` is not identifiable — collapsing `x1**3` and
#   `x1**4` onto `x1**C` would make them indistinguishable to the decoder.
#
# The cell below asserts the round trip over **every** expression in all three
# splits. If it ever fails, the baseline's numbers are not comparable and the run
# should stop there.
#
# Three upstream properties are preserved verbatim by the adapter: the character
# vocabulary recipe (`sorted(set(text) + ['_','T','<','>',':'])`), the
# `<eq>` / next-token-shift / right-pad encoding, and the channel-first
# `(numVars + numYs, numPoints)` point layout that `tNet` consumes.
#
# **`blockSize` is the one upstream hyperparameter that has to change.** Upstream
# uses 32, sized for its own short equations; ours reach far past that as
# character strings, and truncating them would silently delete the tail of every
# long target. `BLOCK_SIZE = 0` therefore derives it as the longest
# `'<' + skeleton + '>'` across all splits — the smallest value that never
# truncates.
#
# **Why `num_workers = 0`.** Our training set resamples its point cloud on every
# `__getitem__` using the global NumPy RNG. Forked DataLoader workers would
# inherit identical RNG state and hand back identical "augmented" clouds. Zero is
# also upstream's own setting in `symbolicGPT.py`, so nothing is being traded
# away.

# %%
gt_prefix = {
    'train': [tokenizer.decode(s['input_ids'].tolist()) for s in synth_train.samples],
    'val':   [tokenizer.decode(s['input_ids'].tolist()) for s in synth_val.samples],
    'test':  [tokenizer.decode(s['input_ids'].tolist()) for s in synth_test.samples],
}
all_prefix = [p for split in gt_prefix.values() for p in split]

roundtrip = check_roundtrip(all_prefix)
print(f'prefix -> infix -> prefix : {roundtrip["n_checked"]} checked, '
      f'{roundtrip["n_failed"]} failed')
for bad in roundtrip['failures']:
    print(f'  FAILURE {bad}')
assert roundtrip['n_failed'] == 0, (
    'prefix <-> infix conversion is not exact on this dataset; '
    'predictions could not be scored against our evaluator'
)

all_equations = [prefix_to_infix(p) for p in all_prefix]
chars = build_char_vocab(all_equations)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
PADDING_ID = stoi['_']

longest = max(len(eq) + 2 for eq in all_equations)
BLOCK_SIZE = BLOCK_SIZE if BLOCK_SIZE > 0 else longest

print(f'\ncharacter vocab ({len(chars)}): {"".join(chars)}')
print(f'longest "<skeleton>"     : {longest} characters')
print(f'block_size in use        : {BLOCK_SIZE} '
      f'(upstream default is 32 — too short for our equations)')
assert BLOCK_SIZE >= longest, (
    f'BLOCK_SIZE={BLOCK_SIZE} truncates equations up to {longest} characters')

# %%
sgpt_train = SymbolicGPTDataset(synth_train, tokenizer, stoi, BLOCK_SIZE)
sgpt_val = SymbolicGPTDataset(synth_val, tokenizer, stoi, BLOCK_SIZE)
sgpt_test = SymbolicGPTDataset(synth_test, tokenizer, stoi, BLOCK_SIZE)

print(f'adapted train / val / test : '
      f'{len(sgpt_train)} / {len(sgpt_val)} / {len(sgpt_test)}')

# %% [markdown]
# ## 4. Sanity check — a few `(points, expression)` pairs
#
# Read this carefully before training. It shows what the model actually receives:
# the point tensor's shape and range, the skeleton string, and the encoded
# input/output pair decoded back to characters (they must differ by exactly the
# one-position shift).

# %%
for split_name, ds in (('train', sgpt_train), ('test', sgpt_test)):
    print(f'\n=== {split_name} ===')
    for i in range(3):
        inputs, outputs, points, num_vars = ds[i]
        text_in = ''.join(itos[int(t)] for t in inputs).rstrip('_')
        text_out = ''.join(itos[int(t)] for t in outputs).rstrip('_')
        print(f'\n[{i}] prefix   : {ds.prefix_strings[i]}')
        print(f'    skeleton : {ds.equation_strings[i]}')
        print(f'    inputs   : {text_in[:100]}')
        print(f'    outputs  : {text_out[:100]}')
        print(f'    numVars  : {int(num_vars)}')
        print(f'    points   : shape {tuple(points.shape)} '
              f'(numVars+numYs, numPoints), '
              f'x in [{points[0].min():.2f}, {points[0].max():.2f}], '
              f'y in [{points[1].min():.2f}, {points[1].max():.2f}]')
        assert inputs[1:len(ds.equation_strings[i]) + 1].tolist() == \
            outputs[:len(ds.equation_strings[i])].tolist()

# %% [markdown]
# ## 5. Construct upstream SymbolicGPT
#
# `PointNetConfig` + `GPTConfig` + `GPT` straight from upstream `models.py`. The
# point encoder is upstream's `tNet` (three `Conv1d` layers, global max pool, two
# dense layers) — `GPT.__init__` selects it, we do not. With `EMB_SUM` the point
# embedding is broadcast over the sequence and added to the token and position
# embeddings.

# %%
pconf = PointNetConfig(
    embeddingSize=EMBEDDING_SIZE,
    numberofPoints=N_POINTS,
    numberofVars=NUM_VARS,
    numberofYs=NUM_YS,
    method=METHOD,
    variableEmbedding=VARIABLE_EMBEDDING,
)
mconf = GPTConfig(
    len(chars), BLOCK_SIZE,
    n_layer=N_LAYER, n_head=N_HEAD, n_embd=EMBEDDING_SIZE,
    padding_idx=PADDING_ID,
)
model = GPT(mconf, pconf)

n_params = sum(p.numel() for p in model.parameters())

CONFIG = {
    'experiment_version': EXPERIMENT_VERSION,
    'seed': SEED,
    'upstream_repo': UPSTREAM_URL,
    'upstream_commit': UPSTREAM_SHA,
    'data': {
        'source': SYNTH_PKL,
        'max_vars': MAX_VARS, 'n_points': N_POINTS, 'max_seq_len': MAX_SEQ,
        'synth_seed': SYNTH_SEED, 'max_synth': MAX_SYNTH,
        'dedupe_by_tokens': DEDUPE_BY_TOKENS,
        'group_by_tokens': GROUP_BY_TOKENS,
        'n_train': len(sgpt_train), 'n_val': len(sgpt_val),
        'n_test': len(sgpt_test),
    },
    'representation': {
        'tokenization': 'character-level over infix skeleton',
        'char_vocab_size': len(chars),
        'block_size': BLOCK_SIZE,
        'block_size_source': 'derived from data (upstream default 32 truncates)',
        'padding_token': '_', 'sos': '<', 'eos': '>',
    },
    'model': {
        'point_encoder': 'tNet (upstream models.py)',
        'method': METHOD, 'variable_embedding': VARIABLE_EMBEDDING,
        'n_layer': N_LAYER, 'n_head': N_HEAD, 'n_embd': EMBEDDING_SIZE,
        'num_vars': NUM_VARS, 'num_ys': NUM_YS,
        'n_parameters': n_params,
    },
    'training': {
        'objective': 'next-character cross-entropy (upstream GPT.forward)',
        'trainer': 'upstream trainer.Trainer',
        'max_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE, 'betas': list(BETAS),
        'weight_decay': WEIGHT_DECAY, 'grad_norm_clip': GRAD_NORM_CLIP,
        'lr_decay': LR_DECAY, 'warmup_tokens': WARMUP_TOKENS,
        'final_tokens': 2 * len(sgpt_train) * BLOCK_SIZE,
        'num_workers': NUM_WORKERS,
        'model_selection': 'lowest val loss (upstream Trainer)',
    },
    'decoding': {
        'sampler': 'upstream utils.sample_from_model',
        'sample': DECODE_SAMPLE, 'temperature': DECODE_TEMPERATURE,
        'top_k': DECODE_TOP_K, 'top_p': DECODE_TOP_P,
        'steps': BLOCK_SIZE,
    },
    'evaluation': {
        'scorer': 'symbolic_jepa.evaluation.evaluate_predictions',
        'max_eval_examples': MAX_EVAL_EXAMPLES,
    },
    'modifications_to_upstream': [
        'block_size raised from 32 to the longest equation in our data; '
        'the upstream value truncates our targets',
        'utils.min / utils.max rebound to the Python builtins, which '
        '`from numpy import *` had shadowed, so top_k_top_p_filtering runs',
    ],
}

print(json.dumps(CONFIG, indent=2))
print(f'\nparameters: {n_params / 1e6:.1f}M')

# %% [markdown]
# ## 6. Train
#
# The next cell runs the **full training job** — `NUM_EPOCHS` epochs over the
# whole training split, on GPU. Expect hours, not minutes; the per-epoch cost is
# dominated by resampling a 1000-point cloud per example.
#
# `trainer.Trainer` handles everything: AdamW with upstream's decay/no-decay
# parameter split, linear warmup into cosine decay, gradient clipping, and
# writing `best.pt` whenever validation loss improves. Interrupting the cell
# leaves the best checkpoint so far on disk.
#
# If this runs out of GPU memory: upstream's `batchSize = 128` was sized for 249
# points and 32-character contexts, and ours are 1000 points and a context several
# times longer, so the `tNet` activations are the likely culprit. Lowering
# `BATCH_SIZE` is then the smallest defensible deviation — it is part of the
# printed and saved config, so a reduced value stays visible in the record.

# %% tags=["long-running"]
tconf = TrainerConfig(
    max_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
    grad_norm_clip=GRAD_NORM_CLIP,
    lr_decay=LR_DECAY,
    warmup_tokens=WARMUP_TOKENS,
    final_tokens=2 * len(sgpt_train) * BLOCK_SIZE,
    num_workers=NUM_WORKERS,
    ckpt_path=str(CKPT_PATH),
)
trainer = Trainer(model, sgpt_train, sgpt_val, tconf, None, device=TRAINER_DEVICE)

t_start = time.time()
try:
    trainer.train()
except KeyboardInterrupt:
    print('KeyboardInterrupt — keeping the best checkpoint written so far')
train_seconds = time.time() - t_start

best_val_loss = float(trainer.best_loss)
print(f'\ntraining time  : {train_seconds / 60:.1f} min')
print(f'best val loss  : {best_val_loss:.4f}')
print(f'checkpoint     : {CKPT_PATH}')

# %% [markdown]
# ## 7. Load the best checkpoint
#
# Same two lines as upstream's script. Everything below scores *this* file, not
# whatever weights training happened to end on.

# %%
if not CKPT_PATH.exists():
    raise FileNotFoundError(
        f'{CKPT_PATH} does not exist — training never completed an epoch, so '
        f'there is nothing to evaluate. Re-run section 6.')

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model = model.eval().to(DEVICE)
print(f'loaded {CKPT_PATH}')

# %% [markdown]
# ## 8. Evaluation
#
# Two passes:
#
# 1. **Teacher-forced validation** — deterministic re-run of the validation loss
#    plus character-level accuracy at non-padding positions. Note this is a
#    *character* accuracy, so it is not on the same scale as the prefix-token
#    `val_acc` our own runs report, and it has no branching-position variant
#    (character prefixes do not correspond to symbolic decision points).
# 2. **Autoregressive decoding on the test split**, scored by our
#    `evaluate_predictions`.
#
# Decoding is the expensive half: upstream's sampler takes one example at a time
# and always runs the full `BLOCK_SIZE` steps with no early stop on `>`. Both are
# upstream properties and are left alone. Set `MAX_EVAL_EXAMPLES` to a few hundred
# for a quick read before committing to the whole test split.

# %%
@torch.no_grad()
def teacher_forced_validation(model, dataset, batch_size, device, pad_id):
    """Deterministic val loss + character accuracy at non-padding positions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    correct = total = 0
    loss_sum, n_batches = 0.0, 0
    for x, y, p, v in tqdm(loader, desc='val (teacher forced)', leave=False):
        x, y, p, v = x.to(device), y.to(device), p.to(device), v.to(device)
        logits, loss = model(x, y, p, v)
        mask = y != pad_id
        correct += int(((logits.argmax(-1) == y) & mask).sum())
        total += int(mask.sum())
        loss_sum += float(loss)
        n_batches += 1
    return loss_sum / max(n_batches, 1), correct / max(total, 1)


val_loss, val_char_acc = teacher_forced_validation(
    model, sgpt_val, BATCH_SIZE, DEVICE, PADDING_ID)
print(f'val loss (recomputed, deterministic) : {val_loss:.4f}')
print(f'val character accuracy               : {val_char_acc:.4f}')

# %% tags=["long-running"]
n_eval = len(sgpt_test) if MAX_EVAL_EXAMPLES == 0 else min(MAX_EVAL_EXAMPLES,
                                                           len(sgpt_test))
test_loader = DataLoader(sgpt_test, batch_size=1, shuffle=False, num_workers=0)

predictions = []          # (gt_prefix, pred_prefix) for our evaluator
predicted_text = []       # the raw character strings, kept for inspection
parse_errors = []

t_start = time.time()
for i, (x, y, p, v) in enumerate(tqdm(test_loader, total=n_eval, desc='decode')):
    if i >= n_eval:
        break
    out = sample_from_model(
        model,
        x[:, 0:1].to(DEVICE),          # the '<' SOS character, as upstream does
        BLOCK_SIZE,
        points=p.to(DEVICE),
        variables=v.to(DEVICE),
        temperature=DECODE_TEMPERATURE,
        sample=DECODE_SAMPLE,
        top_k=DECODE_TOP_K,
        top_p=DECODE_TOP_P,
    )[0]

    text = decode_prediction(out, itos)
    predicted_text.append(text)
    try:
        pred_prefix = infix_to_prefix(text)
    except ValueError as exc:
        # Not recoverable and not meant to be: an unparseable prediction is a
        # real failure and our evaluator counts it as one.
        pred_prefix = ''
        parse_errors.append((i, str(exc)))
    predictions.append((sgpt_test.prefix_strings[i], pred_prefix))

decode_seconds = time.time() - t_start
print(f'\ndecoded {len(predictions)} test examples in '
      f'{decode_seconds / 60:.1f} min')
print(f'unparseable predictions: {len(parse_errors)}/{len(predictions)}')

# %%
results = evaluate_predictions(predictions, synth_test, tokenizer)

# %% [markdown]
# ## 9. Summary
#
# `exact` is string-identical prefix recovery; `equiv` additionally counts
# algebraically equivalent forms (SymPy expansion, constant permutation, or
# held-out R² ≥ 0.999 after BFGS constant fitting); `parseable` is the share of
# predictions that are well-formed expressions at all. For a character-level
# decoder that last number is a real failure mode — unbalanced parentheses are
# enough to lose an otherwise correct equation.

# %%
metrics = {
    **CONFIG,
    'train_seconds': train_seconds,
    'decode_seconds': decode_seconds,
    'best_val_loss': best_val_loss,
    'val_loss': val_loss,
    'val_char_acc': val_char_acc,
    'exact': float(results['exact_match']),
    'equiv': float(results['algebraic_equiv']),
    'token_accuracy': float(results['token_accuracy']),
    'r2_above_0.9': float(results['r2_above_0.9']),
    'mean_r2': float(results['mean_r2']),
    'median_r2': float(results['median_r2']),
    'parseable': results['n_parseable'] / max(results['n_total'], 1),
    'n_parseable': results['n_parseable'],
    'n_r2_computed': results['n_r2_computed'],
    'r2_status_counts': results['r2_status_counts'],
    'n_total': results['n_total'],
}

print(f'=== {EXPERIMENT_VERSION} (upstream {UPSTREAM_SHA[:8]}) ===')
print(f'seed             : {metrics["seed"]}')
print(f'val_loss         : {metrics["val_loss"]:.4f} '
      f'(best during training {metrics["best_val_loss"]:.4f})')
print(f'val_acc (char)   : {metrics["val_char_acc"]:.4f}')
print(f'token accuracy   : {metrics["token_accuracy"]:.4f}   (prefix tokens)')
print(f'exact            : {metrics["exact"]:.4f}')
print(f'equiv            : {metrics["equiv"]:.4f}')
print(f'R2 > 0.9         : {metrics["r2_above_0.9"]:.4f}')
print(f'parseable %      : {100 * metrics["parseable"]:.1f}%   '
      f'({metrics["n_parseable"]}/{metrics["n_total"]})')
print(f'mean / median R2 : {metrics["mean_r2"]:.4f} / {metrics["median_r2"]:.4f}')
print(f'R2 status counts : {metrics["r2_status_counts"]}')

# %%
details = results['details']
print(f'\n=== {min(N_EXAMPLES_TO_PRINT, len(details))} example predictions ===')
for i, d in enumerate(details[:N_EXAMPLES_TO_PRINT]):
    r2 = 'N/A' if d['r2'] is None else f'{d["r2"]:.4f}'
    print(f'\n[{i}]')
    print(f'GT:        {d["gt"]}')
    print(f'GT (infix):{sgpt_test.equation_strings[i]}')
    print(f'Pred:      {d["pred"] if d["pred"] else "<unparseable>"}')
    print(f'Pred (raw):{predicted_text[i]}')
    print(f'parseable: {d["parseable"]}')
    print(f'exact:     {bool(d["exact"])}')
    print(f'equiv:     {bool(d["equiv"])}')
    print(f'R2:        {r2}   ({d["r2_status"]})')

# %% [markdown]
# ### Save artifacts
#
# `metrics.json` carries the full per-example detail and the config; `metrics.csv`
# is the one-row summary to concatenate across seeds. When Drive is mounted, both
# and the checkpoint are copied into `DRIVE_RUN_DIR`.

# %%
with open(RUN_DIR / 'metrics.json', 'w') as f:
    json.dump({**metrics, 'details': details,
               'predicted_text': predicted_text}, f, indent=2)

SUMMARY_FIELDS = [
    'experiment_version', 'seed', 'upstream_commit', 'val_loss',
    'best_val_loss', 'val_char_acc', 'token_accuracy', 'exact', 'equiv',
    'r2_above_0.9', 'parseable', 'mean_r2', 'median_r2', 'n_total',
]
with open(RUN_DIR / 'metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    writer.writerow({k: metrics[k] for k in SUMMARY_FIELDS})

with open(RUN_DIR / 'config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

print(f'wrote {RUN_DIR}/metrics.json, metrics.csv, config.json')

if DRIVE_RUN_DIR is not None:
    import shutil

    DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.json', 'metrics.csv', 'config.json', 'best.pt'):
        src = RUN_DIR / name
        if src.exists():
            shutil.copy2(src, DRIVE_RUN_DIR / name)
    print(f'backed up to {DRIVE_RUN_DIR}')
