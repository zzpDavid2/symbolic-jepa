# ══════════════════════════════════════════════════════════════════════════
# Bank the parsed expressions / branch tree from a LIVE session.
#
# Paste into a running notebook that has already paid the ~11 min parse. It
# writes the same cache entries jepa_pretrain_v2 will look up, so the next
# runtime loads them in ~90 s instead of re-parsing.
#
# Needs in globals: synth_exprs, tokenizer, SYNTH_PKL, MAX_SYNTH, MAX_SEQ,
#                   MAX_VARS, DEDUPE_BY_TOKENS.   synth_train is optional.
#
# It git-pulls first, on purpose: the cache key includes a hash of the library
# source, so the entry must be keyed against the code the NEXT session will
# import. Pulling only rewrites files on disk — your live kernel keeps its
# objects, so synth_exprs is safe. Restart the runtime AFTER this cell.
# ══════════════════════════════════════════════════════════════════════════
import os
import sys
from pathlib import Path

REPO_DIR = Path(globals().get('REPO_DIR', '/content/drive/MyDrive/Symba/symbolic-jepa'))
DRIVE_BASE = Path(globals().get('DRIVE_BASE', '/content/drive/MyDrive/Symba'))
BRANCH = 'main'
IN_COLAB = 'google.colab' in sys.modules

# Only the package is updated, not the notebooks. Colab writes outputs straight
# into the .ipynb on Drive, so the working tree is almost certainly dirty and a
# plain `git pull` would refuse; checking out one directory from origin works
# regardless and cannot clobber your notebooks.
os.system(f'git -C "{REPO_DIR}" fetch origin --quiet')
rc = os.system(f'git -C "{REPO_DIR}" checkout origin/{BRANCH} -- symbolic_jepa/')
os.system(f'git -C "{REPO_DIR}" log -1 --format="library now at %h %s" origin/{BRANCH}')
if rc != 0:
    raise SystemExit(
        f'Could not update symbolic_jepa/ from origin/{BRANCH} (exit {rc}).\n'
        f'Any entry written now would be keyed against the wrong library '
        f'source and would never hit. Fix the checkout at {REPO_DIR} first.'
    )

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# datacache has never been imported, so this picks up the freshly pulled file.
# Its fingerprints hash the module files on DISK, not the stale copies in this
# kernel, so the key is correct even though the old library is still loaded.
try:
    from symbolic_jepa.datacache import (
        DataCache, expressions_cache_key, prefix_tree_cache_key,
    )
except ImportError as e:
    raise SystemExit(
        f'Could not import symbolic_jepa.datacache ({e}).\n'
        f'The pull did not bring in the caching code, so any entry written now '
        f'would be keyed against the wrong library source and never hit.\n'
        f'Fix the checkout at {REPO_DIR} first (check `git status` for local '
        f'changes blocking the fast-forward), then re-run this cell.'
    )

if IN_COLAB:
    LOCAL_CACHE_ROOT = Path('/content/symbolic_jepa_cache')
    DRIVE_CACHE_ROOT = DRIVE_BASE / 'symbolic_jepa_cache'
else:
    LOCAL_CACHE_ROOT = REPO_DIR / 'local_runs' / 'symbolic_jepa_cache'
    DRIVE_CACHE_ROOT = None

cache = DataCache(LOCAL_CACHE_ROOT, DRIVE_CACHE_ROOT)
print(f'\nlocal : {LOCAL_CACHE_ROOT}\ndrive : {DRIVE_CACHE_ROOT}\n')

# ── 1. the deduplicated expressions ───────────────────────────────────────
# Every Expression that has been sampled holds a lambdify callable, which is
# built by exec under a generated name and cannot be pickled. This kernel's
# Expression class predates the __getstate__ that drops it, so clear the caches
# by hand and put them back afterwards — they rebuild lazily anyway, and this
# way the live session is left exactly as it was found.
saved_fns = [e._fn for e in synth_exprs]
for e in synth_exprs:
    e._fn = None
try:
    key = expressions_cache_key(
        SYNTH_PKL, tokenizer,
        max_seq_len=MAX_SEQ, max_vars=MAX_VARS,
        max_expressions=MAX_SYNTH, dedupe_by_tokens=DEDUPE_BY_TOKENS,
    )
    print(f'expressions key : {key}')
    cache.put('expressions', key, synth_exprs,
              label=f'{len(synth_exprs)} deduped expressions')
finally:
    for e, fn in zip(synth_exprs, saved_fns):
        e._fn = fn
    del saved_fns

# ── 2. the branch tree ────────────────────────────────────────────────────
# Keyed on the training token sequences themselves, so it is stored only if
# this session actually built the splits.
_train = globals().get('synth_train')
if _train is None:
    print('\nsynth_train not defined — skipping the branch tree. Re-run this '
          'cell after the splits exist if you want it banked too.')
else:
    _keys = _train.token_keys
    _tree = globals().get('BRANCH_TREE')
    if _tree is None:
        from symbolic_jepa import build_prefix_tree
        print('\nBRANCH_TREE not defined — building it (a few seconds)')
        _tree = build_prefix_tree(_keys)
    tkey = prefix_tree_cache_key(_keys)
    print(f'\nbranch tree key : {tkey}')
    cache.put('branch_tree', tkey, _tree,
              label=f'{len(_tree)} prefixes over {len(_keys)} train sequences')

print('\nDone. Restart the runtime, then run jepa_pretrain_v2.ipynb top to '
      'bottom — sections 3 and 4 should report "[cache] hit".')
