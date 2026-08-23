# Task: Update Subsample-JEPA to Use the Current Canonical + Dynamic-Constant Pipeline

Create a new subsample-JEPA experiment based on the **current audited constant-augmentation code**, not the older fixed-expression pipeline.

Do not modify `subsample_jepa_v2` in place.

Suggested name:

```text
subsample_jepa_constant_aug_v1.ipynb
```

## Goal

The existing `subsample_jepa_v2` still uses:

- fixed coefficient realizations
- old expression-based dataset loading
- 4 Transformer layers

We want a clean comparison against the current `jepa_constant_aug_v1` experiment.

The new experiment must use:

```text
canonical symbolic template
    ↓
sample dynamic coefficients
    ↓
instantiate one numerical function
    ↓
sample one shared point pool
   ↙                  ↘
subsample view A   subsample view B
```

Both views must come from the **same instantiated function / same sampled coefficients**.

## Match `jepa_constant_aug_v1`

Use `jepa_constant_aug_v1` as the source of truth for all shared settings.

Keep identical:

- canonical template file / representation
- canonical train/val/test split
- no canonical-form leakage
- dynamic coefficient augmentation
- constant sampler
- deterministic augmentation logic
- train-only fitting of coefficient distribution
- deterministic val/test
- template fingerprinting
- checkpoint/resume behavior
- tokenizer
- 8-layer Transformer architecture
- model dimensions / heads / FFN
- supervised training epochs
- optimizer / LR schedule
- evaluation code
- test set
- metrics
- seeds

The intended major difference should be **only the JEPA pretraining target/objective**.

## Dataset

Do **not** use:

```python
cached_synthetic_expressions(...)
build_multiview_synthetic_splits(...)
```

or any other old fixed-expression path.

Use the canonical-template pipeline and the existing dynamic multi-view implementation, preferably:

```python
DynamicConstantMultiViewDataset
build_multiview_template_splits(...)
```

or the current equivalent already implemented in `symbolic_jepa/templates.py`.

Do not duplicate this logic unnecessarily.

## Multi-view generation

For each training item / epoch:

1. load canonical template `S`
2. deterministically sample coefficients `theta`
3. instantiate `f_(S,theta)`
4. sample one larger point pool from that function
5. generate two subsampled views from that shared pool

Conceptually:

```text
S
↓
theta
↓
f
↓
shared point pool P
├── view A
└── view B
```

Do **not** independently resample coefficients between view A and view B.

The JEPA objective should therefore learn invariance to point subsampling, not accidentally compare different functions.

## Canonical split / leakage

Canonical symbolic form must remain the split unit.

Assert:

```text
train_forms ∩ val_forms = ∅
train_forms ∩ test_forms = ∅
val_forms ∩ test_forms = ∅
```

Dynamic coefficient realizations must never alter split membership.

## Deterministic augmentation

Use the current stateless augmentation scheme.

Randomness should be deterministically keyed by things such as:

```text
data_seed
stage
epoch
sample_idx
stream
attempt
view_idx
```

Use separate deterministic streams for:

- coefficient sampling
- point-pool sampling
- subsample view A
- subsample view B

Do not depend on mutable worker RNG state.

Resume behavior must remain identical to uninterrupted training.

## Seed handling

Prefer to incorporate the new **decoupled seed design** if it is already implemented by the time this notebook is updated:

```python
MODEL_SEED
DATA_SEED
```

`MODEL_SEED` should control:

- initialization
- dropout
- optimization-side randomness
- shuffle order if appropriate

`DATA_SEED` should control:

- coefficients
- point pool
- view subsampling

For a fixed `(MODEL_SEED, DATA_SEED)`, the 0-pretrain and 10-pretrain arms must receive exactly the same Stage-2 supervised data trajectory.

If the decoupled-seed work is not yet merged, structure the notebook so it is easy to adopt without changing the experiment logic.

## Subsample JEPA objective

Preserve the current subsample-JEPA idea unless a bug is found.

For two views:

```text
z_a = encoder(view_a)
z_b = encoder(view_b)
```

Use the current centered cosine objective / stop-gradient behavior.

If the current experiment uses:

```python
stopgrad(z_b)
```

preserve that.

Pretraining should update the **encoder only**.

Construct the pretraining optimizer explicitly over:

```python
model.encoder.parameters()
```

rather than all model parameters.

The decoder should remain untouched during Stage 1.

## Stage 2

After subsample-JEPA pretraining:

- transition to the exact same supervised training procedure as `jepa_constant_aug_v1`
- use ordinary CE only
- reset/rebuild optimizer appropriately
- use the same canonical dynamic-constant training dataset
- use the same deterministic Stage-2 augmentation trajectory for the paired 0/10 conditions

The Stage-2 architecture/training path should be identical between:

```text
subsample pretrain = 0
subsample pretrain = 10
```

except for encoder initialization.

## Experiment

Initially run:

```python
PRETRAIN_EPOCHS = [0, 10]
MODEL_SEEDS = [42, 123, 7]
```

and use the same fixed `DATA_SEED` policy as the corresponding symbolic-JEPA experiment.

If the result is promising, expand to the full 8 model seeds.

Report:

```text
model_seed
data_seed
pre_ep
val_loss
val_acc
val_branch
exact
equiv
R²>.9
```

and paired deltas:

```text
Δequiv
ΔR²>.9
```

## Diagnostics

Before training, show several examples proving:

```text
canonical template: same
sampled coefficients: same across views
point pool: shared
view A: different subset
view B: different subset
target tokens: same
```

Also verify:

```text
view_A != view_B
```

for normal examples, while both evaluate the same instantiated function.

## Tests

Add or update tests for:

1. canonical split has zero overlap;
2. coefficient sampler uses train forms only;
3. same `(data_seed, epoch, idx)` gives identical coefficients;
4. view A and B share coefficients;
5. view A and B differ only by subsampling;
6. changing epoch changes augmentation;
7. worker-count independence;
8. resume equivalence;
9. val/test remain deterministic;
10. decoder parameters do not change during Stage-1 subsample pretraining;
11. Stage-2 0ep/10ep paired data fingerprints match.

## Comparison goal

The final experiment should make this comparison valid:

```text
Symbolic-target JEPA:
numeric encoder → symbolic representation

vs.

Subsample JEPA:
numeric view A → numeric view B
```

while holding constant:

```text
dataset
canonical split
coefficient augmentation
model size
training recipe
evaluation
randomness policy
```

Do not compare against results from the old `subsample_jepa_v2` as though they were directly equivalent.

## Important

The purpose of this task is not to redesign subsample JEPA.

It is to **port the existing subsample-JEPA objective onto the final canonical-template + dynamic-coefficient + deterministic-augmentation experimental pipeline**, so that any performance difference can actually be attributed to the pretraining objective.