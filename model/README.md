# Sahayak — the model directory

Everything for the **risk classifier** (the pipeline's differentiator).

## Layout

```
model/
├── prepare_dataset.py      ← builds the labeled dataset (stdlib only)
├── data/
│   ├── train.jsonl         ← 80% — one {"text", "label"} per line
│   └── val.jsonl           ← 20% — stratified, same class mix as train
└── artifacts/              ← trained LoRA adapter + metrics (gitignored)
```

## Label schema (3 classes — DIRECTION of favor)

| label | meaning | asymmetry mapping (Day 4) |
|---|---|---|
| `favors_them` | clause disproportionately protects the OTHER party | negative score |
| `balanced` | roughly even obligations | ~0 |
| `favors_you` | clause disproportionately protects OUR reader | positive score |

Score = direction × model confidence × 100 → the −100…+100 asymmetry scale.

## Provenance (honesty matters)

`prepare_dataset.py` ships a hand-curated **seed set** (~72 clauses of
realistic Indian freelance / employment / gig-platform boilerplate,
~24 per class). This is a *demo-grade* dataset — deliberate trade-off of
the 7-day lock-in. Documented upgrade path: the **CUAD** dataset (Atticus
Project, 510 commercial contracts, 41 clause types) mapped onto this
3-class scheme.

**Label quality is the ceiling on classifier quality** — the seed set is
small but clean, balanced, and stratified-split with a fixed seed (42),
so the split is reproducible.

## Rebuild the dataset

```bash
python model/prepare_dataset.py
```

Prints class counts per split and asserts no class was lost — a split
that silently drops a class makes validation metrics lie.
