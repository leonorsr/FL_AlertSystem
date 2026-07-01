# Advanced class-prototype experiments

These experiments try to improve the no-weight class-prototype approach while
keeping the main objective local-first.

Strategies:

- `multi_proto_k2`: sends two prototypes per class and aligns each local
  embedding with the nearest same-class prototype.
- `cosine_margin_proto`: sends one prototype per class, normalizes embeddings,
  and applies a cosine margin loss against the opposite class prototype.
- `warmup_proto_schedule`: sends one prototype per class, disables prototype
  alignment for the first five rounds, then gradually increases the prototype
  alignment weight.

Run all strategies:

```powershell
python .\class-prototype-improvement-experiments\run_all.py --runs 10
```

Generate the results TXT:

```powershell
python .\class-prototype-improvement-experiments\generate_summary.py
```

Smoke test:

```powershell
python .\class-prototype-improvement-experiments\run_all.py --runs 1 --num-rounds 2 --local-epochs 1 --results-dir .\class-prototype-improvement-experiments\_smoke_results
```
