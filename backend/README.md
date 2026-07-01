# Backend alert simulation

This backend pipeline runs the selected FL strategy for the alerting system:

- Strategy: soft-label communication without weight sharing
- Local-first variant: keep the best local model
- Client split: the same client-disjoint train/dev/test split used in the FL experiments

## Run the simulation

From the repository root:

```bash
python backend/run_alert_simulation.py
```

For a quick smoke test:

```bash
python backend/run_alert_simulation.py --smoke-test
```

Artifacts are written under [backend/results](backend/results).
