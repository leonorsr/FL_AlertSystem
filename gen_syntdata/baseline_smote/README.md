# Baseline SMOTE

Baseline não generativa comparável com as duas experiências GAN.

## Protocolo corrigido

- Base experimental fixa: todas as 259 942 linhas reais.
- Split real: 208 401 treino, 25 750 desenvolvimento e 25 791 teste.
- O SMOTE é ajustado exclusivamente às quedas dos clientes de treino.
- Imputação pela mediana, normalização standard, `k=5` e interpolação
  `x_new = x_i + lambda * (x_neighbor - x_i)`.
- Desenvolvimento e teste contêm apenas dados reais.
- Repartições 25/75, 50/50, 70/30 e 90/10: as 259 942 linhas reais ficam
  fixas e são acrescentadas 86 648, 259 942, 606 532 e 2 339 478 linhas SMOTE.
- 4 repartições × 3 estratégias = 12 smoke tests.

## Execução

```bash
venv/bin/python -m gen_syntdata.baseline_smote.prepare_datasets --force
venv/bin/python -m gen_syntdata.baseline_smote.run_smoke_grid --dry-run
venv/bin/python -m gen_syntdata.baseline_smote.run_smoke_grid
```

Os novos artefactos ficam em `full_data_datasets/` e
`full_data_smoke_results/`. As pastas antigas correspondem ao protocolo de
5 000 linhas e não devem entrar na comparação corrigida.
