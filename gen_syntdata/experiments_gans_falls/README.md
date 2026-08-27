# GANs — apenas quedas sintéticas

Experiência em que todas as amostras sintéticas têm `label=1`. Os dados reais
continuam a conter ambas as classes.

## Protocolo corrigido

- Base experimental fixa: todas as 259 942 linhas reais.
- Split real: 208 401 treino, 25 750 desenvolvimento e 25 791 teste.
- Todas as GANs, incluindo a TimeGAN própria desta experiência, aprendem apenas
  com quedas pertencentes aos clientes de treino.
- Desenvolvimento e teste contêm apenas dados reais.
- Repartições sintético/real: 25/75, 50/50, 70/30 e 90/10; são acrescentadas
  86 648, 259 942, 606 532 e 2 339 478 quedas sintéticas.
- 5 GANs × 4 repartições × 3 estratégias = 60 smoke tests.

## Execução

```bash
venv/bin/python -m gen_syntdata.experiments_gans_falls.prepare_smoke_datasets --force --force-model
venv/bin/python -m gen_syntdata.experiments_gans_falls.run_smoke_grid --dry-run
venv/bin/python -m gen_syntdata.experiments_gans_falls.run_smoke_grid
```

O relatório `full_data_smoke_results/best_partitions_by_model.txt` contém o top
10 e os comandos dos três melhores treinos completos. Os novos artefactos usam
as pastas `full_data_*`; os resultados antigos sem esse prefixo são legado do
protocolo de 5 000 linhas.
