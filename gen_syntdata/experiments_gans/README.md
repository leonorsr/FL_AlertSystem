# GANs — geração de quedas e não-quedas

Experiência de controlo para comparar com `experiments_gans_falls`.

## Protocolo corrigido

- Base experimental fixa: todas as 259 942 linhas reais.
- Split real: 208 401 treino, 25 750 desenvolvimento e 25 791 teste.
- Cada GAN aprende exclusivamente com linhas dos clientes de treino.
- Geração separada de `label=0` e `label=1`; a proporção sintética por classe
  segue a distribuição observada no treino.
- Desenvolvimento e teste contêm apenas dados reais.
- Repartições sintético/real: 25/75, 50/50, 70/30 e 90/10. Mantêm-se sempre
  as 259 942 linhas reais; varia apenas a quantidade sintética.
- 5 GANs × 4 repartições × 3 estratégias = 60 smoke tests.

As quantidades sintéticas são 86 648, 259 942, 606 532 e 2 339 478,
respetivamente. Por omissão, a TimeGAN usa 100 épocas de embedding, 100
supervisionadas e 200 conjuntas. Os parâmetros podem ser alterados na CLI.

## Execução

```bash
venv/bin/python -m gen_syntdata.experiments_gans.prepare_smoke_datasets --force --force-models
venv/bin/python -m gen_syntdata.experiments_gans.run_smoke_grid --dry-run
venv/bin/python -m gen_syntdata.experiments_gans.run_smoke_grid
venv/bin/python -m gen_syntdata.experiments_gans.run_full_top3
```

Os novos artefactos ficam em `full_data_datasets/`,
`full_data_smoke_results/` e `full_data_full_results/`. As pastas antigas sem o
prefixo `full_data_` correspondem ao protocolo anterior de 5 000 linhas e não
devem ser usadas na análise corrigida.
