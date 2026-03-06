**Sweeps do not respect the objective sparsity coeff.**

See below:

```sh
uv run scripts/launch.py train --sweep contrib/trait_discovery/sweeps/train_saes_vitb.py sae.activation:relu objective:matryoshka
```

```py
(Pdb) set([cfg.objective for cfg in cfgs])
{Matryoshka(sparsity_coeff=0.0004, n_prefixes=10)}
```
