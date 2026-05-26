# gin-relu-output-bottleneck

**Súbor:** `ai/models/gin.py:65-69`  
**Závažnosť:** MEDIUM

## Popis

`_make_gin_conv` vytvorí dvojvrstvové MLP pre všetky konvolúcie vrátane výstupnej. Pre výstupnú vrstvu je `out_dim=output_dim=1`, čo dáva `Linear(hidden_dim→1) → ReLU → Linear(1→1)`. Ak výstup prvého lineárneho obsahuje záporné hodnoty, ReLU ich oreže na 0 — finálny `Linear(1,1)` potom ignoruje všetky features a predikuje len konštantný bias.

```python
# gin.py:65 — chybné pre výstupnú vrstvu:
mlp = nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.ReLU(),           # bottleneck: oreže záporné hodnoty pred výstupom
    nn.Linear(out_dim, out_dim),
)
```

## Dôsledok

Výstupná vrstva GIN má výrazne obmedzenú expresivitu — môže predikovať len nezáporné hodnoty, čo nie je zámer pre regresnú hlavu. Degraduje trénovaciu schopnosť modelu.

## Oprava

```python
def _make_gin_conv(self, in_dim: int, out_dim: int, *, is_output: bool = False) -> GINConv:
    if is_output:
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
    else:
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    return GINConv(mlp, train_eps=True)
```
