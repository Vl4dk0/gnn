# loopy-conv-index-off-by-one

**Súbor:** `ai/models/loopy.py:148, 186`  
**Závažnosť:** MEDIUM

## Popis

`LoopyLayer` pri `shared=False` vytvára `max(r, 1)` konvolúcií (indexy 0..r-1), ale forward mapouje `conv_idx = min(L, len(self.path_convs) - 1)`. Pre `r=3`:
- `L=1 → idx=1` (preskočí idx=0)
- `L=2 → idx=2`
- `L=3 → idx=min(3,2)=2` (L=2 a L=3 zdieľajú tú istú konvolúciu)

`path_convs[0]` sa nikdy nepoužije — zbytočné parametre. `L=r` a `L=r-1` zdieľajú váhy namiesto oddelených.

```python
# loopy.py:148 — chybné:
num_convs = 1 if shared else max(r, 1)

# loopy.py:186 — chybné:
conv_idx = min(L, len(self.path_convs) - 1)
# Pre L≥1 malo byť: conv_idx = L - 1
```

## Dôsledok

Model má menej trénovateľných parametrov ako zamýšľané (jedna konvolúcia nevyužitá), a L=r a L=r-1 nevhodne zdieľajú váhy — degeneruje expresivitu Loopy GNN.

## Oprava

```python
# __init__:
num_convs = 1 if shared else r   # r konvolúcií pre L=1..r

# forward:
conv_idx = 0 if shared else (L - 1)   # L=1→0, L=2→1, ..., L=r→r-1
```


## Test

Proven by [`test_loopy_layer_has_distinct_conv_per_path_length`](../../tests/test_issues_ai.py) (test fails → bug confirmed).
