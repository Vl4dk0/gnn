# pgl2-keyerror

**Súbor:** `ai/cage/voltage/groups.py:237-248`  
**Závažnosť:** LOW

## Popis

V `_mat_mult` vnútri `pgl2` môže `first_nonzero` ostať 0 ak by boli všetky položky výsledkovej matice nula:

```python
first_nonzero = 0
for v in (ra, rb, rc, rd):
    if v != 0:
        first_nonzero = v
        break
inv_fn = _gf_inv_mult(first_nonzero, p)  # pow(0, p-2, p) = 0
# → normalizovaná n-tica (0,0,0,0) nie je v rep_to_idx → KeyError
```

## Dôsledok

Pre valídne invertibilné matice (det ≠ 0) sa to nemôže stať — súčin dvoch invertibilných matíc je invertibilná. Chýba však explicitná ochrana alebo assertion, čo sťažuje debugovanie pri neplatnom vstupe.
