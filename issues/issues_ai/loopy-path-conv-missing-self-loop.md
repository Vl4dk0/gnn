# loopy-path-conv-missing-self-loop

**Súbor:** `ai/models/loopy.py:257-260`  
**Závažnosť:** HIGH

## Popis

V triede `PathConv` (ktorá implementuje GIN-like konvolúciu na cestách) dochádza k prepísaniu tenzora `x` výsledkom propagácie susedov pred samotným aplikovaním GIN-formuly:

```python
        # Propagate along path using 1D conv with kernel [1, 0, 1]
        x = self._path_propagate(x)

        # Apply MLP and sum over path
        x = cast(Tensor, self.mlp((1 + self.eps) * x))
```

To znamená, že namiesto výpočtu $(1 + \epsilon) x_i + \sum x_{\text{neighbors}}$ sa vypočíta iba $(1 + \epsilon) \sum x_{\text{neighbors}}$. Vlastný stav/črta vrcholu na ceste sa pred MLP úplne zahodí.

## Dôsledok

Loopy GNN stráca identitu vrcholov počas propagácie na ceste. Model nedokáže správne zachovať informácie o uzloch, čo výrazne degraduje jeho expresívnu silu (Loopy GNN má byť ekvivalentom 1-WL nad cestami, ale kvôli tejto chybe stráca dôležité štrukturálne vlastnosti).

## Oprava

Uchovať pôvodnú hodnotu `x` (vlastnú črtu) a spočítať kombináciu s agregovaným príspevkom správne:

```python
        # Propagate along path using 1D conv with kernel [1, 0, 1]
        agg = self._path_propagate(x)

        # Apply MLP and sum over path
        x = cast(Tensor, self.mlp((1 + self.eps) * x + agg))
```


## Test

Proven by [`test_pathconv_preserves_self_contribution`](../../tests/test_issues_ai.py) (test fails → bug confirmed).
