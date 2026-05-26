# gnn-single-layer-dim-crash

**Súbor:** `ai/models/gcn.py:54-59`, `ai/models/gin.py:55-61`, `ai/models/sage.py:52-58`  
**Závažnosť:** MEDIUM

## Popis

Pri `num_layers=1` všetky tri modely (GCN, GIN, SAGE) vytvoria output vrstvu s `in_channels=hidden_dim`, ale forward prechod zavolá túto vrstvu priamo na vstup s tvarom `[N, input_dim]`. Keďže `input_dim ≠ hidden_dim` (napr. 1 vs 64), nastane `RuntimeError: size mismatch`.

```python
# gcn.py:59 — output vrstva predpokladá hidden_dim vstup:
self.convs.append(GCNConv(hidden_dim, output_dim))
# Pri num_layers=1 forward prechod: convs[-1](x) kde x má tvar [N, input_dim=1]
# → RuntimeError pri forward (1 ≠ 64)

# forward slučka:
for i in range(num_layers-1):   # range(0) = prázdne
    x = self.convs[i](x, edge_index)
return self.convs[-1](x, edge_index)   # ← volá sa na [N, input_dim], nie [N, hidden_dim]
```

## Dôsledok

Crash pri akomkoľvek trénovaní/inferenci s `num_layers=1`. GPS a Loopy GNN majú oddelenú vstupnú projekciu a sú správne.

## Oprava

Pridať špeciálny prípad pre `num_layers=1`:

```python
if num_layers == 1:
    self.convs.append(GCNConv(input_dim, output_dim))
else:
    self.convs.append(GCNConv(input_dim, hidden_dim))
    for _ in range(num_layers - 2):
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
    self.convs.append(GCNConv(hidden_dim, output_dim))
```
