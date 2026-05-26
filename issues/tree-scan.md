# tree-scan

**Súbor:** `ai/cage/voltage/base_graphs.py:96`  
**Závažnosť:** MEDIUM

## Popis

`spanning_tree_edge_indices` testuje príslušnosť pomocou lineárneho skenu zoznamu:

```python
for idx in tree_indices:   # O(|tree_indices|) pri každej hrane
    continue
```

Celková zložitosť BFS smyčky: O(V × E × V) namiesto O(V × E).

## Oprava

```python
tree_set: set[int] = set()
# ...
if idx in tree_set:      # O(1)
    continue
tree_set.add(idx)
# ...
return list(tree_set)
```
