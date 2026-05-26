# voltage-model-clamp-silent-corruption

**Súbor:** `ai/cage/voltage/model.py:114-117`  
**Závažnosť:** LOW

## Popis

Napätia indexujúce embedding tabuľku sú ticho `clamp`-ované na `[0, max_group_order - 1]` namiesto vyhodenia chyby. Pri inferenci na skupinách s rádom väčším ako `max_group_order`, napätia mimo rozsahu sa tichmo mapujú na posledný embedding index — totálne nesprávna reprezentácia.

```python
# model.py:114-117:
voltage_idx = edge_attr.squeeze(-1).long()
voltage_idx = voltage_idx.clamp(0, self.max_group_order - 1)  # tiché maskovanie chyby
edge_h = self.edge_proj(self.edge_embed(voltage_idx))
```

## Dôsledok

Model produkuje nezmyselné predikcie pre skupiny s rádom > `max_group_order` bez akéhokoľvek upozornenia. Chyba je neviditeľná — môže viesť k falošne dôveryhodným výsledkom.

## Oprava

```python
if voltage_idx.max().item() >= self.max_group_order:
    raise ValueError(
        f"Voltage index {voltage_idx.max().item()} exceeds "
        f"max_group_order={self.max_group_order}"
    )
```
