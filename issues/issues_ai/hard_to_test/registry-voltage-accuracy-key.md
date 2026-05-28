# registry-voltage-accuracy-key

**Súbor:** `ai/registry.py:137`  
**Závažnosť:** MEDIUM

## Popis

`list_trained_models` triedi modely podľa kľúča `"accuracy"`, ale voltage girth tréning ukladá metriky pod kľúčom `"test_accuracy"`. Výsledok: všetky voltage_girth modely dostanú sort score 0 → zoradenie je fakticky náhodné → `get_best_model_id("voltage_girth")` nevráti najlepší model.

```python
# registry.py:137 — chybné:
models.sort(
    key=lambda x: x.get("metrics", {}).get("accuracy", 0),  # pre voltage_girth vždy 0!
    reverse=True,
)
```

## Dôsledok

Pri automatickom výbere najlepšieho voltage girth modelu sa vráti náhodný model namiesto toho s najvyššou presnosťou.

## Oprava

```python
sort_key = "test_accuracy" if task == "voltage_girth" else "accuracy"
models.sort(
    key=lambda x: x.get("metrics", {}).get(sort_key, 0),
    reverse=True,
)
```
