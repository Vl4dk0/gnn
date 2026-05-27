# voltage-rl-train-cpu-cuda-mismatch

**Súbor:** `ai/cage/voltage/rl_train.py:292, 335`  
**Závažnosť:** HIGH

## Popis

`returns` tensor je vytvorený na CPU, ale `value` (výstup GNN modelu) je na GPU pri CUDA trénovaní. Device mismatch spôsobí `RuntimeError`.

```python
# Riadok 292:
returns = advantages + torch.tensor(value_buffer)
# ← advantages aj value_buffer → CPU tensor

# Riadok 335:
value_loss = 0.5 * (returns[i] - value.squeeze()) ** 2
# ← value je na GPU (výstup modelu) → RuntimeError pri CUDA!
```

Rovnaký bug existuje aj v `ai/cage/rl/train.py` (zdokumentovaný v `rl-train-cpu-cuda-mismatch.md`).

## Dôsledok

`RuntimeError: Expected all tensors to be on the same device` — tréning zlyháva na GPU. Lokálne na CPU funguje správne, čo maskuje bug.

## Oprava

```python
device = next(agent.parameters()).device
advantages = compute_gae(reward_buffer, values, done_buffer, gamma, gae_lambda)
advantages = advantages.to(device)
returns = advantages + torch.tensor(value_buffer, device=device)
```
