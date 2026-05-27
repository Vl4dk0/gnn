# rl-train-cpu-cuda-mismatch

**Súbor:** `ai/cage/rl/train.py:353-356`  
**Závažnosť:** HIGH

## Popis

`compute_gae` vracia CPU tensor, zatiaľ čo `value_buffer` tensor je vytváraný na `device` (potenciálne CUDA). Sčítanie tensorov z rôznych zariadení hodí `RuntimeError`.

```python
# compute_gae vracia:
return torch.tensor(advantages, dtype=torch.float)  # CPU tensor!

# Riadok 353-356:
advantages = compute_gae(reward_buffer, values, done_buffer, gamma, gae_lambda)
# advantages je na CPU

returns = advantages + torch.tensor(value_buffer, device=device)
# ← RuntimeError: Expected all tensors to be on the same device pri CUDA!
```

## Dôsledok

Tréning zlyháva na GPU (`RuntimeError: Expected all tensors to be on the same device`) — CPU vs CUDA tensor nesúlad. Na CPU funguje správne, čo môže maskovať bug pri lokálnom vývoji.

## Oprava

```python
advantages = compute_gae(reward_buffer, values, done_buffer, gamma, gae_lambda)
advantages = advantages.to(device)  # ← presunúť pred sčítaním
returns = advantages + torch.tensor(value_buffer, device=device)
```
