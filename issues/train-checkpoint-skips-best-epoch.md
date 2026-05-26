# train-checkpoint-skips-best-epoch

**Súbor:** `ai/cage/voltage/train.py:338-353`  
**Závažnosť:** MEDIUM

## Popis

Model sa checkpointuje ako "best" iba na epochách, ktoré sú násobkami `print_every`, plus epocha 1 a posledná. Validačná strata sa vôbec nevyhodnocuje v ostatných epochách.

```python
if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
    val = _evaluate(model, val_loader, device, regression_weight)
    ...
    if val["loss"] < best_val_loss:
        best_state = ...   # nikdy sa nezavolá pre epochy mimo print_every
        best_epoch = epoch
```

Ak je skutočná najlepšia epocha napr. 47 pri `print_every=10`, model na nej nikdy nebol evaluovaný a nikdy sa neuloží.

## Dôsledok

Uložený model môže byť výrazne horší ako skutočné tréningové optimum — degraduje kvalitu natrénovaného girth prediktora.

## Oprava

```python
# Evaluovať každú epochu, loggovať len každých print_every:
val = _evaluate(model, val_loader, device, regression_weight)
if val["loss"] < best_val_loss:
    best_val_loss = val["loss"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = epoch
if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
    print(f"Epoch {epoch:4d} | ...")
```
