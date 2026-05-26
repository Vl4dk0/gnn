# curriculum-unlock-sparse-history

**Súbor:** `ai/cage/voltage/rl_env.py:315-318`  
**Závažnosť:** HIGH

## Popis

Curriculum odomykanie sleduje históriu pre `newest = configs[unlocked - 1]` (poslednú odomknutú konfiguráciu). Ale keďže epizódy sa vyberajú uniformne náhodne z `pool = configs[:unlocked]`, história pre `newest` rastie len v `1/unlocked` epizód.

```python
# Riadok 315-318:
newest = self.configs[self.unlocked - 1]
history = self.success_history.get(newest, [])
recent = history[-8:]
if sum(recent) >= 6 and self.unlocked < len(self.configs):
    self.unlocked += 1
```

Pre `unlocked = 5`, iba ~20% epizód ide do `newest` stage. Kritérium `len(history) >= 8` si vyžaduje ~40 epizód (namiesto 8) — navyše agent sa najnovšej stage učí hlavne len 20% času.

## Dôsledok

Curriculum postup je neúmerne pomalý pre vyšší počet odomknutých stagov. Agent strávi dlhý čas na starých (ľahkých) úlohách pred tým, ako sa odomkne ďalšia náročnejšia konfigurácia.

## Oprava

Buď sledovať históriu per-stage zvlášť a počítať `recent` len z posledných N **relevantných** epizód, alebo použiť vážený sampling (novšie stagey oveľa pravdepodobnejšie) a odomknúť na základe rolling window posledných 8 epizód s newest stage:

```python
recent = [s for s in history[-40:] if ...][-8:]   # filter len pre newest
```
