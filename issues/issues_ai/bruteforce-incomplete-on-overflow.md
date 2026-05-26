# bruteforce-incomplete-on-overflow

**Súbor:** `ai/cage/functions/bruteforce.py:80-89`  
**Závažnosť:** HIGH

## Popis

Po prekročení `upper_bound` a popnutí zásobníka sa nekontroluje, či zásobník ostal prázdny. Ak pop odoberie posledný prvok, `is_complete` zostáva `False` — search beží donekonečna.

```python
# Riadok 80-89 — chybné:
if self.graph.number_of_nodes() > self.upper_bound:
    if self.search_stack:
        _ = self.search_stack.pop()
        if self.search_stack:
            self.graph = self.search_stack[-1][0].copy()
        # ← keď zásobník ostane prázdny po pop(), is_complete sa nenastaví!
    else:
        self.is_complete = True
        self.success = False
    return
```

Ďalší `step()` bude pokračovať s prázdnym `search_stack` a neplatným `self.graph`.

## Dôsledok

Nekonečná slučka alebo nesprávny stav po vyčerpaní priestoru stavov cez `upper_bound`. Search nikdy neohlási dokončenie.

## Oprava

```python
if self.graph.number_of_nodes() > self.upper_bound:
    if self.search_stack:
        _ = self.search_stack.pop()
        if self.search_stack:
            self.graph = self.search_stack[-1][0].copy()
        else:
            self.is_complete = True   # ← pridať
            self.success = False
    else:
        self.is_complete = True
        self.success = False
    return
```
