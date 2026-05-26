# api-blueprint-unregistered

**Súbor:** `backend/routes/api.py`, `backend/app.py:29-31`  
**Závažnosť:** LOW

## Popis

`api_bp` blueprint definovaný v `backend/routes/api.py` nie je nikdy zaregistrovaný v `create_app()`. Všetky endpointy v tomto súbore sú mŕtvy kód.

```python
# app.py:29-31 — api_bp chýba:
app.register_blueprint(degree_bp)
app.register_blueprint(cage_bp)
app.register_blueprint(min_cycle_bp)
# api_bp z backend/routes/api.py nie je zaregistrovaný!
```

Existuje separátny `/api/config` endpoint priamo v `app.py`, čo naznačuje, že `backend/routes/api.py` je zastaraný duplikát.

## Dôsledok

Endpointy v `backend/routes/api.py` nikdy nedostanú požiadavku. Kód je zbytočný a mätúci.

## Oprava

Odstrániť `backend/routes/api.py` ak je jeho obsah duplikátom, alebo zaregistrovať `api_bp` v `app.py` ak sú endpointy potrebné.
