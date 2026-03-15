"""Model naming helpers."""


def build_name(model_name: str, extra: str = "") -> str:
    """Build a model name with an optional extra suffix."""
    base = model_name.strip()
    extra_clean = extra.strip().strip("_")
    if not extra_clean:
        return base
    return f"{base}_{extra_clean}"
