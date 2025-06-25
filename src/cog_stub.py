# cog_stub.py  ───────────────────────────────────────────────────────────
from pathlib import Path as _Path

# ----------------------------------------------------
# Minimal stand-ins for the 4 cog symbols we import
# ----------------------------------------------------
class Input:                         # noqa: N801
    def __init__(self, *_, **__):
        pass

class BasePredictor:                 # noqa: N801
    def setup(self):
        pass

Path = _Path                         # noqa: N801

class BaseModel:                     # noqa: N801
    """Accept any keyword args and store them as attributes."""
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    def dict(self):
        return self.__dict__
# -----------------------------------------------------------------------