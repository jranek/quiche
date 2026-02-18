_PLOT_IMPORT_ERROR = None

try:
    from .plot import *  # noqa: F401,F403
except ImportError as exc:  # pragma: no cover - exercised in optional dependency environments
    _PLOT_IMPORT_ERROR = exc
    __all__ = []


def __getattr__(name):
    if _PLOT_IMPORT_ERROR is not None:
        raise ImportError(
            "Plotting dependencies are missing. Install quiche with the [plot] extra."
        ) from _PLOT_IMPORT_ERROR
    raise AttributeError(name)
