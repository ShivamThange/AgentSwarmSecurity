__version__ = "1.0.0"


def __getattr__(name):
    if name == "Engine":
        from .engine import Engine
        return Engine
    raise AttributeError(name)


__all__ = ["Engine", "__version__"]
