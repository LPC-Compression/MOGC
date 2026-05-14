class Registry(dict):
    """A dict subclass usable as a decorator-based class registry."""

    def register(self, name: str):
        def decorator(cls):
            self[name] = cls
            return cls
        return decorator
