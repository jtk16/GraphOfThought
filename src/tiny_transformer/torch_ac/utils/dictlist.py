import torch

class DictList(dict):
    """A dictionary that behaves like a list of dictionaries, or a dictionary of lists."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        if len(self) == 0:
            return 0
        return len(next(iter(self.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, int):
            return {key_: value[key] for key_, value in self.items()}
        elif isinstance(key, slice):
            return DictList({key_: value[key] for key_, value in self.items()})
        else:
            raise TypeError("Invalid argument type")

    def __setattr__(self, name, value):
        if name in self:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)