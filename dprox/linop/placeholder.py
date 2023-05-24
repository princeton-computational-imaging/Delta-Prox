from .constant import Constant


class Placeholder(Constant):
    def __init__(self, default=None):
        super().__init__(default)
        self.watchers = []

    @property
    def value(self):
        return self._value.to(self.device)

    @value.setter
    def value(self, val):
        """Assign a value to the variable.
        """
        self._value = val
        for watcher in self.watchers:
            watcher(val)

    def change(self, fn):
        self.watchers.append(fn)
