from .constant import Constant


class Placeholder(Constant):
    def __init__(self, default=None):
        super().__init__(default)

    @property
    def value(self):
        return self._value.to(self.device)

    @value.setter
    def value(self, val):
        """Assign a value to the variable.
        """
        self._value = val
