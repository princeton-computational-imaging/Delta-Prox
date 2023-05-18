
class matmul:
    def __init__(self, var, A):
        self.A = A
        self.var = var

    def __eq__(self, other):
        return equality(self, other)

    def __le__(self, other):
        return less(self, other)


class equality:
    def __init__(self, left: matmul, right):
        self.left = left
        self.right = right


class less:
    def __init__(self, left: matmul, right):
        self.left = left
        self.right = right
