
import numpy as np


# class Symbol(np.ndarray):

    # def __new__(

    # def __init__

    # pass


class Symbol(np.ndarray):

    # def __init__(self, values):
    #     self.values = np.array(values)

    def __new__(cls, values):
        # return np.array(values)

    def __add__(self, other):
        print "hello"
        return np.array(['%s + %s' % (a, b) for a, b in zip(self.values, other)])

    __radd__ = __add__

    def __str__(self, other):
        return str(self.values)


if __name__ == "__main__":

    s = Symbol(['a', 'b'])
    r = np.arange(2)

    print s + r
    print r + s
