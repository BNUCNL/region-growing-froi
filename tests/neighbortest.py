__author__ = 'ZhenZongLei'
import os

from algorithm.neighbor import *


if __name__ == "__main__":
    print os.getcwd()
    nb = Connectivity(3, 26, (15, 15, 15))
    a = nb.compute()
    print a
