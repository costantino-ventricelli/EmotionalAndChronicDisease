# coding=utf-8

import sys
sys.path.append('..')

from Expreriment.Experiment_2 import Experiment2


# Questo esperimento è il più dispendioso in assoluto, in fatti non è mai stato portato a termine, questo perché lo scopo
# è quello di combinare in ogni modo possibile tutti i 23 task che costituiscono il dataset hand.


def main():
    Experiment2(50, 2500)


if __name__ == '__main__':
    main()
