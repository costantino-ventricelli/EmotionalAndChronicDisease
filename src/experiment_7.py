# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import Experiment4
from DeepLearningClassifier import RTPExtraction


# Questo esperimento seleziona il migliore dei task tra tutti usando come feature l'RTP invece degli RHS, utilizzando
# Experiment_4 come modello.


def main():
    select_task = Experiment4('experiment_8.txt', 2500, 50, RTPExtraction(2500, 50), feature=4)
    select_task.select_task()


if __name__ == '__main__':
    main()
