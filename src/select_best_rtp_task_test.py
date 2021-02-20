# coding=utf-8
import sys
sys.path.append("..")

from Expreriment import SelectTask
from DeepLearningClassifier import RTPExtraction


def main():
    select_task = SelectTask('best_task.txt', 2500, 50, RTPExtraction(2500, 50), feature=4)
    select_task.select_task()


if __name__ == '__main__':
    main()
