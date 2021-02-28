# coding=utf-8
import sys
sys.path.append("..")

import os
import csv

from Expreriment import CompleteTaskTest
from DatasetManager import HandManager
from DatasetManager.Costants import *

EXPERIMENT_RESULT = os.path.join("experiment_result", "experiment_11.txt")


def main():
    results = {}
    HandManager.set_root_directory()
    if not os.path.exists(EXPERIMENT_RESULT):
        os.mkdir(EXPERIMENT_RESULT)
    for task in TASKS:
        file = os.path.join(EXPERIMENT_RESULT, TASKS_MAME.get(task) + ".txt")
        accuracy, precision, recall, f_score = CompleteTaskTest.start_experiment(healthy_task=[task], disease_task=[task])
        results = append_dictionary((accuracy, precision, recall, f_score), task, results)
    with open(EXPERIMENT_RESULT, 'w') as file:
        csv_file = csv.writer(file, delimiter=';', quotechar="'", quoting=csv.QUOTE_NONE)
        for key, item in results.items():
            csv_file.writerow([key, item])
        file.close()


def append_dictionary(metrics, task, results):
    if metrics in results.keys():
        value = results.get(metrics)
        if not isinstance(value, list):
            value = [value, task]
        else:
            value.append(task)
    else:
        value = task
    results[metrics] = value
    return results


if __name__ == '__main__':
    main()
