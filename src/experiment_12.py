# coding=utf-8

import sys
import os
import experiment_11
import csv

sys.path.append("..")

from Expreriment import CompleteTaskTest
from DatasetManager import HandManager
from DatasetManager.Costants import *
from copy import deepcopy


EXPERIMENT_11 = os.path.join("experiment_result", "experiment_11.txt")
EXPERIMENT_12 = os.path.join("experiment_result", "experiment_12.txt")

# Questo esprimento prova le combinazioni di tutti i task in modo dipendente:
#   healthy_task = disease_task = test_task

KEY = 0
VALUE = 1


def main():
    HandManager.set_root_directory()
    if not os.path.exists(EXPERIMENT_11):
        print("Executing experiment 11 to get the list of singular task result")
        experiment_11.main()
    if not os.path.exists(EXPERIMENT_12):
        print(EXPERIMENT_12, "file created")
        file = open(EXPERIMENT_12, 'w')
        file.close()
    key, tasks = max(read_dictionary().items())
    best_result = {key: tasks}
    previous_max = ((0.00, 0.00, 0.00, 0.00), '')
    actual_tuple = max(best_result.items())
    tasks = deepcopy(TASKS)
    del tasks[tasks.index(actual_tuple[VALUE])]
    while previous_max[KEY] <= actual_tuple[KEY]:
        best_result = {}
        for task in tasks:
            if task not in actual_tuple[VALUE]:
                test_tasks = deepcopy(actual_tuple[VALUE])
                if isinstance(test_tasks, list):
                    test_tasks.append(task)
                else:
                    test_tasks = [test_tasks, task]
                print("Starting model construction for task: ", test_tasks)
                accuracy, precision, recall, f_score = CompleteTaskTest.start_experiment(healthy_task=test_tasks,
                                                                                         disease_task=test_tasks,
                                                                                         test_task=test_tasks)
                partial_result = ((accuracy, precision, recall, f_score), test_tasks)
                if partial_result[KEY] in best_result.keys():
                    value = best_result.get(partial_result[KEY])
                    if isinstance(value, list):
                        value.append(partial_result[VALUE])
                    else:
                        best_result[partial_result[KEY]] = [value, partial_result[VALUE]]
                else:
                    best_result[partial_result[KEY]] = partial_result[VALUE]
        actual_tuple = max(best_result.items())
        if actual_tuple >= previous_max:
            previous_max = deepcopy(actual_tuple)
        print("Task selection end")
        with open(EXPERIMENT_12, 'a') as file:
            csv_file = csv.writer(file, delimiter=";")
            csv_file.writerow([actual_tuple[KEY], actual_tuple[VALUE]])
            file.close()


def read_dictionary():
    result_dictionary = {}
    with open(EXPERIMENT_11, 'r') as file:
        csv_file = csv.reader(file, delimiter=';', quotechar="'", quoting=csv.QUOTE_NONE)
        for row in csv_file:
            key = eval(row[KEY])
            task = row[VALUE]
            if key in result_dictionary:
                tasks = result_dictionary.get(key)
                if not isinstance(tasks, list):
                    tasks = [tasks]
                tasks.append(task)
            else:
                result_dictionary[key] = task
        file.close()
    return result_dictionary


if __name__ == '__main__':
    main()
