# coding=utf-8

import csv
import os

from DeepLearningClassifier import LeaveOneOut
from DatasetManager import FileManager

METRICS_KEY = 0
HEALTHY_INDEX = 1
DISEASE_INDEX = 2
TASK_INDEX = 1
FEATURES = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

SQUARE_BRACKETS = '[]'


class ShiftTaskSelection:

    def __init__(self, saving_path, path_dictionary, minimum_samples, samples_len):
        FileManager.set_root_directory()
        self.__saving_path = saving_path
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        if self.__saving_path is not None:
            self.__results = self.__set_previous_state(self.__saving_path)
        else:
            self.__results = {}
        self.__value_tasks = {}
        for key, item in path_dictionary.items():
            self.__value_tasks[key] = max(ShiftTaskSelection.__set_previous_state(item).items())[TASK_INDEX]

    def start_shift_selection(self, first_combination, file_name):
        leave_one_out = LeaveOneOut(self.__minimum_samples, self.__samples_len, FEATURES, "Dataset")
        file_path = os.path.join("experiment_result", os.path.join("shift_selection", file_name + ".txt"))
        for input_category, tasks in first_combination.items():
            for category, item in self.__value_tasks.items():
                if category != input_category:
                    for value in item:
                        if isinstance(value[HEALTHY_STRING], list):
                            healthy = tasks[HEALTHY_STRING] + value[HEALTHY_STRING]
                            disease = tasks[DISEASE_STRING] + value[DISEASE_STRING]
                        else:
                            healthy = tasks[HEALTHY_STRING] + [value[HEALTHY_STRING]]
                            disease = tasks[DISEASE_STRING] + [value[DISEASE_STRING]]
                        if not self.__is_already_do(healthy, disease):
                            print("Healthy: ", healthy)
                            print("Disease: ", disease)
                            accuracy, precision, recall, f_score = leave_one_out.leave_one_out(healthy, disease)
                            if os.path.exists(file_path):
                                mode = 'a'
                            else:
                                mode = 'w'
                            with open(file_path, mode) as file:
                                csv_file = csv.writer(file, delimiter=';')
                                csv_file.writerow([(accuracy, precision, recall, f_score), healthy, disease])
                                file.close()
                        else:
                            print("Combination already do...")
    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    @staticmethod
    def __set_previous_state(saving_path):
        with open(saving_path, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            result = {}
            for row in csv_file:
                key = eval(row[METRICS_KEY])
                value_dict = {HEALTHY_STRING: ShiftTaskSelection.__cast_into_list(row[HEALTHY_INDEX]),
                              DISEASE_STRING: ShiftTaskSelection.__cast_into_list(row[DISEASE_INDEX])}
                if key in result.keys():
                    list_of_dict = result.get(key)
                    list_of_dict.append(value_dict)
                else:
                    result[key] = [value_dict]
            file.close()
        return result

    """
        Questo metodo verifica se una combinazione di task è già stata provata, verificando la sua esistenza nel diziona-
        rio di istanza contenente i risultati della valutazione.
    """
    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING] == healthy_task and item[DISEASE_STRING] == disease_task:
                    return True
        return False

    @staticmethod
    def __cast_into_list(string):
        for character in SQUARE_BRACKETS:
            string = string.replace(character, '')
        strings = string.split(',')
        for i in range(len(strings)):
            strings[i] = strings[i].replace('\'', "").strip()
        return strings
