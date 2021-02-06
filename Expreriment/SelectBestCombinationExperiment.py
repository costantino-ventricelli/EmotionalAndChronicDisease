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


class SelectBestCombinationExperiment:

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
            self.__value_tasks[key] = max(SelectBestCombinationExperiment.__set_previous_state(item).items())[TASK_INDEX]

    def start_linear_selection(self):
        healthy = []
        disease = []
        learning_method = LeaveOneOut(self.__minimum_samples, self.__samples_len, FEATURES, "Dataset")
        file_path = os.path.join("experiment_result", "linear_selection.txt")
        for _, values in self.__value_tasks.items():
            for value in values:
                healthy.append(value[HEALTHY_STRING])
                disease.append(value[DISEASE_STRING])
                accuracy, precision, recall, f_score = learning_method.leave_one_out(healthy, disease)
                print("METRICS: ", accuracy, ", ", precision, ", ", recall, ", ", f_score)
                if os.path.exists(file_path):
                    mode = 'a'
                else:
                    mode = 'w'
                with open(file_path, mode) as file:
                    csv_file = csv.writer(file, delimiter=';')
                    csv_file.writerow([(accuracy, precision, recall, f_score), healthy, disease])
                    file.close()

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
                value_dict = {HEALTHY_STRING: row[HEALTHY_INDEX], DISEASE_STRING: row[DISEASE_INDEX]}
                if key in result.keys():
                    list_of_dict = result.get(key)
                    list_of_dict.append(value_dict)
                else:
                    result[key] = [value_dict]
            file.close()
        return result
