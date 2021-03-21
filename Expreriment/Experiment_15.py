# coding=utf-8

import csv
import os

from DatasetManager import HandManager
from DatasetManager.Costants import *
from DeepLearningClassifier import DeepKFoldValidation

EXPERIMENT_FOLDER = os.path.join("experiment_result", "experiment_15.txt")


class AttentionLayer:

    def __init__(self, min_age, max_age):
        HandManager.set_root_directory()
        with open(EXPERIMENT_FOLDER, 'w') as file:
            file.write("Experiment in age range: (" + str(min_age) + ", " + str(max_age) + ")\n")
            file.close()
        healthy_ids, mild_ids, disease_ids = HandManager.get_ids_age(min_age, max_age)
        file_manager = HandManager("Dataset")
        self.__file_paths = file_manager.get_files_path()
        self.__dataset = HandManager.get_dict_dataset(healthy_ids + mild_ids + disease_ids,
                                               [0 for _ in range(len(healthy_ids))]
                                               + [1 for _ in range(len(mild_ids))]
                                               + [2 for _ in range(len(disease_ids))], self.__file_paths)
        self.__k_fold_validation = DeepKFoldValidation(healthy_ids, mild_ids, disease_ids, self.__dataset)
        for task in TASKS:
            accuracy, precision, recall, f_score = self.__k_fold_validation.start_k_fold_validation(task, task, task, task)
            with open(EXPERIMENT_FOLDER, 'a') as file:
                csv_file = csv.writer(file, delimiter=';')
                csv_file.writerow([(accuracy, precision, recall, f_score), task, task, task])
                file.close()
