# coding=utf-8

import csv
import os

from DatasetManager import HandManager
from DatasetManager.Costants import *
from DeepLearningClassifier import DeepKFoldValidation


class Experiment16:

    def __init__(self, age_min, age_max):
        healthy_ids, mild_ids, disease_ids = HandManager.get_ids_age(age_min, age_max)
        self.__file_manager = HandManager("Dataset")
        self.__file_paths = self.__file_manager.get_files_path()
        self.__dataset = HandManager.get_dict_dataset(healthy_ids + mild_ids + disease_ids,
                                                      [0 for _ in range(len(healthy_ids))]
                                                      + [1 for _ in range(len(mild_ids))]
                                                      + [2 for _ in range(len(disease_ids))],
                                                      self.__file_paths)
        self.__k_fold_validation = DeepKFoldValidation(healthy_ids, mild_ids, disease_ids, self.__dataset)

    def start_validation(self, healthy_tasks=None, mild_task=None, disease_task=None):
        if healthy_tasks is None:
            healthy_tasks = TASKS
        if mild_task is None:
            mild_task = TASKS
        if disease_task is None:
            disease_task = TASKS
        return self.__k_fold_validation.start_k_fold_validation(healthy_tasks, mild_task, disease_task)