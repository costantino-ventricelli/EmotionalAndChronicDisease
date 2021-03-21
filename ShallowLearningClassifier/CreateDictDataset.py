# coding=utf-8

import os

import numpy as np
import pandas as pd

from DatasetManager import HandManager
from DatasetManager import TaskManager
from DatasetManager.Costants import *

FEATURES_DIRECTORY = os.path.join("resource", "features")


class CreateDictDataset:

    def __init__(self, features_selected, healthy_task, disease_task):
        self.__patients_list = CreateDictDataset.__read_patient_list()
        self.__dataset_dict = {}
        self.__features_selected = features_selected
        for id in self.__patients_list:
            self.__dataset_dict[id] = {
                'ground_thought': HandManager.get_state_from_id(id),
                'user_tasks': dict.fromkeys(TASKS, np.ndarray)
            }
        if healthy_task is None:
            healthy_task = TASKS
        if disease_task is None:
            disease_task = TASKS
        self.__healthy_task = healthy_task
        self.__disease_task = disease_task
        for task in TASKS:
            print("Opening file: ", TASKS_MAME.get(task) + '.csv')
            with open(os.path.join(FEATURES_DIRECTORY, TASKS_MAME.get(task) + '.csv'), 'r') as file:
                dataframe = pd.read_csv(file)
                features = dataframe.columns.to_list()
                for i in range(len(self.__patients_list)):
                    row = np.zeros(0)
                    row_data = dataframe.iloc[i]
                    for index in self.__features_selected:
                        if isinstance(row_data[features.index(index)], str):
                            row = np.append(row, np.abs(eval(row_data[features.index(index)])))
                        else:
                            row = np.append(row, row_data[features.index(index)])
                    row = np.array(row).astype(np.float32)
                    row = np.where(row == np.inf, np.finfo(np.float32).max, row)
                    row = np.where(row == -np.inf, np.finfo(np.float32).min, row)
                    row = np.where(np.isnan(row), 0.00, row)
                    user_tasks = self.__dataset_dict.get(self.__patients_list[i]).get('user_tasks')
                    user_tasks[task] = row
                file.close()

    def get_dataset(self, test_id, test_task=None):
        if test_task is None:
            test_task = TaskManager.get_tasks_difference(TASKS, self.__healthy_task, self.__disease_task)
        healthy_dataset = []
        disease_dataset = []
        test_dataset = []
        test_ground_thought = []
        for key, value in self.__dataset_dict.items():
            user_tasks = value.get('user_tasks')
            if key == test_id:
                test_dataset = CreateDictDataset.__get_task_from_patient(test_task, test_dataset, user_tasks)
                test_ground_thought = [value.get('ground_thought') for _ in range(len(test_dataset))]
            else:
                ground_thought = value.get('ground_thought')
                if ground_thought == HEALTHY:
                    healthy_dataset = CreateDictDataset.__get_task_from_patient(self.__healthy_task, healthy_dataset, user_tasks)
                else:
                    disease_dataset = CreateDictDataset.__get_task_from_patient(self.__disease_task, disease_dataset, user_tasks)
        healthy_dataset, disease_dataset = HandManager.balance_dataset(healthy_dataset, disease_dataset)
        train_ground_thought = [HEALTHY for _ in range(len(healthy_dataset))] + [DISEASE for _ in range(len(disease_dataset))]
        return healthy_dataset + disease_dataset, test_dataset, train_ground_thought, test_ground_thought, test_task

    def get_patient_list(self):
        return self.__patients_list

    @staticmethod
    def __get_task_from_patient(tasks, dataset, user_tasks):
        for sub_task in tasks:
            if not isinstance(sub_task, list):
                sub_task = [sub_task]
                for task in sub_task:
                    row = user_tasks.get(task)
                    if not isinstance(row, type) and np.any(row):
                        dataset.append(row)
        return dataset

    @staticmethod
    def __read_patient_list():
        with open(os.path.join(FEATURES_DIRECTORY, "patients.txt")) as file:
            data = file.readline().replace('[', '').replace(']', '')
            results = np.array(data.split(',')).astype(int)
            file.close()
        return results
