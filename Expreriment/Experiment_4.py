# coding=utf-8

import os
import csv

from DeepLearningClassifier import LeaveOneOut
from DatasetManager.Costants import *
from DatasetManager import HandManager
from DeepLearningClassifier import TaskManager

METRICS_KEY = 0
TASKS_KEY = 1


class SelectTask:

    def __init__(self, saving_path, minimum_samples, samples_len, feature_extraction, feature):
        HandManager.set_root_directory()
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        self.__feature_extraction = feature_extraction
        self.__feature = feature
        if saving_path is not None:
            self.__saving_path = os.path.join("experiment_result", saving_path)
            self.__task_to_do = self.__set_task_to_do()
        else:
            self.__saving_path = os.path.join("experiment_result", "experiment_4.txt")
            self.__task_to_do = TASKS

    def select_task(self):
        leave_one_out = LeaveOneOut(self.__minimum_samples, self.__samples_len, self.__feature_extraction, self.__feature, "Dataset")
        for task in self.__task_to_do:
            accuracy, precision, recall, f_score = leave_one_out.leave_one_out(task, task)
            if os.path.exists(self.__saving_path):
                mode = 'a'
            else:
                mode = 'w'
            with open(self.__saving_path, mode) as file:
                csv_file = csv.writer(file, delimiter=";")
                csv_file.writerow([(accuracy, precision, recall, f_score), task])
                file.close()

    def __set_task_to_do(self):
        tasks = []
        with open(self.__saving_path, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            for row in csv_file:
                tasks.append(row[TASKS_KEY])
            file.close()
        return TaskManager.get_tasks_difference(TASKS, tasks, [])
