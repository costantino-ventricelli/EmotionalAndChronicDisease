# coding=utf-8

import pandas as pd
import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from copy import deepcopy
from DatasetManager.Costants import *
from DatasetManager import HandManager
from .FeaturesExtraction import FeatureExtraction


FEATURES_DIRECTORY = os.path.join("resource", "features")


class FeatureSelection:

    def __init__(self, healthy_task=None, disease_task=None):
        print("Generating dataset")
        HandManager.set_root_directory()
        self.__patients = FeatureSelection.__read_patient_list()
        if healthy_task is None and disease_task is None:
            healthy_task = TASKS
            disease_task = TASKS
        self.__dataset, self.__ground_thought = self.__set_dataset(healthy_task, disease_task)
        self.__dataset = np.array(self.__dataset).astype(np.float32)
        self.__dataframe = None
        self.__feature_selected = None
        self.__best_params = None
        self.__random_gird = {
                    'n_estimators': [int(item) for item in np.linspace(start=200, stop=2000, num=10)],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': [int(item) for item in np.linspace(start=10, stop=110, num=11)] + [None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]}

    def select_feature(self):
        print("Starting optimization")
        random_forest_classifier = RandomForestClassifier()
        hyperparameters = RandomizedSearchCV(estimator=random_forest_classifier, param_distributions=self.__random_gird,
                                             n_iter=100, cv=3, verbose=3, n_jobs=-1)
        hyperparameters.fit(self.__dataset, self.__ground_thought)
        print("Finish optimization")
        print("Selected hyperparameters: ", hyperparameters.best_params_)
        self.__best_params = hyperparameters.best_params_
        print("Start features selection")
        model_selection = SelectFromModel(RandomForestClassifier(n_estimators=self.__best_params.get('n_estimators'),
                                                                 max_features=self.__best_params.get('max_features'),
                                                                 max_depth=self.__best_params.get('max_depth'),
                                                                 min_samples_split=self.__best_params.get('min_samples_split'),
                                                                 min_samples_leaf=self.__best_params.get('min_samples_leaf'),
                                                                 bootstrap=self.__best_params.get('bootstrap')))
        model_selection.fit(self.__dataset, self.__ground_thought)
        self.__dataframe = pd.DataFrame(data=self.__dataset, columns=list(FeatureExtraction.get_file_dictionary()))
        print("Finish features selection")
        self.__feature_selected = self.__dataframe.columns[model_selection.get_support()]
        return self.__best_params, self.__feature_selected

    def get_best_params(self):
        return self.__best_params

    def get_feature_selected(self):
        return self.__feature_selected

    def get_dataframe(self):
        return self.__dataframe

    def get_ground_thought(self):
        return self.__ground_thought

    def __set_dataset(self, healthy_task, disease_task):
        healthy_data_frame = []
        disease_data_frame = []
        healthy_ids = []
        disease_ids = []
        for task in list(set(healthy_task + disease_task)):
            print("Extracting task: ", TASKS_MAME.get(task))
            healthy_data_frame_task = []
            healthy_ids_task = []
            disease_data_frame_task = []
            disease_ids_task = []
            with open(os.path.join(FEATURES_DIRECTORY, TASKS_MAME.get(task) + ".csv")) as file:
                data_frame = pd.read_csv(file)
                for i in range(len(data_frame)):
                    if data_frame[data_frame.columns[0]][i] != 0.00:
                        row = deepcopy(data_frame.iloc[i])
                        for index in range(len(row)):
                            if isinstance(row[index], str):
                                row[index] = np.abs(eval(row[index]))
                        row = np.array(row).astype(np.float32)
                        if HandManager.get_state_from_id(self.__patients[i]) == HEALTHY and task in healthy_task:
                            healthy_data_frame_task.append(row)
                            healthy_ids_task.append(self.__patients[i])
                        elif HandManager.get_state_from_id(self.__patients[i]) == DISEASE and task in disease_task:
                            disease_data_frame_task.append(row)
                            disease_ids_task.append(self.__patients[i])

                healthy_data_frame_task, disease_data_frame_task = HandManager.balance_dataset(healthy_data_frame_task, disease_data_frame_task)
                healthy_data_frame += healthy_data_frame_task
                healthy_ids += healthy_ids_task[0: len(healthy_data_frame_task)]
                disease_data_frame += disease_data_frame_task
                disease_ids += disease_ids_task[0: len(disease_data_frame_task)]
        dataset = np.array(healthy_data_frame + disease_data_frame)
        # Questi tre punti sono importanti in quanto permettono di rimuovere i dati nocivi dal dataset.
        dataset = np.where(dataset == np.inf, np.finfo(np.float32).max, dataset)
        dataset = np.where(dataset == -np.inf, np.finfo(np.float32).min, dataset)
        dataset = np.where(np.isnan(dataset), 0.00, dataset)
        ground_through = np.array([HEALTHY for _ in range(len(healthy_ids))] + [DISEASE for _ in range(len(disease_ids))])
        return dataset, ground_through

    @staticmethod
    def __read_patient_list():
        with open(os.path.join(FEATURES_DIRECTORY, "patients.txt")) as file:
            data = file.readline().replace('[', '').replace(']', '')
            results = np.array(data.split(',')).astype(int)
            file.close()
        return results

    def __get_ground_through(self):
        ground_through = []
        for id in self.__patients:
            ground_through.append(HandManager.get_state_from_id(id))
        return ground_through
