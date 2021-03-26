# coding=utf-8

from random import shuffle

import numpy as np
import os
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from DatasetManager import TaskManager
from DatasetManager.Costants import *
from DeepLearningClassifier import AttentionModel
from DeepLearningClassifier import MLModel

EXPERIMENT_LOG = os.path.join("resource", "experiment_15.log")


class DeepKFoldValidation:

    def __init__(self, healthy_ids, mild_ids, disease_ids, dataset, fold_number=None):
        if fold_number is None:
            fold_number = 10
        elif fold_number > len(healthy_ids):
            raise ValueError("The number of folder must be less than number of element for class")
        self.__element_for_folder = (len(healthy_ids) * 3) // fold_number
        print("element_for_folder:", self.__element_for_folder)
        self.__fold_number = fold_number
        self.__dataset = dataset
        self.__ids = []
        for iter in range(len(healthy_ids)):
            self.__ids.append(healthy_ids[iter])
            self.__ids.append(mild_ids[iter])
            self.__ids.append(disease_ids[iter])

    def start_k_fold_validation(self, healthy_tasks, mild_tasks, disease_tasks, test_tasks=None):
        disease_tasks, healthy_tasks, mild_tasks, test_tasks = DeepKFoldValidation.__check_input_value(disease_tasks, healthy_tasks,
                                                                                        mild_tasks, test_tasks)
        predicted_values = np.zeros(0)
        ground_thoughts = np.zeros(0)
        for iter_validation in range(self.__fold_number + 1):
            train_ids, validation_ids, test_ids = self.__get_ids_lists(iter_validation)
            training_dataset = []
            training_label = []
            validation_dataset = []
            validation_label = []
            test_dataset = []
            test_label = []
            for id in train_ids:
                ground_thought, tasks_dict, temp_task = self.__get_dataset_label(healthy_tasks, id, mild_tasks, disease_tasks)
                temp_dataset = DeepKFoldValidation.__get_row_data(temp_task, tasks_dict)
                if np.shape(temp_dataset)[0] != 0:
                    training_dataset.append(temp_dataset)
                    training_label.append(ground_thought)
            for id in validation_ids:
                ground_thought, tasks_dict, temp_task = self.__get_dataset_label(healthy_tasks, id, mild_tasks, disease_tasks)
                temp_dataset = DeepKFoldValidation.__get_row_data(temp_task, tasks_dict)
                if np.shape(temp_dataset)[0] != 0:
                    validation_dataset.append(temp_dataset)
                    validation_label.append(ground_thought)
            for id in test_ids:
                temp_dict = self.__dataset.get(id)
                ground_thought = temp_dict.get('ground_thought')
                tasks_dict = temp_dict.get('tasks')
                temp_dataset = DeepKFoldValidation.__get_row_data(test_tasks, tasks_dict)
                if np.shape(temp_dataset)[0] != 0:
                    test_dataset.append(temp_dataset)
                    test_label.append(ground_thought)
            training_dataset, validation_dataset, test_dataset = DeepKFoldValidation.__prepare_dataset(training_dataset, validation_dataset, test_dataset)
            training_label, validation_label, test_label = DeepKFoldValidation.__prepare_label(training_label, validation_label, test_label)
            try:
                ml_model = AttentionModel(training_dataset, training_label, validation_dataset, validation_label, test_dataset, test_label)
                predicted_values = np.concatenate((predicted_values, np.reshape(ml_model.test_model(), -1)))
                ground_thoughts = np.concatenate((ground_thoughts, np.reshape(test_label, -1)))
            except ValueError as error:
                if not os.path.exists(EXPERIMENT_LOG):
                    file = open(EXPERIMENT_LOG, 'w')
                else:
                    file = open(EXPERIMENT_LOG, 'a')
                file.write("Error: " + str(error) + "\n\ttraining_dataset: " + str(training_dataset.shape) + "\n")
                file.close()
        return MLModel.evaluate_results(predicted_values, ground_thoughts)

    @staticmethod
    def __prepare_dataset(training_dataset, validation_dataset, test_dataset):
        training_dataset = DeepKFoldValidation.__scale_dataset(training_dataset)
        validation_dataset = DeepKFoldValidation.__scale_dataset(validation_dataset)
        test_dataset = DeepKFoldValidation.__scale_dataset(test_dataset)
        return training_dataset, validation_dataset, test_dataset

    @staticmethod
    def __scale_dataset(dataset):
        scale = StandardScaler()
        for i in range(len(dataset)):
            temp = dataset[i]
            temp = scale.fit_transform(temp)
            dataset[i] = temp
        return dataset

    @staticmethod
    def __prepare_label(training_label, validation_label, test_label):
        training_label = np.array(to_categorical(training_label)).astype(np.float32)
        validation_label = np.array(to_categorical(validation_label)).astype(np.float32)
        test_label = DeepKFoldValidation.__prepare_test_label(test_label)
        return training_label, validation_label, test_label

    @staticmethod
    def __prepare_test_label(labels):
        test_label = np.zeros((len(labels), 3))
        for i in range(len(labels)):
            test_label[i][labels[i]] = 1
        return test_label

    @staticmethod
    def __make_tensor(test_dataset, test_label, train_dataset, train_label, validation_dataset, validation_label):
        train_dataset = np.array(train_dataset).astype(np.float32)
        train_label = np.array(train_label).astype(np.float32)
        validation_dataset = np.array(validation_dataset).astype(np.float32)
        validation_label = np.array(validation_label).astype(np.float32)
        test_dataset = np.array(test_dataset).astype(np.float32)
        test_label = np.array(test_label).astype(np.float32)
        return test_dataset, test_label, train_dataset, train_label, validation_dataset, validation_label

    def __get_ids_lists(self, iter_validation):
        temp_train_ids = self.__ids[: iter_validation * self.__element_for_folder] + self.__ids[(iter_validation + 1) * self.__element_for_folder:]
        validation_number = int(np.ceil(len(temp_train_ids) * 0.20))
        train_ids = temp_train_ids[0: len(temp_train_ids) - validation_number]
        shuffle(train_ids)
        validation_ids = temp_train_ids[len(temp_train_ids) - validation_number: len(temp_train_ids)]
        shuffle(validation_ids)
        test_ids = self.__ids[iter_validation * self.__element_for_folder: (iter_validation + 1) * self.__element_for_folder]
        shuffle(test_ids)
        return train_ids, validation_ids, test_ids

    @staticmethod
    def __check_input_value(disease_tasks, healthy_tasks, mild_tasks, test_tasks):
        if test_tasks is None:
            test_tasks = TaskManager.get_tasks_difference(TASKS, healthy_tasks, mild_tasks, disease_tasks)
        elif not isinstance(test_tasks, list):
            test_tasks = [test_tasks]
        if not isinstance(healthy_tasks, list):
            healthy_tasks = [healthy_tasks]
        if not isinstance(mild_tasks, list):
            mild_tasks = [mild_tasks]
        if not isinstance(disease_tasks, list):
            disease_tasks = [disease_tasks]
        return disease_tasks, healthy_tasks, mild_tasks, test_tasks

    def __get_dataset_label(self, healthy_tasks, id, mild_tasks, disease_tasks):
        temp_dict = self.__dataset.get(id)
        ground_thought = temp_dict.get('ground_thought')
        tasks_dict = temp_dict.get('tasks')
        if ground_thought == 0:
            temp_task = healthy_tasks
        elif ground_thought == 1:
            temp_task = mild_tasks
        else:
            temp_task = disease_tasks
        return ground_thought, tasks_dict, temp_task

    @staticmethod
    def __get_row_data(tasks, tasks_dict):
        dataset = np.zeros((0, 2))
        for task in tasks:
            row_point = tasks_dict.get(task)
            if row_point is not None:
                dataset = np.concatenate((dataset, row_point))
        return dataset
