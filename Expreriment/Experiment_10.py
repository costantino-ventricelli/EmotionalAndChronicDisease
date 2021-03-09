# coding=utf-8

import numpy as np

from ShallowLearningClassifier import SVCModel
from ShallowLearningClassifier import CreateDictDataset
from collections import Counter


class CompleteTaskTest:

    @staticmethod
    def start_experiment(healthy_task=None, disease_task=None, test_task=None):
        svm_model = SVCModel(healthy_task, disease_task)
        dataset_creator = CreateDictDataset(svm_model.get_feature_selected(), healthy_task, disease_task)
        patient_list = dataset_creator.get_patient_list()
        predicted_values = np.zeros(0)
        ground_thought = np.zeros(0)
        for id in patient_list:
            train_dataset, test_dataset, train_ground_thought, test_ground_thought = dataset_creator.get_dataset(id, test_task)
            print("Patients id: ", id, ", task for id: ", test_task, " dataset len: ", len(test_task))
            if len(test_dataset) > 0:
                temp_predicted_values = svm_model.train_test_svc(train_dataset, train_ground_thought, test_dataset)
                predicted_values = np.concatenate((predicted_values, temp_predicted_values))
                ground_thought = np.concatenate((ground_thought, test_ground_thought))
        print("Predicted values counter: ", Counter(predicted_values).items())
        print("Ground thought counter: ", Counter(ground_thought).items())
        return SVCModel.evaluate_results(predicted_values, ground_thought)
