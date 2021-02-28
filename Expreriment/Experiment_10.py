# coding=utf-8

import numpy as np

from sklearn.model_selection import LeaveOneOut
from ShallowLearningClassifier import SVCModel


class CompleteTaskTest:

    # FIXME: adeguare al nuovo SVC

    @staticmethod
    def start_experiment(healthy_task=None, disease_task=None):
        svm_model = SVCModel(healthy_task, disease_task)
        dataset = svm_model.get_dataset()
        ground_thought = svm_model.get_ground_thought()
        leave_one_out = LeaveOneOut()
        predicted_values = np.zeros(0)
        for train, test in leave_one_out.split(dataset):
            predicted_value = svm_model.train_test_svc(train, test)
            predicted_values = np.append(predicted_values, predicted_value)
        return SVCModel.evaluate_results(predicted_values, ground_thought)
