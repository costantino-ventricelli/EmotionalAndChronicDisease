# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as get_four_metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from .FeaturesSelection import FeatureSelection


class SVCModel:

    """
        Il metodo di inizializzazione di questa classe avvia un'ottimizzazione sugli iperparametri per costriuire il SVC,
        che prendendo in input il dataset con i task passati al costruttore permette di individuare le feature
        migliori per costruire il classificatore.
    """
    def __init__(self, healthy_task=None, disease_task=None):
        feature_selection = FeatureSelection(healthy_task, disease_task)
        _, self.__feature_selected = feature_selection.select_feature()
        dataframe = feature_selection.get_dataframe()
        print("Getting best feature data")
        self.__dataframe = self.__get_dataframe(dataframe)
        self.__random_gird = {
            'C': np.linspace(start=0.1, stop=100, num=20),
            'gamma': ['auto', 'scale', 1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        svc = SVC()
        self.__dataset = dataframe.to_numpy()
        self.__ground_thought = feature_selection.get_ground_thought()
        print("Tuning hyperparameters")
        hyperparameters = RandomizedSearchCV(estimator=svc, param_distributions=self.__random_gird, n_iter=100, cv=3,
                                             verbose=3, n_jobs=-1)
        hyperparameters.fit(self.__dataset, self.__ground_thought)
        self.__best_hyperparameters = hyperparameters.best_params_

    def get_feature_selected(self):
        return self.__feature_selected

    """
        Questo metodo permette di utilizzare le fearure selezionate dal random forest per addestrare un SVC il quale a sua
        volta utilizza gli iperparametri individuati dal costruttore, il metodo quindi avvia l'addestramento del classificatore
        e avvia il test sul dataset restituendo le 4 metriche.
    """
    def train_test_svc(self, train_dataset, train_ground_thought, test_dataset):
        support_vector_classifier = SVC(C=self.__best_hyperparameters.get('C'), kernel=self.__best_hyperparameters.get('kernel'),
                                        gamma=self.__best_hyperparameters.get('gamma'), verbose=True, max_iter=-1)
        print("Training model")
        support_vector_classifier.fit(train_dataset, train_ground_thought)
        print("Testing model shape: ", np.shape(test_dataset))
        return np.array(support_vector_classifier.predict(test_dataset))

    @staticmethod
    def evaluate_results(predicted_values, ground_thought):
        accuracy = accuracy_score(ground_thought, predicted_values)
        precision, recall, f_score, _ = get_four_metrics(ground_thought, predicted_values, labels=[0, 1], average='macro')
        return accuracy, precision, recall, f_score

    def __get_dataframe(self, dataframe):
        new_dataframe = {}
        dataframe_dict = dataframe.to_dict(orient='dict', into=dict)
        for feature, value in dataframe_dict.items():
            if feature in self.__feature_selected:
                new_dataframe[feature] = value
        return pd.DataFrame().from_dict(data=new_dataframe, orient='columns')
