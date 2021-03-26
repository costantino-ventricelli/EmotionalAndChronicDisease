# coding=utf-8

from collections import Counter

import numpy as np

from ShallowLearningClassifier import CreateDictDataset
from ShallowLearningClassifier import SVCModel


class ShallowLeaveOneOut:

    @staticmethod
    def start_experiment(healthy_task=None, disease_task=None, test_task=None):
        # Genero il modello a vettori di supporto selezionando le feature migliori e gli iperparametri migliori per il
        # modello di shallow learning
        svm_model = SVCModel(healthy_task, disease_task)
        # Una volta fatto il tuning del modello viene generato il dizionario contente il dataset.
        dataset_creator = CreateDictDataset(svm_model.get_feature_selected(), healthy_task, disease_task)
        # Lista dei pazienti.
        patient_list = dataset_creator.get_patient_list()
        predicted_values = np.zeros(0)
        ground_thought = np.zeros(0)
        # Per ogni paziente vengono generati i dataset adatti all'addestramento e al test del modello.
        for id in patient_list:
            train_dataset, test_dataset, train_ground_thought, test_ground_thought, test_task = dataset_creator.get_dataset(id, test_task)
            print("Patients id: ", id, ", task for id: ", test_task, " dataset length: ", len(test_task))
            if len(test_dataset) > 0:
                temp_predicted_values = svm_model.train_test_svc(train_dataset, train_ground_thought, test_dataset)
                predicted_values = np.concatenate((predicted_values, temp_predicted_values))
                ground_thought = np.concatenate((ground_thought, test_ground_thought))
        print("Predicted values counter: ", Counter(predicted_values).items())
        print("Ground thought counter: ", Counter(ground_thought).items())
        return SVCModel.evaluate_results(predicted_values, ground_thought)
