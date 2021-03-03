# coding=utf-8

from DatasetManager import HandManager
from DatasetManager.Costants import *
from .MachineLearningModel import MLModel
from DatasetManager.TaskManager import TaskManager

import numpy as np


class LeaveOneOut:

    def __init__(self, minimum_samples, samples_len, feature_extractor, feature, dataset):
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        self.__file_manager = HandManager(dataset)
        self.__feature = feature
        self.__feature_extractor = feature_extractor

    def leave_one_out(self, healthy_task, disease_task):
        # Seleziono inizialmente tutti gli utenti distinguendoli tra sani e malati.
        healthy_ids, disease_ids = HandManager.get_healthy_disease_list()
        # Recupero tutti gli utenti in un unica lista
        test_ids = HandManager.get_all_ids()
        # Ottengo tutti i file degli utenti sani
        healthy_file = HandManager.get_all_files_ids_tasks(healthy_ids, healthy_task,
                                                           self.__file_manager.get_files_path())
        # Ottengo tutti i file degli utenti malati
        disease_file = HandManager.get_all_files_ids_tasks(disease_ids, disease_task,
                                                           self.__file_manager.get_files_path())
        # Elimino tutti i file che non raggiungono il numero minimo di righe stabilito.
        healthy_file = HandManager.filter_file(healthy_file, self.__minimum_samples)
        disease_file = HandManager.filter_file(disease_file, self.__minimum_samples)
        # Calcolo il mumero di file che sarranno necessari per la validazione.
        validation_number = HandManager.get_validation_number(len(healthy_file), len(disease_file))
        # Ottengo la lista dei task per il test come differenza tra i task per gli utenti sani e malati.
        test_tasks = TaskManager.get_tasks_difference(TASKS, healthy_task, disease_task)
        predicted_states = np.zeros(0)
        theoretical_states = np.zeros(0)
        # Per ogni utente avvio il training, la validation e il test del modello.
        for test_id in test_ids:
            # In questo punto elimino i file appartenenti all'utente selezionato per il test.
            healthy_file_deleted = HandManager.delete_files(test_id, healthy_file)
            disease_file_deleted = HandManager.delete_files(test_id, disease_file)
            # Genero il tensorre di training assieme alla lista degli stati.
            training_tensor, training_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[0: len(healthy_file_deleted) - validation_number] + disease_file_deleted[
                                                                                         0: len(disease_file_deleted)])
            # Genero il tensore di validazione assiame alla lista degli stati.
            validation_tensor, validation_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[len(healthy_file_deleted) - validation_number: len(healthy_file_deleted)]
                + disease_file_deleted[len(disease_file_deleted) - validation_number: len(disease_file_deleted)])
            # Ottengo la lista dei file con cui eseguirò i test.
            test_file = HandManager.get_all_files_ids_tasks(test_id, test_tasks, self.__file_manager.get_files_path())
            # Filtro i file per lunghezza.
            test_file = HandManager.filter_file(test_file, self.__minimum_samples)
            test_tensor = np.zeros((0, self.__samples_len * 2, self.__feature))
            test_states = np.zeros(0)
            # In questo ciclo genero il tensore per i file di test e la lista degli stati.
            for file in test_file:
                partial_tensor, partial_states = self.__feature_extractor.extract_rhs_file(file)
                test_tensor = np.concatenate((test_tensor, partial_tensor))
                test_states = np.concatenate((test_states, partial_states))
            try:
                # Addestro e valido il modello
                ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                # Testo il modello.
                predicted_states_partial, evaluation_result, _ = ml_model.test_model(test_tensor, test_states)
                # Aggiungo i risultati ottenuti alla lista completa dei risultati
                theoretical_states = np.concatenate((theoretical_states, test_states))
                predicted_states = np.concatenate((predicted_states, predicted_states_partial))
            except ValueError as error:
                print("Error: ", error)
                print("Training tensor: ", np.shape(training_tensor), " training states: ", np.shape(training_states))
                print("Validation tensor: ", np.shape(validation_tensor), " validation states: ",
                      np.shape(validation_states))
                print("Test tensor: ", np.shape(test_tensor), " test states; ", np.shape(test_states))
                print("Validation number: ", validation_number)
                print("Healthy file: ", len(healthy_file_deleted), " Disease file: ", len(disease_file_deleted),
                      " Test id: ", test_id)
        # Restituisco le quattro metriche sulla base dei risultati conseguiti.
        return MLModel.evaluate_results(predicted_states, theoretical_states)