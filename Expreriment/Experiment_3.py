# coding=utf-8

import concurrent.futures as features
import csv
import os

import numpy as np

from DatasetManager import HandManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *

KEY_TUPLE = 0
VALUE_TUPLE = 1
FEATURES = 3


class IndependentTaskSelection:

    def __init__(self, samples_len, minimum_samples):
        self.__tasks = TASKS
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__best_results = {}
        self.__file_manager = HandManager("Dataset")
        self.__feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        self.__ids = HandManager.get_all_ids()
        healthy_ids, disease_ids = HandManager.get_healthy_disease_list()
        directory = os.path.join('experiment_result', 'experiment_3')
        # Saveremo i risultati delle misurazioni in un file così sarà pià facile recuperarli in futuro
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(os.path.join(directory, 'init_file.csv'), 'w') as file:
            csv_file = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            csv_file.writerow(['METRICS', 'HEALTHY TASK', 'DISEASE TASK'])
            file.close()
        # In questo ciclo verranno selezionati tutti i task che verranno ad uno ad uno valutati per addestrare il sistema
        # attribuendoli agli utenti sani
        for healthy_task in self.__tasks:
            healthy_paths = HandManager.get_all_files_ids_tasks(healthy_ids, healthy_task,
                                                                self.__file_manager.get_files_path())
            healthy_paths = HandManager.filter_file(healthy_paths, self.__minimum_samples)
            # Calcolo in base al numero di file rimasti quelli sufficienti per la validazione che abbiamo fissato al 20%
            # del totale.
            healthy_validation = int(np.ceil(len(healthy_paths) * 0.20))
            # Questo ciclo permette di selezionare e valutare ogni task per addestrare il modello attribuendoli agli
            # utenti etichettati come malati.
            for disease_task in self.__tasks:
                if healthy_task != disease_task:
                    disease_paths = HandManager.get_all_files_ids_tasks(disease_ids, disease_task,
                                                                        self.__file_manager.get_files_path())
                    disease_paths = HandManager.filter_file(disease_paths, self.__minimum_samples)
                    # Qui vengono ottenuti tutti i task che non sono stati selezionati per l'addestramento così da
                    # utilizzarli per il test
                    test_tasks = TaskManager.get_tasks_difference(self.__tasks, healthy_task,
                                                                  disease_task)
                    # Azzero i vettori che mi permetteranno di valutare i risultati del sistema.
                    predicted_status = np.zeros(0)
                    theoretical_status = np.zeros(0)
                    # Calcolo in base al numero di file rimasti quelli sufficienti per la validazione che abbiamo fissato al 20%
                    # del totale.
                    disease_validation = int(np.ceil(len(disease_paths) * 0.20))
                    # Questo ciclo ci permette di effettuare il test su tutti gli utenti del dataset, in pratica per ogni
                    # coppia di healthy_task viene effettuato un addestramento di tipo leave-one-out e il sistema è valutato sull'
                    # interezza dei risultati.
                    # Qui individuo quale è il numero effettivo di file utili per la validazione e lo scelgo in
                    # base a quale è il minore tra le due categorie di utenti
                    if disease_validation < healthy_validation:
                        validation = disease_validation
                    else:
                        validation = healthy_validation
                    with features.ThreadPoolExecutor() as executor:
                        features_completed = {executor.submit(IndependentTaskSelection.__execute_id_analysis, disease_paths,
                                                              healthy_paths, validation, test_id, test_tasks, self.__file_manager,
                                                              self.__minimum_samples, self.__feature_extraction, self.__samples_len): test_id for test_id in
                                              self.__ids}
                        for feature in features.as_completed(features_completed):
                            predicted_status_partial, theoretical_status_partial = feature.result()
                            predicted_status = np.concatenate(predicted_status, predicted_status_partial)
                            theoretical_status = np.concatenate(theoretical_status, theoretical_status_partial)
                    accuracy, precision, recall, f_score = MLModel.evaluate_results(predicted_status, theoretical_status)
                    # Salvo i risultati nel file.
                    with open(os.path.join(directory, 'init_file.csv'), 'a') as file:
                        csv_file = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                        csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
                        file.close()

    @staticmethod
    def __execute_id_analysis(disease_paths, healthy_paths, validation, test_id, test_tasks,
                              file_manager, minimum_samples, features_extraction, samples_len):
        # Questo punto del codice ci permette di eliminare tutti i file dell'utente che andemo a testare.
        healthy_paths_deleted = HandManager.delete_files(test_id, healthy_paths)
        disease_paths_deleted = HandManager.delete_files(test_id, disease_paths)
        # Ottengo tutti i percorsi dei file che verranno utilizzati per il test
        test_paths = HandManager.get_all_files_ids_tasks(test_id, list(test_tasks),
                                                         file_manager.get_files_path())
        test_paths = HandManager.filter_file(test_paths, minimum_samples)
        # Se riesco a individuare file di test che rispettano i parametri necessari per la costruzone del
        # tensore di test avvio apprendimento, validazione e test.
        predicted_status_partial = np.zeros(0)
        test_states = np.zeros(0)
        if len(test_paths) > 0:
            # Qui ottengo i tensori per training, validation e test con i rispettivi vettori delle eti-
            # chette di stato.
            training_tensor, training_states, validation_tensor, validation_states, test_tensor, test_states = \
                IndependentTaskSelection.__extract_rhs_segment(validation, healthy_paths_deleted, disease_paths_deleted, test_paths, features_extraction, samples_len)
            # Il stitema viene addestrato validato e testato, i risultati del test vengono messi nel vettore
            # generale che riporterà i rislutati complessivi del leave-one-out.
            try:
                ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                print("Created model for id: ", test_id)
                predicted_status_partial, _, _ = ml_model.test_model(test_tensor, test_states)
            except ValueError as error:
                print("Exception raise: ", error)
                print("Training tensor: ", np.shape(training_tensor), ", training states: ",
                      np.shape(training_states))
                print("Validation tensor: ", np.shape(validation_tensor), ", validation states: ",
                      np.shape(validation_states))
                print("Validation file number: ", validation)
        else:
            print("There are not test file for id: ", test_id)
        return predicted_status_partial, test_states

    """
        Questo metodo permette di eliminare tutti i file appartenenti ad un utente identificato attraverso l'id da una
        lista di file passata come parametro
        @:param id: è l'id dell'utente di cui eliminare i file.
        @:param file_list: è la lista dei file da cui bisogna eliminare i file.
    """
    @staticmethod
    def __delete_files(id, file_list):
        to_delete = HandManager.get_all_file_of_id(id, file_list)
        for file in to_delete:
            del file_list[file_list.index(file)]
        return file_list

    """
        Questo metodo permette di estrarre tutti i tensori e i vettori di stato necessari alla costruzione del modello.
    """
    @staticmethod
    def __extract_rhs_segment(validation, healthy_paths, disease_paths, test_paths, feature_extraction, samples_len):
        training_paths = healthy_paths[0: validation] + disease_paths[0: validation]
        validation_paths = healthy_paths[validation: len(healthy_paths)] + disease_paths[validation: len(disease_paths)]
        training_tensor, training_states, _ = feature_extraction.extract_rtp_known_state(training_paths)
        validation_tensor, validation_states, _ = feature_extraction.extract_rtp_known_state(validation_paths)
        test_tensor = np.zeros((0, samples_len * 2, FEATURES))
        test_states = np.zeros(0)
        for path in test_paths:
            partial_tensor, partial_states = feature_extraction.rtp(path)
            test_tensor = np.concatenate((test_tensor, partial_tensor))
            test_states = np.concatenate((test_states, partial_states))
        return training_tensor, training_states, validation_tensor, validation_states, test_tensor, test_states
