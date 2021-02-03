# coding=utf-8

from collections import Counter
from copy import deepcopy

import numpy as np
import os

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *

KEY_TUPLE = 0
VALUE_TUPLE = 1
FEATURES = 3


class TaskSelection:

    """
        @:param samples_len: contiene la lunghezza che avranno i campioni all'interno dei tensori
        @:param minumum_samples: contiene la lunghezza minima in termini di righe dei file da considerare come adatti
        Il metodo di init permette di generare una prima classifica dei tasks che ottengono i risultati migliori basandosi
        sulle quattro metriche messe in ordine di importanza all'interno della tupla (accuracy, precision, recall, f_score)
    """
    def __init__(self, samples_len, minimum_samples):
        # Lista dei tasks del dataset hand
        self.__tasks = TASKS
        self.__samples_len = samples_len
        self.__minimum_samples = minimum_samples
        self.__best_results = {}
        self.__file_manager = FileManager("Dataset")
        self.__feature_extraction = RHSDistanceExtract(minimum_samples, samples_len)
        # Con questo loop posso avviare apprendimenti su ogni tasks del dataset, ciò mi permetterà di individiare il tasks
        # migliore da cui iniziare la selezione.
        for task in self.__tasks:
            ids = FileManager.get_all_ids()
            predicted_states = np.zeros(0)
            theoretical_states = np.zeros(0)
            for test_id in ids:
                print("Creating model and testing for id: ", test_id, " with healthy_task: ", task)
                # Per il test vengono selezionati due utenti uno etichettato come sano e uno etichettato come malato, per ottenere
                # misuarizioni sulle prestazioni del sistema più precise ed affidabili.
                paths, test_paths, validation_number = self.__select_paths_from_tasks([task], test_id)
                if len(test_paths) > 0:
                    # Estraggo i segmenti RHS per training, validazione e test del sistema.
                    test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                        paths, test_paths, validation_number)
                    try:
                        ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                        predicted_states_partial, _, _ = ml_model.test_model(test_tensor, test_states)
                        predicted_states = np.concatenate(
                            (predicted_states, np.array(predicted_states_partial).astype(float)))
                        theoretical_states = np.concatenate((theoretical_states, np.array(test_states).astype(float)))
                    except ValueError as val:
                        print("Exception: ", val)
                else:
                    print("There are no test for user: ", test_id, " with healthy_task: ", task)
            print("Predicted results: ", Counter(predicted_states).items())
            print("Theoretical states: ", Counter(theoretical_states).items())
            accuracy, precision, recall, f_score = MLModel.evaluate_results(predicted_states, theoretical_states)
            TaskSelection.__fill_dictionary(self.__best_results, accuracy, f_score, precision, recall, task)

    """
        Questo metodo mi permette di avviare la selezione dei tasks partendo da quelli che hanno dato il miglior risultato
        nell inizializzazione.
    """
    def execute_simple_task_selection(self):
        file = open(os.path.join('experiment_result', 'log_file.txt'), 'w')
        # Selezioni i tasks migliori con i rispettivi risultati.
        previous_max = max(self.__best_results.items())
        actual_tuple = max(self.__best_results.items())
        file.write("Max tuple in init: " + str(previous_max) + "\n")
        file.close()
        # Creo il nuovo dizionario che permetterà di salvare i risultati
        best_results = {}
        # La selezione continuerà fino al deterioramento dei risultati, ovvero appena uno dei quatto paramentri si abbassa
        # la selezione viene interrotta.
        while previous_max[KEY_TUPLE] <= actual_tuple[KEY_TUPLE]:
            print("Previous value: ", previous_max[KEY_TUPLE])
            print("Actual value: ", actual_tuple[KEY_TUPLE])
            # Scansionando tutti i tasks dovrei essere in grado di aggiungere nuovi tasks alla selezione.
            for task in self.__tasks:
                best_results = self.__for_all_task(actual_tuple, best_results, task)
            # Salvo i risultati precedenti prima di aggiornare il sitema con i nuovi dati.
            actual_tuple = max(best_results.items())
            if previous_max[KEY_TUPLE] <= actual_tuple[KEY_TUPLE]:
                previous_max = deepcopy(actual_tuple)
            file = open(os.path.join('experiment_result', 'log_file.txt'), 'a')
            file.write("Previous max: " + str(previous_max) + "\n")
            file.write("Actual max: " + str(actual_tuple) + "\n")
            file.write("Results for tasks: " + str(best_results.items()) + "\n\n")
            file.close()
            best_results = {}
        return previous_max

    def __for_all_task(self, actual_tuple, best_results, task):
        # Seleziono la lista dei tasks appartenente al migliore dei risultati selezionato precedentemente.
        actual_tasks = list.copy(actual_tuple[VALUE_TUPLE])
        try:
            np.shape(actual_tasks)[1]
        except IndexError:
            actual_tasks = [actual_tasks]
        # Verifico che il tasks selezionato non sia già stato preso in analisi.
        for tasks in actual_tasks:
            print("Task: ", task)
            print("Actual tasks: ", tasks)
            if task not in tasks:
                # Aggiungo il tasks alla lista di nuovi tasks da analizzare.
                tasks.append(task)
                print("Selected tasks: ", tasks)
                file = open(os.path.join('experiment_result', 'log_file.txt'), 'a')
                file.write("Selected tasks: " + str(tasks) + "\n")
                file.close()
                ids = FileManager.get_all_ids()
                theoretical_states = np.zeros(0)
                predicted_states = np.zeros(0)
                for test_id in ids:
                    print("Creating model and testing for id: ", test_id, " with tasks: ", tasks)
                    # Seleziono i file per il modello compresi i file di test e il numero di file che verranno destinati
                    # alla validazione.
                    paths, test_paths, validation_number = self.__select_paths_from_tasks(tasks, test_id)
                    if len(test_paths) > 0:
                        # Estraggo i segmenti RHS per i tensori.
                        test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor = self.__extract_rhs_segment(
                            paths, test_paths, validation_number)
                        try:
                            ml_model = MLModel(training_tensor, training_states, validation_tensor, validation_states)
                            predicted_states_partial, _, _ = ml_model.test_model(test_tensor, test_states)
                            theoretical_states = np.concatenate((theoretical_states, np.array(test_states).astype(float)))
                            predicted_states = np.concatenate((predicted_states, np.array(predicted_states_partial).astype(float)))
                        except ValueError as error:
                            print("Exception raise: ", error)
                            print("Training tensor: ", np.shape(training_tensor), ", training states: ",
                                  np.shape(training_states))
                            print("Validation tensor: ", np.shape(validation_tensor), ", validation states: ",
                                  np.shape(validation_states))
                    else:
                        print("There are no tests for id: ", test_id, " with tasks: ", tasks)
                print("Predicted results: ", Counter(predicted_states).items())
                print("Theoretical states: ", Counter(theoretical_states).items())
                accuracy, precision, recall, f_score = MLModel.evaluate_results(predicted_states, theoretical_states)
                best_results = TaskSelection.__fill_dictionary(best_results, accuracy, precision, recall, f_score, tasks)
        return best_results

    """
        Questo metdodo permette di individuare i percorsi che verranno utilizzati per l'addestramento e la validazione 
        del modello, separandoli da quelli di test.
    """
    def __select_paths_from_tasks(self, tasks, test_id):
        paths = []
        # Raccolgo tutti i percorsi che contengono i campioni di quel tasks
        for task in tasks:
            paths += TaskManager.get_task_files(task, self.__file_manager.get_files_path())
        # Filtro i file in base alla dimensione degli stessi
        paths = FileManager.filter_file(paths, min_dim=self.__minimum_samples + 1)
        test_paths = FileManager.get_all_file_of_id(test_id, paths)
        # Elimino i file del test da quelli usati per il training e la validazione.
        for i in range(len(test_paths)):
            if test_paths[i] in paths:
                del paths[paths.index(test_paths[i])]
        # Calcolo il numero di file necessari per la validazione.
        validation_number = int(np.ceil(len(paths) * 0.2))
        return paths, test_paths, validation_number

    """
        Questo metodo permette di generare i tre tensori con i relativi targhet per addestramento, validazione e test del modello.
    """
    def __extract_rhs_segment(self, paths, test_paths, validation_number):
        # Genero il tensore di training.
        training_tensor, training_states, _ = self.__feature_extraction.extract_rhs_known_state(
            paths[0: (len(paths) - validation_number)])
        # Genero il tensore di validazione.
        validation_tensor, validation_states, _ = self.__feature_extraction.extract_rhs_known_state(
            paths[(len(paths) - validation_number): len(paths)])
        # Genero il tensore di test.
        test_tensor = np.zeros((0, self.__samples_len * 2, FEATURES), dtype=float)
        test_states = np.zeros(0)
        for path in test_paths:
            partial_tensor, partial_states = self.__feature_extraction.extract_rhs_file(path)
            test_tensor = np.concatenate((test_tensor, partial_tensor))
            test_states = np.concatenate((test_states, partial_states))
        return test_states, test_tensor, training_states, training_tensor, validation_states, validation_tensor

    """
        Questo metodo permette di aggiornare un dizionario che permetterà di tracciare i risultati ottenuti dal sistema
    """
    @staticmethod
    def __fill_dictionary(best_results, accuracy, f_score, precision, recall, task):
        if (accuracy, precision, recall, f_score) in best_results:
            tasks = best_results.get((accuracy, precision, recall, f_score), None)
            if isinstance(task, list):
                try:
                    # Questo comando serve appositamente a scatenare l'eccezione nel caso in cui ci siano più liste di
                    # healthy_task che restituiscano lo stesso risultato sulle 4 metriche.
                    np.shape(tasks)[1]
                except IndexError:
                    tasks = [tasks]
            tasks.append(task)
        else:
            best_results[(accuracy, precision, recall, f_score)] = task if isinstance(task, list) else [task]
        return best_results
