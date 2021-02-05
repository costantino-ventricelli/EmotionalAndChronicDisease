# coding=utf-8

import numpy as np
import os
import csv

from DatasetManager import FileManager
from DatasetManager.Costants import *
from DeepLearningClassifier import *

METRICS_KEY = 0
HEALTHY_VALUE = 1
DISEASE_VALUE = 2
FEATURE = 3

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"


class CategoryTaskExtraction:

    """
        @:param prev_file: contiene il percorso del file frutto di un esecuzione precedente, nel momento in cui viene
                passato None il processo assume che è la prima volta che viene avviata l'analisi
        @:param tasks: contiene la lista di task selezionata.
        @:param samples_len: contiene la lunghezza di ogni campione.
        @:param minumum_samples: contiene il numero minimo di esempi che devono essere presenti nel file.
        @:param category: contiene il nome della categoria che si sta selezionando, il quale nel caso di prima selezione
                            diverrà il nome del file di salvataggio.
    """
    def __init__(self, prev_file, tasks, minimum_samples, samples_len, category):
        FileManager.set_root_directory()
        self.__tasks = tasks
        # Il file contente i valori calcolati ci permette di ricominciare l'esecuzione del programma nel caso dovesse
        # interrompersi per qualsiasi ragione.
        if prev_file is not None:
            self.__prev_file = prev_file
            self.__results = self.__set_previous_state()
        else:
            self.__prev_file = prev_file
            self.__results = {}
        self.__minimum_samples = minimum_samples
        self.__samples_len = samples_len
        self.__feature_extractor = RHSDistanceExtract(self.__minimum_samples, self.__samples_len)
        self.__category = category
        # Ottengo l'intero dataset da FileManager
        self.__file_manager = FileManager("Dataset")

    """
        Questo metodo avvia la selezione dei task per gli utenti etichettati come sani, per farlo richiama il metodo 
        che permetterà di selezionare i task per gli utenti etichettati come non sani tante volte quanti sono i task 
        che si è deciso di analizzare.
    """
    def start_selection(self):
        print("Starting selection on tasks: ", self.__tasks)
        for healthy_task in self.__tasks:
            self.__start_healthy_tasks(healthy_task)

    """
        In questo metodo ogni task selezionato per gli utenti sani viene combianto con tutti i task rimanenti.
        I quali vengono utilizzati per selezionare campioni rappresententanti gli utenti malati.
    """
    def __start_healthy_tasks(self, healthy_task):
        for disease_task in self.__tasks:
            result = self.__start_disease_tasks(healthy_task, disease_task)
            if result is not None:
                print("Execution ended: ", result.items())
            else:
                print("Combination already tried")

    """
        In questo metodo effettua una serie di controlli sui task prima di avviare la modellazione della rete con i task
        risultati idonei alla modellazione, in caso di successo il metodo restituisce un dizionario contentente come chiave
        le metriche calcolate per la modellazione e come valore la lista dei task che hanno generato quei valori.
    """
    def __start_disease_tasks(self, healthy_task, disease_task):
        # Prima di tutto verifico che i task selezionati siano differenti.
        if healthy_task != disease_task:
            # A questo punto con l'ausilio del file passato al costruttore escudo tutte le combinazioni che sono già
            # state provate.
            if not self.__is_already_do(healthy_task, disease_task):
                # Se la combinazione è nuova allora viene avviata la modellazione leave_one_out che restituirà le quattro
                # metriche che ci permetteranno di valutare i risultati.
                accuracy, precision, recall, f_score = self.__leave_one_out(healthy_task, disease_task)
                print("Model evaluated")
                # A questo punto vengono eseguiti i passaggi necessari alla creazione della directory di salvataggio e
                # viene generato un nuovo file nel caso di prima esecuzione del codice, oppure vine aggiornato il file
                # esisitente.
                directory = os.path.join("experiment_result", "category_selection")
                if not os.path.exists(directory):
                    os.mkdir(directory)
                if self.__prev_file is None:
                    self.__prev_file = os.path.join(directory, self.__category + ".txt")
                    file = open(self.__prev_file, 'w')

                else:
                    file = open(self.__prev_file, 'a')
                csv_file = csv.writer(file, delimiter=';')
                csv_file.writerow([(accuracy, precision, recall, f_score), healthy_task, disease_task])
                file.close()
                return {(accuracy, precision, recall, f_score): [healthy_task, disease_task]}
        return None

    """
        Questo metodo si occupa di effettuare una sperimentazione leave_one_out sull'intero insieme di utenti.
        Ovvero per ogni utente viene effettuata una modellazione utilizzando tutti gli utenti in fase di addestramento 
        apparte uno che verrà utilizzato per il test, e questa operazione verrà ripetuta per ogni utente del dataset.
    """
    def __leave_one_out(self, healthy_task, disease_task):
        # Seleziono inizialmente tutti gli utenti distinguendoli tra sani e malati.
        healthy_ids, disease_ids = FileManager.get_healthy_disease_list()
        # Recupero tutti gli utenti in un unica lista
        test_ids = FileManager.get_all_ids()
        # Ottengo tutti i file degli utenti sani
        healthy_file = FileManager.get_all_files_ids_tasks(healthy_ids, healthy_task, self.__file_manager.get_files_path())
        # Ottengo tutti i file degli utenti malati
        disease_file = FileManager.get_all_files_ids_tasks(disease_ids, disease_task, self.__file_manager.get_files_path())
        # Elimino tutti i file che non raggiungono il numero minimo di righe stabilito.
        healthy_file = FileManager.filter_file(healthy_file, self.__minimum_samples)
        disease_file = FileManager.filter_file(disease_file, self.__minimum_samples)
        # Calcolo il mumero di file che sarranno necessari per la validazione.
        validation_number = FileManager.get_validation_number(len(healthy_file), len(disease_file))
        # Ottengo la lista dei task per il test come differenza tra i task per gli utenti sani e malati.
        test_tasks = TaskManager.get_tasks_difference(TASKS, healthy_task, disease_task)
        predicted_states = np.zeros(0)
        theoretical_states = np.zeros(0)
        # Per ogni utente avvio il training, la validation e il test del modello.
        for test_id in test_ids:
            # In questo punto elimino i file appartenenti all'utente selezionato per il test.
            healthy_file_deleted = FileManager.delete_files(test_id, healthy_file)
            disease_file_deleted = FileManager.delete_files(test_id, disease_file)
            # Genero il tensorre di training assieme alla lista degli stati.
            training_tensor, training_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[0: len(healthy_file_deleted) - validation_number] + disease_file_deleted[0: len(disease_file_deleted)])
            # Genero il tensore di validazione assiame alla lista degli stati.
            validation_tensor, validation_states, _ = self.__feature_extractor.extract_rhs_known_state(
                healthy_file_deleted[len(healthy_file_deleted) - validation_number: len(healthy_file_deleted)]
                + disease_file_deleted[len(disease_file_deleted) - validation_number: len(disease_file_deleted)])
            # Ottengo la lista dei file con cui eseguirò i test.
            test_file = FileManager.get_all_files_ids_tasks(test_id, test_tasks, self.__file_manager.get_files_path())
            # Filtro i file per lunghezza.
            test_file = FileManager.filter_file(test_file, self.__minimum_samples)
            test_tensor = np.zeros((0, self.__samples_len * 2, FEATURE))
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
                print("Validation tensor: ", np.shape(validation_tensor), " validation states: ", np.shape(validation_states))
                print("Test tensor: ", np.shape(test_tensor), " test states; ", np.shape(test_states))
                print("Validation number: ", validation_number)
                print("Healthy file: ", len(healthy_file_deleted), " Disease file: ", len(disease_file_deleted),
                      " Test id: ", test_id)
        # Restituisco le quattro metriche sulla base dei risultati conseguiti.
        return MLModel.evaluate_results(predicted_states, theoretical_states)

    """
        Questo metodo verifica se una combinazione di task è già stata provata, verificando la sua esistenza nel diziona-
        rio di istanza contenente i risultati della valutazione.
    """
    def __is_already_do(self, healthy_task, disease_task):
        for _, items in self.__results.items():
            for item in items:
                if item[HEALTHY_STRING].strip() == healthy_task.strip() and item[DISEASE_STRING].strip() == disease_task.strip():
                    return True
        return False

    """
        Questo metodo ha il compito di reimpostare i valori del dizionaro sulla base del file passato al costruttore.
    """
    def __set_previous_state(self):
        with open(self.__prev_file, 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            result = {}
            for row in csv_file:
                key = eval(row[METRICS_KEY])
                value_dict = {HEALTHY_STRING: row[HEALTHY_VALUE], DISEASE_STRING: row[DISEASE_VALUE]}
                if key in result.keys():
                    list_of_dict = result.get(key)
                    list_of_dict.append(value_dict)
                else:
                    result[key] = [value_dict]
            file.close()
        return result
