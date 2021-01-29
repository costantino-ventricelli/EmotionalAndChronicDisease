# coding=utf-8

import csv
from os import path as pt

from DatasetManager import Costants
from DatasetManager.FileManager import FileManager

RESOURCE_DIRECTORY = "resource"


class TaskManager:

    @staticmethod
    def __split(training_numbers, validation_numbers):
        # Ottengo le liste degli id dei pazienti sani e malati.
        healthy_id, diseased_id = FileManager.get_healthy_disease_list()
        # h => healthy, d => diseased
        # Separo il dataset tra training src e validazione a seconda della condizione dei pazienti.
        h_ids_training = healthy_id[0: training_numbers]
        h_ids_validation = healthy_id[training_numbers: training_numbers + validation_numbers]
        h_ids_test = healthy_id[training_numbers + validation_numbers: len(healthy_id)]
        d_ids_training = diseased_id[0:training_numbers]
        d_ids_validation = diseased_id[training_numbers: training_numbers + validation_numbers]
        d_ids_test = diseased_id[training_numbers + validation_numbers: len(diseased_id)]
        return h_ids_training, h_ids_validation, h_ids_test, d_ids_training, d_ids_validation, d_ids_test

    """
        Questo metodo permette di ottenere una serie di liste contenti i percorsi che porterranno ad una serie di file 
        che ci permetteranno di ottenere un dataset per la generazione del modello di ML, ogni lista individua una fase
        del processo di validazone(training, validation, src) individuando inoltre i file che verranno utilizzati per 
        la caratterizzazione delle due class di pazienti: malato e sano.
        @:param paths: contiene i percorsi di tutti i file contenenti i valori campionati per tutti i tasks nel dataset
        @:param healthy_task: contiene una lista di tutti i tasks che sono stati selezionati per addestrare e validare il
                                modello per i pazienti sani.
        @:param diseased_task: contiene una lista di tutti i tasks che sono stai selezionati per addestrare e validare il
                                modello per i pazienti malati.
        @:param test_task: contiene una lista dei tasks che sono stati selezionati per testare il modello.
        @:return: restituisce 6 liste contenti i vari percorsi dei file che sono stati selezionati per tasks e per classe 
                    del paziente.
    """
    @staticmethod
    def split(paths, healthy_task, diseased_task, test_task, training_number, validation_number):
        # Ottengo le liste degli id dei pazienti selezionati per eseguire le varie di modellazione, distinguendo tra id
        # di pazienti sani e malati.
        listh_training, listh_validation, listh_test, listd_training, listd_validation, listd_test = TaskManager.__split(training_number, validation_number)
        training_list_diseased = []
        training_list_healthy = []
        test_list_healthy = []
        test_list_diseased = []
        validation_list_diseased = []
        validation_list_healthy = []
        # In questo for vengono individuati dati path del sistema tutti i tasks che sono stati selezionati per la modellazione
        # se il tasks si identifica come uno dei tasks richiesti allora si verifica l'id a cui il tasks appartiene per essere
        # correttamente smistato nella lista di appartenenza corretta.
        for path in paths:
            id = FileManager.get_id_from_path(path)
            task = "_" + FileManager.get_task_from_path(path) + "."
            # Verifico se il tasks nel path è uno di quelli selezionati per i pazienti con malattia.
            if task in diseased_task:
                # Verifico se l'id del paziente è presente nella lista dei pazienti con malattia.
                training_list_diseased, validation_list_diseased = TaskManager.__id_in_list(
                    id, path, listd_training, listd_validation, training_list_diseased, validation_list_diseased)
            # In questo if si verificano i tasks per i pazienti considerati sani
            if task in healthy_task:
                training_list_healthy, validation_list_healthy = TaskManager.__id_in_list(
                    id, path, listh_training, listh_validation, training_list_healthy, validation_list_healthy)
            # In questo if si verificano i tasks selezionati per i test
            if task in test_task:
                if id in listh_test:
                    test_list_healthy.append(path)
                elif id in listd_test:
                    test_list_diseased.append(path)
        return training_list_diseased, training_list_healthy, test_list_healthy, test_list_diseased, \
            validation_list_diseased, validation_list_healthy

    """
        Questo metodo permette di ottenere i file relativi ad un tasks in particolare
    """
    @staticmethod
    def get_task_files(tasks, paths):
        task_paths = []
        try:
            if not isinstance(tasks, list):
                tasks = [tasks]
            for task in tasks:
                for path in paths:
                    if task in '_' + FileManager.get_task_from_path(path) + '.':
                        task_paths.append(path)
        except TypeError as error:
            print("Error: ", error)
            print("Task: ", tasks)
        return task_paths

    """
        Il metodo permette di scansionare una lista di percorsi e ottenere tutti i task necessari associati al loro 
        esecutore.
    """
    @staticmethod
    def get_task_from_paths(paths, tasks):
        ids_task = {}
        for path in paths:
            id = FileManager.get_id_from_path(path)
            task = "_" + FileManager.get_task_from_path(path) + "."
            if task in tasks:
                if id in ids_task:
                    temp_path = ids_task[id]
                    temp_path.append(path)
                    ids_task.update({id: temp_path})
                else:
                    ids_task[id] = [path]
        return ids_task

    """
        @:param training_item: indica il numero di file che verranno effettivamente utilizzati per l'addestramento del modello.
        @:param validation_item: indica il numero di file che verranno effettivamente utilizzati per la validazionde del modello.
        @:param test_item: indica il numero di file che verranno effettivamente utilizzati per il src del modello.
        @:param minimum_row: indica il nuemero minimo di righe che devono essere presenti nel file per considerarlo 
                                valido per il modello.
        @:param ml_model: è una lista di liste, dove ogni lista è una delle sei liste individuate da "split()":
            ml_model[training_list_diseased[path,path,...], 
                    training_list_healthy[path,path,...],
                    test_list_disease[path,path,...],
                    test_list_healthy[path,path,...],
                    validation_list_diseased[path,path,...],
                    validation_list_healthy[path,path,...]].
        @:return: restituisce semple ml_model dopo aver elimiato tutto ciò che non va bene per la creazione del modello.
    """
    @staticmethod
    def check_file_dimension(training_item, validation_item, test_item, minimum_row, ml_model):
        working_model = list.copy(ml_model)
        ml_model = []
        # Viene valutato il numero di righe di ogni file selezionato per il modello ml, se la lunghezza non è almeno
        # quella indicata come minima il sistema scarta quel file dal modello.
        for paths in working_model:
            new_paths = []
            for path in paths:
                with open(path, newline='') as csv_file:
                    if len(list(csv.reader(csv_file, delimiter=' '))) > minimum_row:
                        new_paths.append(path)
                    csv_file.close()
            ml_model.append(new_paths)
        # Creo un vettore per poter selezionare più agevolmente il mumero di file effettivamente necessari al modello.
        items_number = [training_item, test_item, validation_item]
        # Restringo il numero dei file per ogni fase della creazione del modello, sui valori parametrizzati passati per
        # effettuare l'esperimento.
        for i in range(len(items_number)):
            for j in range(i, i + 2):
                ml_model[i + j] = ml_model[i + j][0:items_number[i]]
        # In questo for vengono salvati vari file il cui contenuto rispecchierà il dataset generato per la creazione del
        # modello di ML.
        for i in range(len(ml_model)):
            file_path = pt.join(RESOURCE_DIRECTORY, "model_paths", Costants.FILE_NAME[i])
            with open(file_path, 'w') as file:
                for path in ml_model[i]:
                    file.write(path + "\n")
            file.close()
        return ml_model[0], ml_model[1], ml_model[2], ml_model[3], ml_model[4], ml_model[5]

    """
        @:param id: contiene l'id del paziente selezionato.
        @:param path: contiente il percorso del file selezionato.
        @:param healthy_list: contiene la lista degli id dei pazienti sani.
        @:param diseased_list: contiene la lista delgli id dei pazienti malati.
        @:healthy_path_list: contiene la lista dei percorsi per i pazienti sani.
        @:diseased_path_list: contiene la lista dei percorsi per i pazienti malati
    """
    @staticmethod
    def __id_in_list(id, path, training_list, validation_list, healthy_path_list, diseased_path_list):
        if id in training_list:
            healthy_path_list.append(path)
        elif id in validation_list:
            diseased_path_list.append(path)
        return healthy_path_list, diseased_path_list
