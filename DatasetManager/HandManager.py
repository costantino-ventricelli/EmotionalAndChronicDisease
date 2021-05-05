# coding=utf-8

import csv
import os
import re
from copy import deepcopy

import numpy as np
import pandas
from numpy import ceil

RESOURCE_DIRECTORY = "resource"


class HandManager:

    def __init__(self, dataset_name):
        HandManager.set_root_directory()
        self.__dataset = os.path.join(RESOURCE_DIRECTORY, dataset_name)
        self.__dataset_directory = HandManager.get_path_directories(self.__dataset)
        self.__patient_paths = HandManager.__get_patient_paths(self.__dataset_directory)
        self.__files_path = HandManager.__get_files_from_paths(self.__patient_paths)

    @staticmethod
    def set_root_directory():
        working_dir = os.path.dirname(os.path.abspath('__file__'))
        if os.path.basename(working_dir) == 'src':
            os.chdir(os.path.dirname(working_dir))

    def get_dataset_directory(self): return self.__dataset_directory

    def get_patient_paths(self): return self.__patient_paths

    def get_files_path(self): return self.__files_path

    @staticmethod
    def __join_directory(split_dir):
        path = os.sep
        for slice in split_dir:
            path = os.path.join(path, slice)
        return path

    """
        @:param id: contiene l'id da verificare.
        @:healthy_id: contiene la lista degli id di cui verificare l'appartenenza.
        @:return: 0 se l'id appartiene alla lista di utenti sani
                  1 se l'id appartiene alla lista di utenti malati.
    """
    @staticmethod
    def get_state_from_id(id):
        healthy, disease = HandManager.get_healthy_disease_list()
        if id in healthy:
            state = 0
        else:
            state = 1
        return state

    """
        @:param main_directory_path: contiene il path per la directory generale del dataset
        @:return: restituisce tutte le directory presenti nel dataset
    """
    @staticmethod
    def get_path_directories(main_directory_path):
        directories = []
        try:
            # Itero su tutte le cartelle contenute nella directory principale del dataset.
            for directory in os.listdir(main_directory_path):
                if not (directory.endswith(".DS_Store")):
                    # Genero i percorsi per ogni directory
                    directories.append(os.path.join(main_directory_path, directory))
        except OSError as er:
            raise er
        return directories

    """
        @:param directories: Contiene la lista delle directories persenti nella cartella principale del dataset.
        @:return: restituisce i percorsi di tutti i pazienti contenuti nelle directories superiori del dataset.
    """
    @staticmethod
    def __get_patient_paths(directories):
        patients = []
        try:
            # Itero su tutte le sotto cartelle del dataset.
            for directory in directories:
                for patient in os.listdir(directory):
                    if not patient.endswith(".DS_Store"):
                        # Genero il percorso che condurrà ad ogni directory che rappresenta i pazienti nel dataset.
                        patients.append(os.path.join(directory, patient))
        except OSError as er:
            raise er
        return patients

    """
        @:param patients_directory: contiene tutti percorsi delle directories che rappresentano i pazienti nel dataset.
        @:return: restituisce tutti i percorsi del dataset con tutti i file per i tasks in esso contenuti.
    """
    @staticmethod
    def __get_files_from_paths(patients_directory):
        files = []
        try:
            # Itero su tutte le directory del dataset prelevando tutti i file di tasks con estensione txt.
            for patient in patients_directory:
                files = HandManager.get_files_from_path(patient, files)
        except OSError as er:
            raise er
        return files

    """
        @:param directory: contiene la cartella da analizzare per ottenere la lista dei file.
        @:param files: contiene la lista di file eventualmente già prelevati, se impostata a None genera e restituisce 
                la lista dei file solo della cartella indicata.
        @:return: restituisce una lista di percorsi aggiornata con i nuovi percorsi individuati nella cartella.
    """
    @staticmethod
    def get_files_from_path(directory, files):
        if files is None:
            files = []
        try:
            for file in os.listdir(directory):
                if file.endswith(".txt"):
                    files.append(os.path.join(directory, file))
        except OSError as er:
            raise er
        return files

    """
        @:param path: contiene il percorso di un file di tasks
        @:return: restituisce l'id del paziente che ha svolto quel tasks.
    """
    @staticmethod
    def get_id_from_path(path):
        # Viene effettuata la ricerca dell'espressione regolare che permetterà di prelevare l'id del paziente dal percorso.
        return int(re.search(r'_u(.*?)_', path).group(1))

    @staticmethod
    def get_ids_from_paths(paths):
        ids = []
        for path in paths:
            ids.append(HandManager.get_id_from_path(path))
        return ids

    @staticmethod
    def get_ids_from_dir(paths):
        ids = []
        for path in paths:
            split = path.split(os.sep)
            ids.append(int(split[len(split) - 1]))
        return ids

    """
        @:param path: contiene il percorso di un tasks.
        @:return: restituisce l'dentificativo del tasks che quel file rappresenta
    """
    @staticmethod
    def get_task_from_path(path):
        # Ricerca l'espressione regolare che identifica il tasks
        return re.search(r'_u(.*?)_(.*?).txt', path).group(2)

    """
        @:return: il metodo restituisce due liste di id, una per gli utenti identificati dallo screening iniziale come 
            sane e una per gli utenti identificati come malati.
    """
    @staticmethod
    def get_healthy_disease_list():
        column_names = ["ID", "DEMENTIA"]
        diagnosis_table_path = os.path.join(RESOURCE_DIRECTORY, "diagnosis_table.csv")
        # Il metodo apre il file contenente i dati per ogni utente che identificano con 0 un utente sano e con 1 un utente
        # malato.
        file = pandas.read_csv(diagnosis_table_path, sep=";", header=None, names=column_names)
        # Trasformo in due liste le informazioni contenute nel file, una per gli ids
        ids = file["ID"].tolist()
        # E una che associerà ogni id allo stato di salute.
        dementias = file["DEMENTIA"].tolist()
        healthy_ids = []
        disease_ids = []
        # Per ogni elemento delle liste verifico e smisto i gli ids nelle liste di appartenenza.
        for i in range(len(dementias)):
            healthy_ids.append(ids[i]) if dementias[i] == 0 else disease_ids.append(ids[i])
        return healthy_ids, disease_ids

    @staticmethod
    def get_all_ids():
        column_names = ["ID", "DEMENTIA"]
        diagnosis_table_path = os.path.join(RESOURCE_DIRECTORY, "diagnosis_table.csv")
        # Il metodo apre il file contenente i dati per ogni utente che identificano con 0 un utente sano e con 1 un utente
        # malato.
        file = pandas.read_csv(diagnosis_table_path, sep=";", header=None, names=column_names)
        # Trasformo in due liste le informazioni contenute nel file, una per gli ids
        ids = file["ID"].tolist()
        return ids

    """
        @:param axis: è la lista dei punti campionati per l'asse x
        @:param y_axis: è la lista dei punti campionati per l'asse y
        @:param time_stamp: è la lista dei punti campionati per il timestamp
        @:param pen_status: è la lista dei punti campionati come bottom status nella scrittura dei tasks.
        @:return: il metodo restituice una tupla di liste contentente tutte le liste di input private di tutti i punti 
            considerati come doppione.
    """
    @staticmethod
    def delete_duplicates(x_axis, y_axis, time_stamp, bottom_status):
        # Genero le nuove liste per la restituzione dei dati
        x_axis_new = []
        y_axis_new = []
        time_stamp_new = []
        bottom_status_new = []
        # Itero per tutti i campioni nelle liste e considero doppione ognuno dei campioni che presenta un timestamp
        # identico a quello del campione precedente.
        for i in range(1, len(x_axis)):
            if time_stamp[i] != time_stamp[i - 1]:
                x_axis_new.append(x_axis[i])
                y_axis_new.append(y_axis[i])
                time_stamp_new.append(time_stamp[i])
                bottom_status_new.append(bottom_status[i])
        return x_axis_new, y_axis_new, time_stamp_new, bottom_status_new

    """
        @:param path: percorso del file che il metodo analizza.
        @:return: restituisco il numero di righe che compongono il file.
    """
    @staticmethod
    def get_file_rows(path):
        with open(path, 'r') as file:
            length = len(list(file))
            file.close()
        return length

    """
        @:param paths: contiene la lista dei percorsi che indicano i file da filtrare.
        @:param min_dim: indica il numero minimo di righe che il file deve avere.
        @:return: restituisce una nuova lista di file filtrata in base al numero minimo di righe.
    """
    @staticmethod
    def filter_file(paths, min_dim):
        min_dim += 1
        filtered_paths = []
        for path in paths:
            if HandManager.get_file_rows(path) >= min_dim:
                filtered_paths.append(path)
        return filtered_paths

    """
        Il metodo restituisce tutti i file appartenenti all'utente il cui id viene passato come paramentro del metodo.
    """
    @staticmethod
    def get_all_file_of_id(id, paths):
        id_files = []
        for path in paths:
            if id == HandManager.get_id_from_path(path):
                id_files.append(path)
        return id_files

    """
        Questo metodo permette di ottenere una lista di file appartenenti ad una lista di id e ad una lista associata di 
        task.
        ex: passiamo ids = [2, 10, 11] e task = ['cdt', 'mom', 'm1'], il metodo restituirà tutti i file che appartengono
        agli utenti specificati e rappresentanti quei file.
    """
    @staticmethod
    def get_all_files_ids_tasks(ids, tasks, paths):
        files = []
        if not isinstance(ids, list):
            ids = [ids]
        if not isinstance(tasks, list):
            tasks = [tasks]
        for path in paths:
            if HandManager.get_id_from_path(path) in ids and ('_' + HandManager.get_task_from_path(path) + '.') in tasks:
                files.append(path)
        return files

    """
        Questo metodo permette di eliminare tutti i file appartenenti ad un utente identificato attraverso l'id da una
        lista di file passata come parametro
        @:param id: è l'id dell'utente di cui eliminare i file.
        @:param file_list: è la lista dei file da cui bisogna eliminare i file.
    """
    @staticmethod
    def delete_files(id, file_list):
        new_list = deepcopy(file_list)
        to_delete = HandManager.get_all_file_of_id(id, file_list)
        for file in to_delete:
            del new_list[new_list.index(file)]
        return new_list

    """
        Questo metodo calcola il 20% del numero dei file sia di quelli sani che di quelli malati e dopo di che restituisce 
        il minore tra i due.
    """
    @staticmethod
    def get_validation_number(len_healthy, len_disease):
        healthy_validation = int(ceil(len_healthy * 0.20))
        disease_validation = int(ceil(len_disease * 0.20))
        return healthy_validation if healthy_validation < disease_validation else disease_validation

    @staticmethod
    def balance_dataset(healthy_dataset, disease_dataset):
        if len(healthy_dataset) < len(disease_dataset):
            end_point = len(healthy_dataset)
        else:
            end_point = len(disease_dataset)
        return healthy_dataset[0: end_point], disease_dataset[0: end_point]

    @staticmethod
    def get_ids_age(age_min, age_max):
        healthy_ids = []
        mild_ids = []
        low_severity_ids = []
        with open(os.path.join(RESOURCE_DIRECTORY, "clean_filtered_diagnosis.csv"), 'r') as file:
            csv_file = csv.reader(file, delimiter=';')
            header = {'ID': 0, 'ETA': 1, 'DEMENZA': 2, 'TIPO/NOTE': 3}
            for row in csv_file:
                row = np.array(row).astype(int)
                id = row[header.get('ID')]
                if age_min <= row[header.get('ETA')] <= age_max:
                    if row[header.get('TIPO/NOTE')] == 0:
                        healthy_ids.append(id)
                    elif row[header.get('TIPO/NOTE')] == 1:
                        mild_ids.append(id)
                    else:
                        low_severity_ids.append(id)
            file.close()
        end_point = min(len(healthy_ids), len(mild_ids), len(low_severity_ids))
        return healthy_ids[0: end_point], mild_ids[0: end_point], low_severity_ids[0: end_point]

    @staticmethod
    def get_dict_dataset(ids, ground_thought, paths):
        from DatasetManager.Costants import TASKS
        from DeepLearningClassifier import RHSDistanceExtract
        dictionary = {}
        for id in ids:
            task_dict = {}
            file_for_id = HandManager.get_all_files_ids_tasks(id, TASKS, paths)
            for file in file_for_id:
                x_point, y_point, _ = RHSDistanceExtract.read_samples_from_file(file)
                task_dict["_" + HandManager.get_task_from_path(file) + "."] = np.reshape(np.column_stack((np.array(x_point).astype(np.float32)
                                                                                                          , np.array(y_point).astype(np.float32))), (len(x_point), 2))
            dictionary[id] = {'ground_thought': ground_thought[ids.index(id)],
                              'tasks': task_dict}
        return dictionary