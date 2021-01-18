# coding=utf-8

"""
   @author: Costantino Ventricelli
   @date: 2020-12-10

   Questa classe si occupa della conversione del dataset Emothaw nel dataset necessario per l'applicazione di riconosci-
   mento dello stato di salute attraverso biometria. Ovvero un modello di Machine Learning adattato nell'anno accademico
   2020 per poter eseguire src su altri dataset.
"""
import csv
import os
import pathlib
import re

from DatasetManager.Costants import *

RESOURCE_DIRECTORY = "resource"
NOT_DIRECTORY = ['.xls', '.txt']
# Queste due strutture a dizionario servono a creare la corrispondenza tra il dataset Emothaw e il dataset necessiario
# per il modello di machine learning.
EMOTHAW_FILE_STRUCTURE = {
    0: 'x_axis',
    1: 'y_axis',
    2: 'timestamp',
    3: 'pen_status',
    4: 'pressure',
    5: 'azimuth',
    6: 'altitude'
}
PATHOLOGICAL_FILE_STRUCTURE = {
    'x_axis': 0,
    'y_axis': 1,
    'pressure': 2,
    'timestamp': 3,
    'azimuth': 4,
    'altitude': 5,
    'pen_status': 6
}


class ConverterEmothawToPathological:

    """
        @:param main_directory: contiene il nome della cartella in cui è presente il dataset.
        Questo metodo prendendo in input il nome della cartella che contiene il dataset, messa nella cartella resource
        del sistema, legge tutte le sotto-directory del dataset e genera un nuovo dataset che rispecchierà la struttura
        del dataset creato per il modello di machine learning.
    """
    @staticmethod
    def convert_dataset(main_directory):
        # Creo, se non esiste, la nuova root directory per il dataset.
        converted_directory = ConverterEmothawToPathological.__make_converted_directory()
        # Ottengo il percorso completo del dataset passato al metodo.
        main_directory = os.path.join(RESOURCE_DIRECTORY, main_directory)
        # Ottengo una prima lista delle cartelle contenute nel dataset, assieme ad una lista di directory che verranno
        # usate nel nuovo dataset.
        old_directories, new_directories = ConverterEmothawToPathological.__scan_directory(main_directory)
        # Creo le nuove cartelle nella root directory del nuovo dataset.
        for i in range(len(new_directories)):
            new_directories[i] = os.path.join(converted_directory, new_directories[i])
            if not os.path.exists(new_directories[i]):
                os.mkdir(new_directories[i])
        # Richiamo il metodo che genererà i nuovi file del dataset.
        ConverterEmothawToPathological.__convert_file_to_pathological(new_directories, old_directories)

    """
        @:param new_directories: contiene una lista dei percorsi contenuti nella nuova root directory.
        @:param old_directories: contiene una lista dei vecchi percorsi nella root directory.
        Questo metodo scansiona tutte le sotto directory ottenendo una matrice di percorsi, dove ogni riga contiene i
        percorsi di ogni utente presente nel dataset.
    """
    @staticmethod
    def __convert_file_to_pathological(new_directories, old_directories):
        for i in range(len(old_directories)):
            # Ottengo i percorsi del vecchio dataset e una lista di utenti necessaria per la generazione del nuovo dataset.
            old_directories[i], user_paths = ConverterEmothawToPathological.__scan_directory(old_directories[i])
            # Richiamo il metodo che estrarrà i task da ogni cartella utente e li modificherà in modo da renderli compa-
            # tibili con il dataset per il modello di ML.
            ConverterEmothawToPathological.__extract_session_from_users(new_directories[i], old_directories[i], user_paths)

    """
        @:param new_directories: contiene la lista dei percorsi del nuovo dataset.
        @:param old_directory: contiene il percorso della cartella dell'utente nel nuovo dataset.
        @:user_paths: contiene la lista degli utenti individuati nel dataset.
        Questo metodo genera nel nuovo dataset le cartelle per ogni utente individuato nel vecchio dataset, dopo di che
        scansiona le cartelle di di sessione del vecchio dataset, dalle quali estrae i task che poi saranno sottoposti a
        conversione, inoltre se la directory di sessione si rivela vuota viene eliminata anche la catella dell'utene a 
        che dovrebbe essere vuota.
    """
    @staticmethod
    def __extract_session_from_users(new_directories, old_directories, user_paths):
        # Individo la collezione a cui appartine l'utente tramite il percorso di salvataggio.
        collection = new_directories[-1]
        for j in range(len(user_paths)):
            # Ottengo i percorsi di sessione dal dataset originale.
            session, _ = ConverterEmothawToPathological.__scan_directory(old_directories[j])
            # Verifico che siano presenti delle cartelle di sessione
            if len(session) > 0:
                # Ottengo la lista dei task
                tasks = os.listdir(session[0])
                # Verifico che ci siano percorsi nella lista dei task
                if len(tasks) > 0:
                    # Ricavo l'id dell'utente del percorso
                    user_id = int(user_paths[j].replace('user', ''))
                    # Genero un nuovo percorso eliminando gli zeri iniziali dell'utente.
                    user_paths[j] = os.path.join(new_directories, str(collection) + str(user_id))
                    # Creo la cartella per l'utente.
                    if not os.path.exists(user_paths[j]):
                        os.mkdir(user_paths[j])
                    # Genero i percorsi per ogni file di task.
                    tasks = [os.path.join(session[0], task) for task in tasks]
                    # Avvio la conversione dei file di task
                    ConverterEmothawToPathological.__convert_file_emothaw_to_pathological(tasks, user_paths[j], user_id, collection)
                else:
                    # Rimuovo la directory dell'utente se non ci sono task per quell'utente.
                    print("Removing user: ", j)
                    os.rmdir(user_paths[j])

    """
        @:param paths_to_task: contiene i percorsi di tutti i task da scansionare.
        @:save_directory: contiene il percorso di salvataggio dell'utente.
        @:user_id: l'id dell'utente a cui appartiene il task.
    """
    @staticmethod
    def __convert_file_emothaw_to_pathological(paths_to_task, save_directory, user_id, collection):
        # Scansiono uno ad uno i task e avvio la conversione degli stessi per scriverli nella struttura dei file presenti
        # nel dataset Pathological.
        for task in paths_to_task:
            # Individuo il numero del task dal suo percorso.
            task_search = re.search(r'hw(.*?).svc', task)
            # Verifico che sia stato individuato il numero del task.
            if task_search:
                # Ottengo il nuovo suffisso per il task individuato.
                task_suffix = ConverterEmothawToPathological.__switch_task(int(task_search.group(1)))
                # Genero il nuovo nome del file di task.
                new_file_name = "c" + str(collection) + "_u" + str(collection) + str(user_id) + task_suffix + "txt"
                # Creo uno scrittore di file csv per scrivere i dati nel nuovo file di task.
                new_file = csv.writer(open(os.path.join(save_directory, new_file_name), 'w'), delimiter=' ', quotechar='"')
                # Apro in lettura il vecchio file di task.
                old_file = open(task, 'r')
                # Ottengo il numero di linee presenti nel file leggendo la prima riga del file stesso.
                rows_in_file = old_file.readline()
                # Avvio la conversione e il salvataggio del nuovo file.
                ConverterEmothawToPathological.__convert_file_line(rows_in_file, new_file, old_file)

    """
        @:param rows_in_file: contiene il numero di linee presenti nel file.
        @:param new_file: contiene il buffer di scrittura per il nuovo file.
        @:param old_file: contiene il buffer di lettura del vecchio file.
        Questo metdo scansiona riga per riga il vecchio file, converte i dati nel formato desiderato(floating point) e 
        poi salva i dati nel nuovo file seguendo la struttura desiderata.
    """
    @staticmethod
    def __convert_file_line(rows_in_file, new_file, old_file):
        for i in range(1, int(rows_in_file)):
            # Leggo al linea di dati dal file e separo i vari valori.
            csv_line = old_file.readline().split()
            csv_len = len(csv_line)
            # Verifico che i campi ottenuti dalla linea CSV siano 7
            if csv_len == 7:
                # Genero un vettore vuoto della lungezza della linea csv.
                new_csv_line = [None] * csv_len
                for j in range(csv_len):
                    # Converto il tutto nel formato richiesto
                    item = ConverterEmothawToPathological.__convert_csv_item_emothaw(csv_line[j], j)
                    # Salvo il valore nel vettore utilizzando i dizionari di associazione generati all'inizio del file.
                    new_csv_line[PATHOLOGICAL_FILE_STRUCTURE.get(EMOTHAW_FILE_STRUCTURE.get(j))] = item
                new_file.writerow(new_csv_line)
            else:
                print("INVALID LINE")

    """
        Questo metodo converte i dati passati nel formato desiderato, ovvero floating point.
    """
    @staticmethod
    def __convert_csv_item_emothaw(item, index):
        if index == 0 or index == 1:
            return_item = item + ".00"
        elif index == 2 or index == 3:
            return_item = str(int(item))
        else:
            return_item = "0." + item
        return return_item

    """
        @:param root_directory: contiene il percorso della cartella da scansionare.
        @:return: il metodo restituisce due liste la prima contenente ii percorsi delle cartelle individudate, la seconda
                    contentente solo i nomi delle cartelle individuate.
        Questo metodo fornisce due liste una dei percorsi conenuti della directory passata e una contentente solo il 
        nome delle directory individuate.
    """
    @staticmethod
    def __scan_directory(root_directory):
        new_directories = []
        directories = []
        for directory in os.listdir(root_directory):
            # Verifico che il file individuato sia una directory.
            if not (re.search(r'DS_Store', directory)) and (pathlib.Path(directory).suffix not in NOT_DIRECTORY):
                # Creo il percorso della directory.
                path = os.path.join(root_directory, directory)
                # Verifico che nella cartella ci siano file
                if len(os.listdir(path)) > 0:
                    # Aggiorno le liste.
                    directories.append(path)
                    new_directories.append(directory)
        return directories, new_directories

    """
        Questo metodo genera la cartella di destinazione del dataset.
    """
    @staticmethod
    def __make_converted_directory():
        converted_directory = os.path.join(RESOURCE_DIRECTORY, "ConvertedEmothaw")
        if not os.path.exists(converted_directory):
            os.mkdir(converted_directory)
            print("Directory make")
        else:
            print("Converted directory already exist")
        return converted_directory

    @staticmethod
    def __switch_task(item):
        switcher = {
            1: PENTAGON,
            2: SQUARE,
            3: FOUR_WORDS,
            4: LEFT_RING,
            5: RIGHT_RING,
            6: CLOCK,
            7: NATURAL_SENTENCE
        }
        return switcher.get(item, "Invalid task")
