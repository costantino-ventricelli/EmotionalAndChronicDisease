# coding=utf-8

import csv
import numpy as np

from DatasetManager.FileManager import FileManager
from collections import Counter

X_COORDINATE = 0
Y_COORDINATE = 1
TIMESTAMP = 3
BOTTOM_STATUS = 6

FEATURES = 3
INTERVALS_NUMBER = 3

HEALTHY_STATE = 0
DISEASE_STATE = 1


class RHSDistanceExtract:

    """
        @:param minimum_samples: contiene il numero di campioni che ogni file deve contentere.
        @:param num_samples: contiene il numero di campioni che verranno prelevati da ogni sequenza RHS.
    """
    def __init__(self, minimum_samples, num_samples):
        self.__num_samples = num_samples
        self.__minimum_samples = minimum_samples
        self.__end_point = [self.__minimum_samples, self.__minimum_samples - 50, 175]
        self.__start_point = [0, 50, 75]

    """
        Questo metodo permette di ottenre il tesore tridimensionale che servirà per il modello di learning, inotre fornisce
        altri dati sulla composizione dei campioni per poter successivamente effettuare i calcoli necessari a verificare 
        la bontà del modello.
        @:param path_list: contiene la lista dei file di campioni da analizzare.
        @:return: come detto il metodo restituisce:
            - three_dimensional_tensor:
                samples[timestamp[x_axis, y_axis, button_status]]]
            _ theoretical_states: una lista contenente tutti gli stati individuati per i file passati in input.
            - total_sample: il numero totale dei campioni calcolati per quella lista di file.
            - samples_file: contiene il numero di campioni estratti per ogni file.
    """
    def extract_rhs_known_state(self, path_list):
        healthy_x = []
        healthy_y = []
        healthy_bs = []
        disease_x = []
        disease_y = []
        disease_bs = []
        states = []
        final_tensor = []
        control_list, _ = FileManager.get_healthy_disease_list()
        for path in path_list:
            id = FileManager.get_id_from_path(path)
            state = FileManager.get_state_from_id(id, control_list)
            partial_x, partial_y, partial_bs = RHSDistanceExtract.__read_samples_from_file(path)
            partial_x, partial_y, partial_bs = self.__transform_point_in_rhs(partial_x, partial_y, partial_bs)
            partial_x, partial_y, partial_bs = self.__extract_subs_from_samples(partial_x, partial_y, partial_bs)
            if state == HEALTHY_STATE:
                self.__create_sample_sequence(healthy_x, healthy_y, healthy_bs, partial_x, partial_y, partial_bs)
            else:
                self.__create_sample_sequence(disease_x, disease_y, disease_bs, partial_x, partial_y, partial_bs)
        healthy_tensor = np.reshape((healthy_x + healthy_y + healthy_bs), (len(healthy_x), self.__num_samples, FEATURES))
        disease_tensor = np.reshape((disease_x + disease_y + disease_bs), (len(disease_x), self.__num_samples, FEATURES))
        for i in range(len(healthy_tensor)):
            final_tensor.append(healthy_tensor[i])
            final_tensor.append(disease_tensor[i])
            states += [HEALTHY_STATE, DISEASE_STATE]
        return np.array(final_tensor), np.array(states), len(final_tensor)

    def extract_rhs_file(self, path):
        x_samples = []
        y_samples = []
        bs_samples = []
        control_list, _ = FileManager.get_healthy_disease_list()
        id = FileManager.get_id_from_path(path)
        state = FileManager.get_state_from_id(id, control_list)
        partial_x, partial_y, partial_bs = RHSDistanceExtract.__read_samples_from_file(path)
        partial_x, partial_y, partial_bs = self.__transform_point_in_rhs(partial_x, partial_y, partial_bs)
        partial_x, partial_y, partial_bs = self.__extract_subs_from_samples(partial_x, partial_y, partial_bs)
        self.__create_sample_sequence(x_samples, y_samples, bs_samples, partial_x, partial_y, partial_bs)
        tensor = np.reshape((x_samples + y_samples + bs_samples), (len(x_samples), self.__num_samples, FEATURES))
        states = [state for _ in range(len(x_samples))]
        return np.array(tensor), np.array(states)

    def __create_sample_sequence(self, healthy_x, healthy_y, healthy_bs, partial_x, partial_y, partial_bs):
        healthy_x += self.__slice_array(partial_x)
        healthy_y += self.__slice_array(partial_y)
        healthy_bs += self.__slice_array(partial_bs)

    def __slice_array(self, array):
        slice = len(array) // self.__num_samples
        sliced_array = []
        for i in range(slice):
            sliced_array.append(array[i * self.__num_samples: (i * self.__num_samples) + self.__num_samples])
        return sliced_array

    """
        @:param path: contiene il file da cui estrarre i campioni RHS.
        @:return: restituisce un tensore tridimensionale conentente i campioni RHS estratti dal file.
    """
    def extract_rhs_from_unknown(self, path):
        # Il processo di estrazione è praticamente identico a quello del precedente metodo solo che in questo caso i file
        # analizzati sono singoli e non si genera il vettore contenente i valori di confronto che saranno necessari nelle
        # fasi di addestramento, validazione e src.
        x_axis = []
        y_axis = []
        bottom_status = []
        partial_bs, partial_x, partial_y = self.__read_samples_from_file(path)
        partial_bs, partial_x, partial_y = self.__transform_point_in_rhs(partial_bs, partial_x, partial_y)
        bottom_status, x_axis, y_axis = self.__extract_subs_from_samples(bottom_status, x_axis, y_axis, partial_bs, partial_x,
                                                                         partial_y)
        three_dimensional_tensor = np.array(np.column_stack((x_axis, y_axis, bottom_status)))
        samples_length = int(len(three_dimensional_tensor) / self.__num_samples)
        three_dimensional_tensor = np.reshape(three_dimensional_tensor, (self.__num_samples, samples_length, FEATURES))
        return three_dimensional_tensor

    """
        @:param healthy_model: contiene il modello degli esempi per i pazienti sani.
        @:param disease_model: contiene il modello degli esempi per i pazienti malati.
        @:return: restituisce le liste contententi i campioni alternati tra healthy e disease.
    """
    def __join_dataset(self, healthy_model, disease_model):
        x_axis = []
        y_axis = []
        bottom_status = []
        expanded_samples = (self.__minimum_samples * 2)
        for i in range(expanded_samples):
            bottom_status, x_axis, y_axis = RHSDistanceExtract.__set_model(i, bottom_status, x_axis,
                                                                           y_axis, expanded_samples, healthy_model)
            bottom_status, x_axis, y_axis = RHSDistanceExtract.__set_model(i, bottom_status, x_axis,
                                                                           y_axis, expanded_samples, disease_model)
        return bottom_status, x_axis, y_axis

    @staticmethod
    def __set_model(i, bottom_status, x_axis, y_axis, expanded_samples, healthy_model):
        bottom_status += healthy_model[0][i * expanded_samples: (i * expanded_samples) + expanded_samples]
        x_axis += healthy_model[1][i * expanded_samples: (i * expanded_samples) + expanded_samples]
        y_axis += healthy_model[2][i * expanded_samples: (i * expanded_samples) + expanded_samples]
        return bottom_status, x_axis, y_axis

    """
        Il metdodo permette di leggere in 4 array tutti i campioni presenti in uno dei file generati dall'esecuzione di 
        un task, dopo di che elimina tutti i duplicati.
        @:param path: contiene il percorso del file che si deve analizzare.
        @:return: restituisco i quattro vettori generati mentre si prelevavano i campioni dal file.
    """
    @staticmethod
    def __read_samples_from_file(path):
        partial_x = []
        partial_y = []
        partial_bs = []
        timestamp = []
        with open(path, newline='') as csv_file:
            # Leggo il file di campioni come fosse un file csv con delimitatore di colonna indicato da uno spazio, anziché
            # una virgola.
            rows = csv.reader(csv_file, delimiter=' ')
            for row in rows:
                partial_x.append(row[X_COORDINATE])
                partial_y.append(row[Y_COORDINATE])
                partial_bs.append(row[BOTTOM_STATUS])
                timestamp.append(row[TIMESTAMP])
            csv_file.close()
        # Elimino i duplicati dalle lista.
        partial_x, partial_y, timestamp, partial_bs = FileManager.delete_duplicates(partial_x,
                                                                                    partial_y, timestamp, partial_bs)
        return partial_x, partial_y, partial_bs

    """
        Questo metodo si occupa della trasformazione di punti campionati in sequenze rhs.
        @:param partial_ps
                partial_x
                partial_y: tutte e tre le liste di input contengono i punti campionati del file, i quali permetteranno 
                            di calcolare le sequenze rhs per asse delle x, delle y e il pen_status delle sequeze stesse.
        @:return: il metodo quindi restituisce tre nuove liste, ma sta volta ogni elemento conterrà una sequenza rhs 
                    anziché un punto campionato.
    """
    def __transform_point_in_rhs(self, partial_x, partial_y, partial_bs):
        # Calcolo le sequenze rhs, come distanza tra due punti contigui.
        for i in range(0, len(partial_x) - 1):
            partial_x[i] = float(partial_x[i + 1]) - float(partial_x[i])
            partial_y[i] = float(partial_y[i + 1]) - float(partial_y[i])
            partial_bs[i] = float(partial_bs[i + 1]) - float(partial_bs[i])
        # Elimino l'ultimo elemento della lista in quanto ha valore solo come punto campionato.
        partial_x.pop()
        partial_y.pop()
        partial_bs.pop()
        # Se il numero di sequenze rhs non dovesse rivelarsi sufficiente, in base al numero minimo di campioni che verrà
        # selezionato, genero "sinteticamente" le sequenze replicando l'ultima fino a raggiungere il numero minimo accettabile.
        if len(partial_x) < self.__minimum_samples:
            partial_x += [partial_x[len(partial_x) - 1] for _ in range(len(partial_x) - 1, self.__minimum_samples)]
            partial_y += [partial_y[len(partial_y) - 1] for _ in range(len(partial_y) - 1, self.__minimum_samples)]
            partial_bs += [partial_bs[len(partial_bs) - 1] for _ in range(len(partial_bs) - 1, self.__minimum_samples)]
        return partial_x, partial_y, partial_bs

    """
        Questo metodo permette di estrarre tre sottosequeze dai campioni rhs, che verranno combinate poi in un unico vettore.
        @:param pen_status: contiene la lista di campioni rhs totale dell'estrazione di tutti i campioni dai file.
        @:param x_axis: contiene la lista dei campioni rhs totale dell'estrazione dei campioni di tutti i file.
        @:param y_axis: contiene la lista dei campioni rhs totale dell'estrazione dei campioni ti tutti i file.
        @:param partial_x: contiene la lista dei campioni rhs estratta per quel file.
        @:param partial_y: contiene la lista dei campioni rhs estratta per quel file.
        @:param partial_ps: contiene la lista dei campioni rhs estratta per quel file.
    """
    def __extract_subs_from_samples(self, partial_x, partial_y, partial_bs):
        x_axis = []
        y_axis = []
        bottom_status = []
        for i in range(0, INTERVALS_NUMBER):
            x_axis += partial_x[self.__start_point[i]: self.__end_point[i]]
            y_axis += partial_y[self.__start_point[i]: self.__end_point[i]]
            bottom_status += partial_bs[self.__start_point[i]: self.__end_point[i]]
        return x_axis, y_axis, bottom_status
