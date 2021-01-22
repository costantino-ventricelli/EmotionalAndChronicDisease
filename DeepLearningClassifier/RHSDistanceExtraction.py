# coding=utf-8

import csv
import numpy as np

from DatasetManager.FileManager import FileManager

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
        self.__num_samples = num_samples * 2
        self.__minimum_samples = minimum_samples
        self.__end_point = [self.__minimum_samples, self.__minimum_samples - 50, 175]
        self.__start_point = [0, 50, 75]

    """
        Questo metodo permette di ottenre il tesore tridimensionale che servirà per il modello di learning, inotre fornisce
        altri dati sulla composizione dei campioni per poter successivamente effettuare i calcoli necessari a verificare 
        la bontà del modello.
        @:param path_list: contiene la lista dei file di campioni da analizzare.
        @:return: come detto il metodo restituisce:
            - final tensor: tensore tridimensionale contentente i capioni prelevati dai file.
            _ states: array contentente i gli stati corrispondenti di ogni campione rilevato.
            - len(final_tensor): contiene il numero dei campioni che compongono il tensore.
    """
    def extract_rhs_known_state(self, path_list):
        healthy_x = []
        healthy_y = []
        healthy_bs = []
        disease_x = []
        disease_y = []
        disease_bs = []
        # TODO: Random forest, decision three, logistic regression
        states = []
        for path in path_list:
            # Acuqisisco l'id del paziente
            id = FileManager.get_id_from_path(path)
            # Acquisisco lo stato del paziente
            state = FileManager.get_state_from_id(id)
            # Leggo i punti campionati nel file
            partial_x, partial_y, partial_bs = RHSDistanceExtract.__read_samples_from_file(path)
            # Trasformo i punti in segmenti RHS.
            partial_x, partial_y, partial_bs = self.__transform_point_in_rhs(partial_x, partial_y, partial_bs)
            # Raddoppio il numero dei campioni RHS così da ampliare il dataset.
            partial_x, partial_y, partial_bs = self.__extract_subs_from_samples(partial_x, partial_y, partial_bs)
            # Suddivido i capioni in base allo stato di salute del paziente a cui appartengono, dopo di che genero degli
            # array mono dimensionali di lunghezza pari alla lunghezza dei campioni richiesta dal Modello.
            if state == HEALTHY_STATE:
                self.__create_sample_sequence(healthy_x, healthy_y, healthy_bs, partial_x, partial_y, partial_bs)
            else:
                self.__create_sample_sequence(disease_x, disease_y, disease_bs, partial_x, partial_y, partial_bs)

        # Dopo aver ultimato l'estrazione genero due tensori tridimensionali, uno per i pazienti sani e uno per i pazienti malati
        healthy_tensor = np.reshape(np.array(healthy_x + healthy_y + healthy_bs), (len(healthy_x), self.__num_samples, FEATURES))
        disease_tensor = np.reshape(np.array(disease_x + disease_y + disease_bs), (len(disease_x), self.__num_samples, FEATURES))
        # A questo punto per ottenere un dataset bilanciato in ogni situazione valuto quale dei due tensori possiede meno.
        if len(healthy_tensor) < len(disease_tensor):
            end_point = len(healthy_tensor)
        else:
            end_point = len(disease_tensor)
        # Il tensore con meno campioni verrà utilizzato per generare il tensore finale, il quale verrà composto inserendo
        # prima tutti gli utenti sani e poi tutti gli utenti sani, si è già provato un approccio alternato, ma ha dato
        # scarsi risultati.
        final_tensor = np.concatenate((healthy_tensor[0: end_point], disease_tensor[0: end_point]))
        # Genero infine il vettore gli stati
        states += [HEALTHY_STATE for _ in range(end_point)] + [DISEASE_STATE for _ in range(end_point)]
        return np.array(final_tensor), np.array(states), len(final_tensor)

    """
        Questo metodo si comporta esattamente come il precedente, solo che questo considera un solo file per volta e quindi
        non effettua il bilanciamento del dataset, questo metodo è utile per prelevare i campioni necessari al test del 
        modello
    """
    def extract_rhs_file(self, path):
        x_samples = []
        y_samples = []
        bs_samples = []
        id = FileManager.get_id_from_path(path)
        state = FileManager.get_state_from_id(id)
        partial_x, partial_y, partial_bs = RHSDistanceExtract.__read_samples_from_file(path)
        partial_x, partial_y, partial_bs = self.__transform_point_in_rhs(partial_x, partial_y, partial_bs)
        partial_x, partial_y, partial_bs = self.__extract_subs_from_samples(partial_x, partial_y, partial_bs)
        self.__create_sample_sequence(x_samples, y_samples, bs_samples, partial_x, partial_y, partial_bs)
        tensor = np.reshape((x_samples + y_samples + bs_samples), (len(x_samples), self.__num_samples, FEATURES))
        states = [state for _ in range(len(x_samples))]
        return np.array(tensor), np.array(states)

    """
        Questo metodo permette di generare dei vettori contenenti i campioni dell'array, generando tre nuovi vettori di
        lunghezza prestabilita dal numero di campioni che il sistema ha richiesto.
    """
    def __create_sample_sequence(self, healthy_x, healthy_y, healthy_bs, partial_x, partial_y, partial_bs):
        healthy_x += self.__slice_array(partial_x)
        healthy_y += self.__slice_array(partial_y)
        healthy_bs += self.__slice_array(partial_bs)

    """
        Questo metodo è quello che effettua la divisione dell'array originale in una matrice con numero di colonne pari 
        al numero di campioni richiesti.
    """
    def __slice_array(self, array):
        slice = len(array) // self.__num_samples
        sliced_array = []
        for i in range(slice):
            sliced_array.append(array[i * self.__num_samples: (i * self.__num_samples) + self.__num_samples])
        return sliced_array

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
                partial_x.append(float(row[X_COORDINATE]))
                partial_y.append(float(row[Y_COORDINATE]))
                partial_bs.append(float(row[BOTTOM_STATUS]))
                timestamp.append(float(row[TIMESTAMP]))
            csv_file.close()
        # Elimino i duplicati dalle lista.
        partial_x, partial_y, timestamp, partial_bs = FileManager.delete_duplicates(partial_x,
                                                                                    partial_y, timestamp, partial_bs)
        return np.array(partial_x).astype(float), np.array(partial_y).astype(float), np.array(partial_bs).astype(float)

    """
        Questo metodo si occupa della trasformazione di punti campionati in sequenze rhs.
        @:param partial_ps
                partial_x
                partial_y: tutte e tre le liste di input contengono i punti campionati del file, i quali permetteranno 
                            di calcolare le sequenze rhs per asse delle x, delle y e il pen_status delle sequeze stesse.
        @:return: il metodo quindi restituisce tre nuove liste, ma sta volta ogni elemento conterrà una sequenza rhs 
                    anziché un punto campionato.
    """
    @staticmethod
    def __transform_point_in_rhs(partial_x, partial_y, partial_bs):
        # Calcolo le sequenze rhs, come distanza tra due punti contigui.
        for i in range(0, len(partial_x) - 1):
            partial_x[i] = float(partial_x[i + 1]) - float(partial_x[i])
            partial_y[i] = float(partial_y[i + 1]) - float(partial_y[i])
            partial_bs[i] = float(partial_bs[i + 1]) - float(partial_bs[i])
        # Elimino l'ultimo elemento della lista in quanto ha valore solo come punto campionato.
        partial_x = np.delete(partial_x, (partial_x.size - 1))
        partial_y = np.delete(partial_y, (partial_y.size - 1))
        partial_bs = np.delete(partial_bs, (partial_bs.size - 1))
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
            x_axis = np.concatenate((x_axis, partial_x[self.__start_point[i]: self.__end_point[i]]))
            y_axis = np.concatenate((y_axis, partial_y[self.__start_point[i]: self.__end_point[i]]))
            bottom_status = np.concatenate((bottom_status, partial_bs[self.__start_point[i]: self.__end_point[i]]))
        return x_axis, y_axis, bottom_status
