"""
    I RowTouchPoint si sono rivelati inutilizzabili per questo dataset in quanto non c'è alcuna consecutività rilevabile
    nei time stamp del dataset.
"""
# coding=utf-8

import csv
import numpy as np

from DatasetManager.FileManager import FileManager

X_COORDINATE = 0
Y_COORDINATE = 1
TIMESTAMP = 3
PEN_STATUS = 6

FEATURES = 3
INTERVALS_NUMBER = 3
ML_INTERVAL = [0, 1]


class RTPExtraction:

    def __init__(self, num_samples, minimum_sample):
        self.__num_samples = num_samples
        self.__minimum_samples = minimum_sample
        self.__end_point = [self.__minimum_samples, self.__minimum_samples - 50, 175]
        self.__start_point = [0, 50, 75]

    """
            Questo metodo permette di ottenre il tesore quadridimensionale che servirà per il modello di learning, inotre fornisce
            altri dati sulla composizione dei campioni per poter successivamente effettuare i calcoli necessari a verificare 
            la bontà del modello.
            @:param path_list: contiene la lista dei file di campioni da analizzare.
            @:return: come detto il metodo restituisce:
                - three_dimensional_tensor:
                    samples[timestamp[x_axis, y_axis, time_stamp, pen_status]]]
                _ theoretical_states: una lista contenente tutti gli stati individuati per i file passati in input.
                - total_sample: il numero totale dei campioni calcolati per quella lista di file.
                - samples_file: contiene il numero di campioni estratti per ogni file.
        """
    def extract_rtp_known_state(self, path_list):
        x_axis = []
        y_axis = []
        pen_status = []
        new_strokes = []
        time_stamps = []
        states = []
        # Ottengo la lista degli id malati e sani
        healthy_ids, disease_ids = FileManager.get_healthy_disease_list()
        # Calcolo il numero totale dei campioni che verranno generati
        total_samples = len(path_list) * self.__num_samples
        # Creo una lista conenente tutti i campioni prelevati da ogni singolo file.
        samples_file = [self.__num_samples for _ in range(len(path_list))]
        for path in path_list:
            # Ottengo l'id dell'utente che sto analizzando
            id = FileManager.get_id_from_path(path)
            # Ottengo lo stato di salute del paziente.
            state = FileManager.get_state_from_id(id, healthy_ids)
            # Leggo i campioni dal file.
            partial_x, partial_y, partial_pen_status, partial_new_strokes, partial_time_stamps = RTPExtraction.__read_samples_from_file(path)
            # Trasformo i campioni in RowTouchPoint
            partial_x, partial_y, partial_pen_status, partial_new_strokes, partial_time_stamps = self.__transform_point_in_rtp(
                partial_x, partial_y, partial_pen_status, partial_new_strokes, partial_time_stamps)
            # Estraggo uno specificato numero di campioni amplificando il dataset.
            x_axis, y_axis, pen_status, new_strokes, time_stamps = self.__extract_subs_from_samples(pen_status, x_axis, y_axis,
                                                                                                    new_strokes, time_stamps,
                                                                                                    partial_pen_status,
                                                                                                    partial_x, partial_y,
                                                                                                    partial_new_strokes,
                                                                                                    partial_time_stamps)
            # Ottengo una lista di stati che servirà per il confronto successivo.
            states += [state for _ in range(self.__num_samples)]
        # Unisco le liste contententi i campioni in una sola matrice.
        fourth_dimensional_tensor = np.array(np.column_stack((x_axis, y_axis, pen_status)))
        # Ottengo la lunghezza di una riga di campioni
        samples_length = int(len(fourth_dimensional_tensor) / total_samples)
        # Genereo il tensore quadridimensionale.
        fourth_dimensional_tensor = np.reshape(fourth_dimensional_tensor, (total_samples, samples_length, FEATURES))
        return fourth_dimensional_tensor, np.array(states), total_samples, samples_file

    @staticmethod
    def __read_samples_from_file(path):
        partial_x = []
        partial_y = []
        partial_pen_status = []
        partial_new_stroke = []
        partial_time_stamps = []
        try:
            with open(path) as csv_file:
                # Leggo i file in modalità csv così da ottenere una separazione dei campi
                rows = csv.reader(csv_file, delimiter=' ')
                # Indica il pen staus precedente, di default l'algoritmo inizia con il pen status non attivo.
                prev_pen_status = 0
                for row in rows:
                    partial_x.append(float(row[X_COORDINATE]))
                    partial_y.append(float(row[X_COORDINATE]))
                    partial_time_stamps.append(float(row[TIMESTAMP]))
                    partial_pen_status.append(float(row[PEN_STATUS]))
                # Elimino i duplicati dai campioni letti
                partial_x, partial_y, partial_time_stamps, partial_pen_status = FileManager.delete_duplicates(partial_x,
                                                                                                              partial_y,
                                                                                                              partial_time_stamps,
                                                                                                              partial_pen_status)
                # Adesso individuo i punti in cui è iniziata una nuova linea nel file.
                for pen_status in partial_pen_status:
                    # Il controllo verifica che la penna sia poggiata sullo schermo del pad, dopo di che verifica che sia
                    # stata poggiata in questo momento, quindi se l'attuale pen status risulta essere differente da quello
                    # precedente posso affermare che si tratta del punto in cui inizierà la nuova linea.
                    if (pen_status == 1) and (prev_pen_status != pen_status):
                        new_stroke = 1
                    else:
                        new_stroke = 0
                    partial_new_stroke.append(new_stroke)
                    prev_pen_status = pen_status
        except FileNotFoundError as exception:
            print(exception)
        return list(reversed(partial_x)), list(reversed(partial_y)), list(reversed(partial_pen_status)), \
               list(reversed(partial_new_stroke)), list(reversed(partial_time_stamps))

    def __transform_point_in_rtp(self, partial_x, partial_y, partial_pen_status, partial_new_stroke, partial_time_stamps):
        # Calcolo i Rows Touch Point.
        for i in range(len(partial_x) - 1):
            partial_x[i] = (partial_x[i] - partial_x[i + 1])
            partial_y[i] = (partial_y[i] - partial_y[i + 1])
            partial_time_stamps[i] = (partial_time_stamps[i] - partial_time_stamps[i + 1])
        # Trasporto il range dei valori da quello attuale in [0, 1] così da ottenere dati che si prestano megli al machine
        # learning
        partial_x = list(np.interp(partial_x, [min(partial_x), max(partial_x)], ML_INTERVAL))
        partial_y = list(np.interp(partial_y, [min(partial_y), max(partial_y)], ML_INTERVAL))
        partial_time_stamps = list(np.interp(partial_time_stamps, [min(partial_time_stamps), max(partial_time_stamps)], ML_INTERVAL))
        # Prelievo un numero definito di campioni da ogni file.
        return partial_x[0: self.__minimum_samples], partial_y[0: self.__minimum_samples], \
                partial_pen_status[0: self.__minimum_samples], partial_new_stroke[0: self.__minimum_samples], \
                partial_time_stamps[0: self.__minimum_samples]

    """
            Questo metodo permette di estrarre tre sottosequeze dai campioni rhs, che verranno combinate poi in un unico vettore.
            @:param pen_status: contiene la lista totale dei pen status estratti dall'elaborazione
            @:param x_axis: contiene la lista totale dei segmenti in x estratti dall'elaborazione
            @:param y_axis: contiene la lista totale dei segmenti in y estratti dall'elaborazione
            @:param new_stroke: contiene la lista totale degli stroke estratti dall'elaborazione
            @:param time_stamp: contiene la lista totale dei time stamp estratti dall'elaborazione
            @:param pen_status: contiene la lista parziale dei pen status estratti dall'elaborazione del file attuale
            @:param x_axis: contiene la lista parziale dei segmenti in x estratti dall'elaborazione del file attuale
            @:param y_axis: contiene la lista parziale dei segmenti in y estratti dall'elaborazione del file attuale
            @:param new_stroke: contiene la lista parziale degli stroke estratti dall'elaborazione del file attuale
            @:param time_stamp: contiene la lista parziale dei time stamp estratti dall'elaborazione del file attuale
        """
    def __extract_subs_from_samples(self, pen_status, x_axis, y_axis, new_strokes, time_stamps, partial_x, partial_y,
                                    partial_ps, partial_new_stroke, partial_time_stamps):
        for i in range(0, INTERVALS_NUMBER):
            x_axis += partial_x[self.__start_point[i]: self.__end_point[i]]
            y_axis += partial_y[self.__start_point[i]: self.__end_point[i]]
            pen_status += partial_ps[self.__start_point[i]: self.__end_point[i]]
            new_strokes += partial_new_stroke[self.__start_point[i]: self.__end_point[i]]
            time_stamps += partial_time_stamps[self.__start_point[i]: self.__end_point[i]]
        return pen_status, x_axis, y_axis, new_strokes, time_stamps
