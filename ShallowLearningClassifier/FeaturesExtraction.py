# coding=utf-8

import csv
import os
import re

import numpy as np

from DatasetManager.Costants import *
from .Features import Features

TASK_STRUCTURE = {
    'x_axis': 0,
    'y_axis': 1,
    'pressure': 2,
    'timestamp': 3,
    'azimuth': 4,
    'altitude': 5,
    'pen_status': 6
}

HEADER_FILE = 'header_file.csv'
FEATURE_FILE_LIST = 'feature_list.csv'
RESOURCE_FOLDER = 'resource'

MAIN_FEATURE_KEY = 'MAIN_FEATURE'
PARAMETER_KEY = 'OTHER_PARAMETERS'
AXIS_KEY = 'AXIS'


"""
    Questa classe permette di estrarre tutte le feature, funzionali e parametriche da ogni dataset.
    Il parametro di input alla classe è il dataset memorizzato sotto forma di lista ogni riga del dataset contiene
    il punto campionato letto dal file con la struttura descritta dal dizionario TASK_STRUCTURE.
        1)  Il primo passaggio del ciclo di estrazione consiste nel generare la lista contenente il nome di tutte le feature 
        che verranno estratte.
        2)  Successivamente in base alla feature verrà richiamato il metodo corretto per estrarre tali valori dal dataset.
            2.1)    Nel caso in cui la feature è di tipo funzionale i valori calcolati per la funzione pemetteranno di calcolare una 
                    serie di statistiche desctitte da __get_statistic_value.
        3)  Dopo di che i valori vengono salvati all'interno di un dizionario che verrà restituito al metodo chiamante e 
            conterrà tutte le feature calcolate per il dataset passato come valore iniziale.
"""


class FeatureExtraction:

    """
        Questo metodo restituisce l'header del file che contiene i vaoli estratti per le feature.
    """
    @staticmethod
    def get_file_dictionary():
        return FeatureExtraction.__get_header_file()

    """
        Questo metodo permette scansiona un vettorre composto da tutte le feature che verranno estratte dai dataset.
    """
    @staticmethod
    def get_features_for_task(dataset):
        # Questo dizionario conterrà i risultati di tutta l'estrazione
        task_file = FeatureExtraction.__get_header_file()
        # Questa lista la lista di tutte le feature che verranno estratte dal dataset
        features = FeatureExtraction.__get_feature_list()
        # Quindi per ogni feature presente nella lista verranno richiamati metodi necessari all'estrazione di quella
        # feature.
        for feature in features:
            print("Computing: ", feature)
            # Qui si verifica che il file non sia vuoto.
            if len(dataset) > 0:
                # In result viene restituito un dizionario che contiene le features estratte che potrebbero essere più
                # di una quando si estraggono feature funzionali di cui vengono calcolate varie funzioni statistiche.
                result = FeatureExtraction.__get_feature_command(feature, dataset)
                # Questo for ispeziona tutti i dati ottenuti e li appone nel dizionario che conterrà tutte le feature
                # estratte per il task.
                for key, item in result.items():
                    try:
                        task_file = FeatureExtraction.__update_dictionary(task_file, key, item)
                    except KeyError as error:
                        print(error)
            else:
                print("Empty file, skipping...")
        return task_file

    """
        Questa funzione permette semplicemente di aggiornare i valori del dizionario dei task utilizzando il nome delle 
        feature come chiave.
    """
    @staticmethod
    def __update_dictionary(dictionary, key, item):
        if key in dictionary.keys():
            value = dictionary[key]
            if value is None:
                dictionary[key] = [item]
            else:
                value.append(item)
        else:
            raise KeyError("The key: " + key + " doesn't exist")
        return dictionary

    """
        Questo metodo permette di salare le funzioni statistiche in maniera ordinata all'interno di un disionario
    """
    @staticmethod
    def __get_statistic_value(data):
        result = {
            '[mean]': np.mean(data),
            '[median]': np.median(data),
            '[stan. dev.]': np.std(data),
            '[1 per]': np.percentile(data, 10),
            '[99 per]': np.percentile(data, 99),
            '[1-99 per]': np.percentile(data, 99) - np.percentile(data, 10)
        }
        return result

    """
        Questo metodo legge il file che contiene tutti i nomi delle feature per crere la lista da scorrere per calcolarle.
    """
    @staticmethod
    def __get_feature_list():
        feature = []
        with open(os.path.join(RESOURCE_FOLDER, FEATURE_FILE_LIST), 'r') as file:
            csv_file = csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            for row in csv_file:
                feature = row
            file.close()
        return feature

    """
        Questo metodo genera il dizionario che conterrà i valori calcolati per ogni feature.
    """
    @staticmethod
    def __get_header_file():
        with open(os.path.join(RESOURCE_FOLDER, HEADER_FILE), 'r') as file:
            csv_file = csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_ALL)
            for row in csv_file:
                header_list = row
            file.close()
        # Utilizzo i nome delle features da estrarre per generare un dizionario il quale sarà inizialmente vuoto
        header = dict.fromkeys(header_list, None)
        return header

    """
        Questo metodo permette di individuare quale feature si sta calcolando e quali sono i parametri passati per il calcolo
        di tale feature
    """
    @staticmethod
    def __get_feature_command(feature, dataset):
        # Qui viene divisio il nome della feature per individuare quale feature calcolare, su che asse eventualmente va
        # calcolate e come bisogna selezionare i punti.
        feature_command = str.split(feature, '_')
        # Qui viene restituito il nome della feature da calcolare.
        feature_command = FeatureExtraction.__name_analyser(feature_command)
        # A questo punto si individuano eventuai assi specificati nel nome della feature.
        group_search = re.search(r'x$|y$', feature_command[0])
        # Se sono stati individuati assi nella specifica del nome questo valore viene salvato per puoi essere utilizzato
        # nel momento in cui si richiamerà la funzione per calcolare la feature
        if group_search:
            axis = group_search.group(0)
            # Si rimuove il nome dell'asse dal nome della feature per poter effettuare la selezione sul tipo di feature.
            feature_command[0] = feature_command[0].replace(axis, '')
            axis += '_axis'
        else:
            axis = None
        # Questo dizionario costituisce l'insieme dei parametri necessari per il calcolo della feature
        command = {
            # Il tipo di feature da calcolare
            MAIN_FEATURE_KEY: feature_command[0],
            # L'eventuale asse su cui calcolare la feature.
            AXIS_KEY: axis,
            # I parametri necessari per calcolare la feature
            PARAMETER_KEY: feature_command[1: len(feature_command)]
        }
        # Quindi la funzione selezionata viene richiamata secono questo modello standard in cui viene passato il nome
        # completo della feature la lista dei parametri che costituiscono il comando per calcolare correttamente la feature
        # e il riferimento dal dataset passato come parametro iniziale.
        function = FeatureExtraction.__command_switch(feature_command)
        return function(feature, command, dataset)

    """
        Questo metodo di switch permette di ricostruire i nomi di quelle feature che si compongono di più parti,
        come ad esempio STROKE_SIZE, STROKE_DURATION, etc...
    """
    @staticmethod
    def __name_analyser(split):
        # Lo switcher restituisce la funzione corretta per riscostruire il nome della feature con il numero di componenti
        # che compongono il nome stesso.
        switcher = {
            'STROKE': (FeatureExtraction.__join_command_name, 2),
            'TOTAL': (FeatureExtraction.__join_command_name, 2),
            'N': (FeatureExtraction.__join_command_name, 2),
            'NORM': (FeatureExtraction.__join_command_name, 2),
            'RATIO': (FeatureExtraction.__join_command_name, 2),
            'PEAK': (FeatureExtraction.__join_command_name, 2),
            'R': (FeatureExtraction.__join_command_name, 2),
            'SHANNON': (FeatureExtraction.__join_command_name, 2),
            'RENYI': (FeatureExtraction.__join_command_name, 2),
            'GL': (FeatureExtraction.__join_command_name, 2),
        }
        # Questo pezzo di codice verifica prima di tutto che la selezione sia andata a buon fine, dopo di che ricompone il
        # nome.
        function, slices_number = switcher.get(split[0], (None, 0))
        if function is not None:
            name = [function(split, slices_number)]
            for i in range(slices_number, len(split)):
                name.append(split[i])
        else:
            name = split
        return name

    """
        Questo metodo ricompone il nome della feature.
    """
    @staticmethod
    def __join_command_name(slices, slices_number):
        name = ""
        for i in range(slices_number):
            name += slices[i]
            if i < slices_number - 1:
                name += "_"
        return name

    """
        Questo switcher permette di selezionare il metodo che ci permetterà di estrarre le feature.
    """
    @staticmethod
    def __command_switch(argument):
        switcher = {
            'DIS': FeatureExtraction.__get_displacement,
            'VEL': FeatureExtraction.__get_velocity,
            'ACC': FeatureExtraction.__get_acceleration,
            'JERK': FeatureExtraction.__get_jerk,
            'STROKE_SIZE': FeatureExtraction.__get_stroke_size,
            'STROKE_DURATION': FeatureExtraction.__get_stroke_duration,
            'STROKE_SPEED': FeatureExtraction.__get_stroke_speed,
            'TIME': FeatureExtraction.__get_time,
            'TOTAL_TIME': FeatureExtraction.__get_total_time,
            'N_STROKE': FeatureExtraction.__get_number_of_stroke,
            'NORM_TIME': FeatureExtraction.__get_normalized_time,
            'RATIO_TIME': FeatureExtraction.__get_ratio_time,
            'RATIO_PUPD': FeatureExtraction.__get_ration_pupd,
            'PEAK_VEL': FeatureExtraction.__get_peak_velocity,
            'PEAK_ACC': FeatureExtraction.__get_peak_velocity,
            'NCV': FeatureExtraction.__get_number_velocity_change,
            'NCA': FeatureExtraction.__get_number_acceleration_change,
            'R_NCV': FeatureExtraction.__get_ratio_number_velocity_change,
            'R_NCA': FeatureExtraction.__get_ratio_number_acceleration_change,
            'PRESSURE': FeatureExtraction.__get_pressure,
            'NCP': FeatureExtraction.__get_number_change_pressure,
            'NVV': FeatureExtraction.__get_normalized_velocity_variability,
            'SHANNON_ENTROPY': FeatureExtraction.__get_shannon_entropy,
            'RENYI_ENTROPY': FeatureExtraction.__get_renyi_entropy,
            'SNR': FeatureExtraction.__get_signal_to_noise_ratio,
            'DCT': FeatureExtraction.__get_discrete_cosine_transform,
            'DFT': FeatureExtraction.__get_discrete_fourier_transform,
            'RCEP': FeatureExtraction.__get_real_cepstrum,
            'GL_FD': FeatureExtraction.__get_fractional_derivative
        }
        function = switcher.get(argument[0], lambda: 'Invalid argument')
        return function

    """
        Questo metodo permette di ottenere un flag che identifica se la feature si riferisce ai dati campionati in aria
        'ia": 0 o sulla superficie 'os': 1.
    """
    @staticmethod
    def __get_finding_status(status):
        return ON_AIR if status == 'ia' else ON_SURFACE

    """
        Questo metodo permette di ottenere il displacement dei punti campionati nel dataset distinguendo tra i punti 
        on air, on surface o non specificati.
        Da qui in poi i metodi si comportano tutti alla stessa maniera più o meno, quindi verranno spiegati solo quei 
        metodi che utilizzano un approccio diverso per calcolare le feature.
    """
    @staticmethod
    def __get_displacement(feature, command, data):
        # Qui vengono ottenuti i parametri supplementari per il calcolo dei valori.
        other_parameters = command[PARAMETER_KEY]
        # Qui invece viene prelevato l'eventuale asse su cui avviare i calcoli.
        axis = command[AXIS_KEY]
        # Se sono presenti dei parametri supplementari significa che la feature specifica quali punti considerare per il
        # calcolo: in air o on surface.
        if len(other_parameters) > 0:
            # Ottengo il flag discriminativo per il tipo di punti da considerare.
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            # Verifico se è satato specificato un asse per il calcolo.
            if axis is None:
                displacement = Features.get_displacement_pen_status(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                                    data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                displacement = Features.get_displacement_axis_pen_status(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                         finding_status)
        else:
            # Qui vine avviato il calcolo della feature se non sono stati indicati punti specifici, ovviamente verificando
            # l'eventuale asse richiesto dal calcolo.
            if axis is None:
                displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
            else:
                displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
        return FeatureExtraction.__generate_result_dictionary(displacement, feature)

    """
        Per il calolo della velocità ovviamente serve lo spazio ed il tempo, quindi prima di procedere al calcolo effetti
        vo della velocità viene prima di tutto ottenuto il displacement dell dataset.
    """
    @staticmethod
    def __get_velocity(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                displacement = Features.get_displacement_pen_status(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                                    data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                displacement = Features.get_displacement_axis_pen_status(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                         finding_status)
            time_stamps = Features.get_time_function(data[TASK_STRUCTURE['timestamp']], data[TASK_STRUCTURE['pen_status']], finding_status)
            velocity = Features.get_displacement_velocity(displacement, time_stamps, bool(finding_status))
        else:
            if axis is None:
                displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
            else:
                displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
            velocity = Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['timestamp']])
        return FeatureExtraction.__generate_result_dictionary(velocity, feature)

    """
        Per il calcolo dell'accellerazione necessitiamo della velocità e per la velocità necessitiamo dello spazio, quindi
        prima del calcolo dell'accellerazione vengono calcolati spazio e velocità.
    """
    @staticmethod
    def __get_acceleration(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                displacement = Features.get_displacement_pen_status(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                                    data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                displacement = Features.get_displacement_axis_pen_status(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                         finding_status)
            time_stamps = Features.get_time_function(data[TASK_STRUCTURE['timestamp']],
                                                     data[TASK_STRUCTURE['pen_status']], finding_status)
            velocity = Features.get_displacement_velocity(displacement, time_stamps)
            acceleration = Features.get_displacement_acceleration(velocity, time_stamps, bool(finding_status))
        else:
            if axis is None:
                displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
            else:
                displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
            velocity = Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['timestamp']])
            acceleration = Features.get_displacement_acceleration(velocity, data[TASK_STRUCTURE['timestamp']])
        return FeatureExtraction.__generate_result_dictionary(acceleration, feature)

    """
        Come per il metodo precedente anche il jerk richiede il calcolo dell'accellerazione, quindi verranno calcolate
        tutte le funzioni necessarie ad ottenere l'accellerazione prima di calcolare il jerk vero e proprio.
    """
    @staticmethod
    def __get_jerk(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                displacement = Features.get_displacement_pen_status(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                                    data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                displacement = Features.get_displacement_axis_pen_status(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                         finding_status)
            time_stamps = Features.get_time_function(data[TASK_STRUCTURE['timestamp']],
                                                     data[TASK_STRUCTURE['pen_status']], finding_status)
            acceleration = Features.get_displacement_acceleration(
                Features.get_displacement_velocity(displacement, time_stamps),
                time_stamps)
            jerk = Features.get_jerk(acceleration, time_stamps, bool(finding_status))
        else:
            if axis is None:
                displacement = Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']])
            else:
                displacement = Features.get_displacement_axis(data[TASK_STRUCTURE[axis]])
            acceleration = Features.get_displacement_acceleration(
                Features.get_displacement_velocity(displacement, data[TASK_STRUCTURE['timestamp']]),
                data[TASK_STRUCTURE['timestamp']])
            jerk = Features.get_jerk(acceleration, data[TASK_STRUCTURE['timestamp']])
        return FeatureExtraction.__generate_result_dictionary(jerk, feature)

    """
        Da qui fino alla trasformazione coseno sono tutte features parametriche, comunque lo sviluppo dei metodi segue la 
        linea guida dei precedenti, in qualche caso però il nome stesso della feature è più che sufficiente al calcolo
        quindi il parametro command potrebbe non essere utilizzato proprio.
    """
    @staticmethod
    def __get_stroke_size(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
            if axis is None:
                strokes_size = Features.get_mean_stroke_size(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                             data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                strokes_size = Features.get_mean_axis_stroke_size(data[TASK_STRUCTURE[axis]], data[TASK_STRUCTURE['pen_status']],
                                                                  finding_status)
        else:
            strokes_size = None
            print("Stroke size error")
        return {feature + "[mean]": strokes_size}

    @staticmethod
    def __get_stroke_duration(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature + "[mean]": Features.get_mean_stroke_duration(data[TASK_STRUCTURE['timestamp']], data[TASK_STRUCTURE['pen_status']],
                                                           finding_status)}

    @staticmethod
    def __get_stroke_speed(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature + "[mean]": Features.get_mean_stroke_velocity(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']],
                                                           data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']],
                                                           finding_status)}

    @staticmethod
    def __get_time(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        return {feature: Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']],
                                           finding_status)}

    @staticmethod
    def __get_total_time(feature, command, data):
        return {feature: Features.get_total_time(data[TASK_STRUCTURE['timestamp']])}

    @staticmethod
    def __get_number_of_stroke(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        if len(other_parameters) > 0:
            finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
        else:
            finding_status = None
        return {feature: Features.get_stroke(data[TASK_STRUCTURE['pen_status']], finding_status)}

    @staticmethod
    def __get_normalized_time(feature, command, data):
        finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
        time = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], finding_status)
        total_time = Features.get_total_time(data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_total_time_norm(time, total_time)}

    @staticmethod
    def __get_ratio_time(feature, command, data):
        time_on_surface = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], ON_SURFACE)
        time_in_air = Features.get_time(data[TASK_STRUCTURE['pen_status']], data[TASK_STRUCTURE['timestamp']], ON_AIR)
        return {feature: Features.get_ratio_time(time_on_surface, time_in_air)}

    @staticmethod
    def __get_ration_pupd(feature, command, data):
        return {feature: Features.get_pen_status_ratio(data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_peak_velocity(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        return {feature + "[mean]": Features.get_mean_function_peak(data[TASK_STRUCTURE['pen_status']], velocity)}

    @staticmethod
    def __get_peak_acceleration(feature, command, data):
        acceleration = Features.get_displacement_acceleration(Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']],
                                                            data[TASK_STRUCTURE['y_axis']]), data[TASK_STRUCTURE['timestamp']]), data[TASK_STRUCTURE['timestamp']])
        return {feature + "[mean]": Features.get_mean_function_peak(data[TASK_STRUCTURE['pen_status']], acceleration)}

    @staticmethod
    def __get_number_velocity_change(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_changes(velocity, data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_number_acceleration_change(feature, command, data):
        acceleration = Features.get_displacement_acceleration(Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']],
                                                              data[TASK_STRUCTURE['y_axis']]), data[TASK_STRUCTURE['timestamp']]), data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_changes(acceleration, data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_ratio_number_velocity_change(feature, command, data):
        velocity = FeatureExtraction.__get_number_velocity_change(feature, None, data)
        time = FeatureExtraction.__get_total_time(feature, None, data)
        return {feature: Features.get_relative_changes(velocity[feature], time[feature])}

    @staticmethod
    def __get_ratio_number_acceleration_change(feature, command, data):
        acceleration = FeatureExtraction.__get_number_acceleration_change(feature, None, data)
        time = FeatureExtraction.__get_total_time(feature, None, data)
        return {feature: Features.get_relative_changes(acceleration[feature], time[feature])}

    @staticmethod
    def __get_pressure(feature, command, data):
        return {feature + "[mean]": Features.get_mean_pressure(data[TASK_STRUCTURE['pressure']], data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_number_change_pressure(feature, command, data):
        return {feature: Features.get_pressure_changes(data[TASK_STRUCTURE['pressure']], data[TASK_STRUCTURE['pen_status']])}

    @staticmethod
    def __get_normalized_velocity_variability(feature, command, data):
        velocity = Features.get_displacement_velocity(Features.get_displacement(data[TASK_STRUCTURE['x_axis']], data[TASK_STRUCTURE['y_axis']]),
                                                      data[TASK_STRUCTURE['timestamp']])
        time = Features.get_total_time(data[TASK_STRUCTURE['timestamp']])
        return {feature: Features.get_normalized_velocity_variability(velocity, time)}

    @staticmethod
    def __get_shannon_entropy(feature, command, data):
        other_parameters = command[PARAMETER_KEY]
        axis = command[AXIS_KEY]
        if len(other_parameters) > 0:
            if other_parameters[0] == 'os' or other_parameters[0] == 'ia':
                finding_status = FeatureExtraction.__get_finding_status(other_parameters[0])
                shannon_entropy = Features.get_shannon_entropy(data[TASK_STRUCTURE[axis]], finding_status, data[TASK_STRUCTURE['pen_status']])
            elif other_parameters[0] == '1Imf' or other_parameters[0] == '2Imf':
                shannon_entropy = Features.get_imf_shannon_entropy(Features.get_imf(data[TASK_STRUCTURE[axis]]))
            else:
                shannon_entropy = None
                print("Error in Shannon Entropy")
        else:
            shannon_entropy = Features.get_shannon_entropy(data[TASK_STRUCTURE[axis]])
        return {feature: shannon_entropy}

    @staticmethod
    def __get_renyi_entropy(feature, command, data):
        axis = command[AXIS_KEY]
        if len(command[PARAMETER_KEY]) > 1:
            status = command[PARAMETER_KEY][0]
            order = int(command[PARAMETER_KEY][1].replace('o', ''))
        else:
            status = None
            order = int(command[PARAMETER_KEY][0].replace('o', ''))
        if status is not None and status == 'ia' or status == 'os':
            finding_status = FeatureExtraction.__get_finding_status(status)
            renyi_entropy = Features.get_renyi_entropy(data[TASK_STRUCTURE[axis]], order, finding_status,
                                                       data[TASK_STRUCTURE['pen_status']])
        elif status is not None and status == '1Imf' or status == '1Imf':
            renyi_entropy = Features.get_imf_renyi_entropy(Features.get_imf(data[TASK_STRUCTURE[axis]]), order)
        else:
            renyi_entropy = Features.get_renyi_entropy(data[TASK_STRUCTURE[axis]], order)
        return {feature: renyi_entropy}

    @staticmethod
    def __get_signal_to_noise_ratio(feature, command, data):
        axis = command[AXIS_KEY]
        if len(command[PARAMETER_KEY]) > 0:
            if command[PARAMETER_KEY][0] == 'ia' or command[PARAMETER_KEY][0] == 'os':
                finding_status = FeatureExtraction.__get_finding_status(command[PARAMETER_KEY][0])
                snr = Features.get_snr(data[TASK_STRUCTURE[axis]], finding_status, data[TASK_STRUCTURE['pen_status']])
            else:
                snr = Features.get_imf_snr(Features.get_imf(data[TASK_STRUCTURE[axis]]))
        else:
            snr = Features.get_snr(data[TASK_STRUCTURE[axis]])
        return {feature: snr}

    @staticmethod
    def __get_discrete_cosine_transform(feature, command, data):
        function_result = FeatureExtraction.__get_all_function_combination(command, data)
        return FeatureExtraction.__generate_result_dictionary(Features.get_discrete_cosine_transform(function_result), feature)

    @staticmethod
    def __get_discrete_fourier_transform(feature, command, data):
        function_result = FeatureExtraction.__get_all_function_combination(command, data)
        return FeatureExtraction.__generate_result_dictionary(Features.get_real_fast_fourier_transform(function_result),
                                                              feature)

    @staticmethod
    def __get_real_cepstrum(feature, command, data):
        function_result = FeatureExtraction.__get_all_function_combination(command, data)
        return FeatureExtraction.__generate_result_dictionary(Features.get_cepstrum(function_result), feature)

    """
        Questo metodo permette di richimare il calcolo di cepstrum, cosine transform, fft su tutte le combinazioni possibili
        di funzioni:
            - displacement;
            - velocità:
            - accellerazione;
            - jerk. 
    """
    @staticmethod
    def __get_all_function_combination(command, data):
        # In other parameter è contenuto il nome della funzione di cui si vuole calcolare la trasformata, il prodotto di
        # convoluzione omomorfa o le derivate frazionarie.
        other_parameter = command[PARAMETER_KEY]
        group_search = re.search(r'x$|y$', other_parameter[0])
        # Acora si individua un eventuale asse.
        if group_search:
            axis = group_search.group(0)
            other_parameter[0] = other_parameter[0].replace(axis, '')
            point = {'axis': data[TASK_STRUCTURE[axis + '_axis']]}
        else:
            point = {'x_axis': data[TASK_STRUCTURE['x_axis']],
                     'y_axis': data[TASK_STRUCTURE['y_axis']]}
        # Qui lo switcher ci permette di capire la funzione e il dominio della funzione stessa: spazio; tempo.
        function, domain = FeatureExtraction.__function_switch(other_parameter[0])
        # Verifichiamo che siano anche specificati i punti di cui calcolare le funzioni.
        if len(other_parameter) > 1:
            finding_status = FeatureExtraction.__get_finding_status(other_parameter[1])
        else:
            finding_status = None
        # Dal nome del dominio si passa ai sui valori funzionali calcolati, quindi se si tratta di tempo vengono ottenuti
        # i time stamp necessari per ottenere la sequenza temporale necessaria.
        if domain == 'timestamp':
            if finding_status is not None:
                # Nel caso in cui sia specificato quali punti considerare verrà fatta ovviamente una selezione sui dati
                # per ottenere solo quelli necessari.
                domain = Features.get_time_function(data[TASK_STRUCTURE['timestamp']],
                                                    data[TASK_STRUCTURE['pen_status']],
                                                    finding_status)
                point = Features.get_point(point, data[TASK_STRUCTURE['pen_status']], finding_status)
            else:
                domain = data[TASK_STRUCTURE['timestamp']]
        else:
            domain = data[TASK_STRUCTURE['pen_status']]
        # Dopo di che la il metodo calcolerà i valori della funzione richiesta i quali poi verranno usati per calcolare
        # traformate, cepstrum o derivate frazionarie
        function_result = function(point, domain, finding_status)
        return function_result

    """
        In questo metodo si calcolano le derivate frazionarie tenendo presnte ogni evetuale combianzione di funzione necessaria,
        asse, punti da considerare e grado di derivazione.
    """
    @staticmethod
    def __get_fractional_derivative(feature, command, data):
        function = command[PARAMETER_KEY][0]
        grade = command[PARAMETER_KEY][1]
        if len(command[PARAMETER_KEY]) > 2:
            status = command[PARAMETER_KEY][2]
            other_parameter = [function, status]
        else:
            other_parameter = [function]
        function = FeatureExtraction.__get_all_function_combination({
            AXIS_KEY: command[AXIS_KEY],
            PARAMETER_KEY: other_parameter
        }, data)
        fractional_derivative = Features.get_fractional_derivative(function, FeatureExtraction.__switch_grade(grade))
        return FeatureExtraction.__generate_result_dictionary(fractional_derivative, feature)

    @staticmethod
    def __switch_grade(grade_value):
        switcher = {
            'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10
        }
        return switcher.get(grade_value)

    @staticmethod
    def __function_switch(function_name):
        switcher = {
            'DIS': (FeatureExtraction.__get_function_displacement, 'pen_status'),
            'VEL': (FeatureExtraction.__get_function_velocity, 'timestamp'),
            'ACC': (FeatureExtraction.__get_function_acceleration, 'timestamp'),
            'JERK': (FeatureExtraction.__get_function_jerk, 'timestamp')
        }
        return switcher.get(function_name)

    @staticmethod
    def __get_function_displacement(point, pen_status=None, finding_status=None):
        if len(point.keys()) > 1:
            if finding_status is not None:
                function = Features.get_displacement_pen_status(point['x_axis'], point['y_axis'], pen_status, finding_status)
            else:
                function = Features.get_displacement(point['x_axis'], point['y_axis'])
        else:
            if finding_status is not None:
                function = Features.get_displacement_axis_pen_status(point['axis'], pen_status, finding_status)
            else:
                function = Features.get_displacement_axis(point['axis'])
        return function

    @staticmethod
    def __get_function_velocity(point, time_stamp, on_surface=None):
        space = FeatureExtraction.__get_function_displacement(point)
        return Features.get_displacement_velocity(space, time_stamp, bool(on_surface))

    @staticmethod
    def __get_function_acceleration(point, time_stamp, on_surface=None):
        velocity = FeatureExtraction.__get_function_velocity(point, time_stamp, bool(on_surface))
        return Features.get_displacement_acceleration(velocity, time_stamp, on_surface)

    @staticmethod
    def __get_function_jerk(point, time_stamp, on_surface=None):
        acceleration = FeatureExtraction.__get_function_acceleration(point, time_stamp, bool(on_surface))
        return Features.get_jerk(acceleration, time_stamp, on_surface)

    @staticmethod
    def __generate_result_dictionary(data, feature):
        result = {}
        for key, item in FeatureExtraction.__get_statistic_value(data).items():
            result[feature + key] = item
        return result
