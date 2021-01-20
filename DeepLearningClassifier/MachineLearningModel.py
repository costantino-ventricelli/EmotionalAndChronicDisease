# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from keras import initializers
from keras import regularizers
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as get_four_metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from DatasetManager.FileManager import FileManager

CLASS_CHANGE = 0.55


class MLModel:

    """
        Il metodo di init dell ML-model imposta la rete neurale ricorrente bidirezionale con layer LSTM, la rete è composta
        da 16 layer e da un layer denso che provvederà ad applicare le trasformazioni necessarie provenienti dai layer
        precedenti per influenzare lo stato che verrà trasmesso agli stati successivi, dopo di che il modello avvia il
        training ed il src sui dati passati al costruttore.
        @:param tensor_training: Contiene il tensore tridimensonale che verrà utilizzato come input per la fase di training
            della rete.
        @:param states_training: Contiene la lista degli stati dei pazienti passata alla rete neurale come secondo parametro
            per il training.
        @:param tensor_validation: Contiene il tensore tridimensionale che verrà utilizzato come input per la fase di
            validation.
        @:param states_validation: Contiene la lista degli stati dei pazienti passata alla rete neurale come secondo
            parametro per la validazione.
        @:param detrend: Indica al modello se effettuare il detrend dei tensori utilizzando la deviazione standard e il
            mean value.
    """
    def __init__(self, tensor_training, states_training, tensor_validation, states_validation, bach_size, detrend):
        self.__detrend = detrend
        if detrend:
            self.__mean = tensor_training.mean(axis=0)
            self.__std_dev = tensor_training.std(axis=0)
            tensor_training -= self.__mean
            tensor_training /= self.__std_dev
            tensor_validation -= tensor_validation.mean(axis=0)
            tensor_validation /= tensor_validation.std(axis=0)
        # Imposto il modello come sequenziale.
        self.__model = Sequential()
        # Aggiungo il layers bidirezionali alla rete di tipo LSTM, con i valori di kernel_inizialization, e recurrent_activation
        # impostati in modo da ottenere una distribuzione normale dei valori iniziali.
        self.__model.add(Bidirectional(LSTM(
            units=256,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            recurrent_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros'), merge_mode='concat'))
        # Aggiungo il layer denso che permetterà di modificare lo stato in ingresso ai layer successivi, utilizzano il
        # regolatore L2 con lambda=0.001
        self.__model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        # Con questo comando vengono impostati i parametri di loss, l'ottimizzatore e la metrica che verrà valutata
        # durante il src e la validazione.
        self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # A questo punto vengono avviati il src e la validazione del modello, impostando le epoche e la demensione del
        # batch.
        self.__history = self.__model.fit(tensor_training, states_training,
                                          epochs=20,
                                          batch_size=bach_size,
                                          validation_data=(tensor_validation, states_validation), verbose=1)
        # Vengono presentate le informazioni relative allo svolgimento di training e validazione.
        print(self.__model.summary())

    """
        Questo metodo permette di mostrare graficamente i risultati dell training e della validazione.
    """
    def show_summary_graph(self):
        train_label = "Training"
        validation_label = "Validating"
        plt.close('all')
        # Il primo grafico mostrerà i valori di loss
        plt.figure(0)
        plt.plot(self.__history.history['loss'], label=train_label)
        plt.plot(self.__history.history['val_loss'], label=validation_label)
        plt.title("Loss value")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Il secondo grafico mostrerà i valori di accuracy.
        plt.figure(1)
        plt.plot(self.__history.history['accuracy'], label=train_label)
        plt.plot(self.__history.history['val_accuracy'], label=validation_label)
        plt.title("Accuracy value")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.show()

    """
        Questo metodo è quello che sottoporrà la rete al src per valutarne successivamente i risultati.
        @:param tensor_test: Contiene il tensore tridimensionale che verrà passato come parametro per avviare la predizione
            dei risultati
        @:return: metodo restituisce una lista di 0-1 che identificano:
            1: healthy
            2: disease
            Inoltre restituisce una lista di valori che rappresentano i risultati della predizione effettuata dalla rete. 
    """
    def test_model(self, tensor_test, states):
        print(np.shape(tensor_test))
        if self.__detrend:
            tensor_test -= self.__mean
            tensor_test /= self.__std_dev
        # Avvio la predizione dei risultati passando alla rete il tensore di src come input.
        predicted_results = np.array(self.__model.predict(tensor_test))
        states_predicted = []
        # Scansiono la lista dei risultati e impostando nella lista degli stati il valore 1 se il risultato supera la
        # soglia di 0.55.
        for result in predicted_results:
            states_predicted.append(0 if result <= CLASS_CHANGE else 1)
        # Trasformo la lista dei risultati in un'array numpay.
        states_predicted = np.array(states_predicted).astype(np.int)
        evaluation_result = self.__model.evaluate(tensor_test, states)
        return states_predicted, evaluation_result

    """
        Questo metodo può essere avviato dopo aver effettuato la prima predizione sui dati e solo se si hanno a disposizone
        dati di confronto, in quanto permette di valutare la classificazione effettuata dalla rete, restituendo tutti i
        dati necessari per valutare la classificazione.
        @:param tensor_test: Contiene il tensore tridimensionale che è stato utilizzato per testare la rete.
        @:param states_test: Contiene la lista di stati conosciuti, che servirà per valutare la classificazione fatta.
        @:param predicted_result: Contiene la lista degli risultati ottenuti dalla predizione della rete neurale.
        @:param states_result: Contiene la lista dei risultati ottenuti valutando i risultati predetti dalla rete neurale.
        @:param test_list: Contiene la lista dei percorsi utilizzati per testare la rete neurale.
        @:param test_samples: Contiene una matrice che presenta i dati campionati utilizzati per testare la rete.
        @:return: Il metodo restituisce una tupla di 10 elementi:
            - evaluation_result: contiene i dati di loss e accuracy ottenuti dal src della rete;
            - accuracy: contiene il valore di accuratezza ottenuto dal src della rete;
            - precision: contiene il valore di precisione ottenuto dal src della rete;
            - recall: contiene il valore di recall ottenuto dal src della rete;
            - f_score: contiene il valore di f1_score ottenuto dal src della rete;
            - il numero di pazienti che sono stati classificati male;
            - accuracy_score: contiene i dati di accuratezza calcolati sui singoli file classificati;
            - precision_score: contiene i dati di precisione calcolati sui singoli file classificati;
            - recall_score: contiene i dati di recall calcolati sui singoli file classificati;
            - f1_score: contiene i dati di f1_score calcolati sui singoli file classificati.
            - wrong_paths: contiene la lista dei file classificati erroneamente.
    """
    def classify_results(self, tensor_test, states_test, predicted_results, states_predicted, test_list, test_samples):
        if self.__detrend:
            tensor_test -= self.__mean
            tensor_test /= self.__std_dev
        # Valuto il modello sui dati di src
        evaluation_result = self.__model.evaluate(tensor_test, states_test)
        # Calcolo l'accuratezza del src.
        accuracy = accuracy_score(states_test, states_predicted)
        # Calcolo precisione, recall e f_score della classificazione, utilizzando una media binaria in quanto ho due
        # classi di classificazione.
        precision, recall, f_score, _ = get_four_metrics(states_test, states_predicted, labels=[0, 1], average='macro')
        avg_predicting_samples = []
        avg_test_samples = []
        # Questo ciclo permette di calcolre la media su ogni sequenza di campioni utilizzati per il src del modello.
        for i in range(len(test_samples)):
            avg_predicting = 0
            avg_test = states_test[i * test_samples[i]]
            # Questo ciclo scansiona tutti i campioni del file calcolado la media di ogni file.
            for j in range(i * test_samples[i], (i * test_samples[i]) + test_samples[i]):
                avg_predicting += predicted_results[j]
            avg_test_samples.append(avg_test)
            avg_predicting_samples.append(1 if (avg_predicting / test_samples[i]) > CLASS_CHANGE else 0)
        # Genero il vettore di ground truth che mi permetterà di verificare i risultati ottenuti dalla calssificazione
        # di ogni file di src.
        print("Diagnosis: ", avg_test_samples)
        print("Predicted: ", avg_predicting_samples)
        health_ids, _ = FileManager.get_healthy_disease_list()
        healthy_wrong = 0
        disease_wrong = 0
        wrong_paths = []
        # Calcolo il numero di file classificati erroneamente e genero la lista dei file classificati in modo errato.
        for i in range(len(test_samples)):
            if avg_predicting_samples[i] != avg_test_samples[i]:
                wrong_paths.append(test_list[i])
                if FileManager.get_id_from_path(test_list[i]) in health_ids:
                    healthy_wrong += 1
                else:
                    disease_wrong += 1
        return evaluation_result, accuracy, precision, recall, f_score,\
            healthy_wrong + disease_wrong, \
            accuracy_score(avg_test_samples, avg_predicting_samples), precision_score(avg_test_samples, avg_predicting_samples, average='macro'), \
            recall_score(avg_test_samples, avg_predicting_samples, average='macro'), f1_score(avg_test_samples, avg_predicting_samples, average='macro'), \
            wrong_paths

    @staticmethod
    def evaluate_results(predicted_states, theoretical_states):
        accuracy = accuracy_score(theoretical_states, predicted_states)
        precision, recall, f_score, _ = get_four_metrics(theoretical_states, predicted_states, labels=[0, 1], average="macro")
        theoretical_states_average = 0
        for state in theoretical_states:
            theoretical_states_average += state
        theoretical_states_average = (1 if theoretical_states_average / len(theoretical_states) > CLASS_CHANGE else 0)
        return accuracy, precision, recall, f_score, theoretical_states_average

    """
        @:param tensor: Contiene il tensore su cui avviare la previsione dei risultati utilizzando il modello precedentemente
                        addestrato.
        @:return result: Restituisce un array contentente i valori binari di classificazione 1: malato, 0: sano.
    """
    def predict_result(self, tensor):
        predict_states = self.__model.predict(tensor)
        result = []
        for state in predict_states:
            result.append(0 if state <= CLASS_CHANGE else 1)
        return result
