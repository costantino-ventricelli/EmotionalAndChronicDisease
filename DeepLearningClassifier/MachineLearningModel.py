# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from keras import initializers
from keras import regularizers
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as get_four_metrics
from sklearn.metrics import precision_score

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
            parametro per la validazione
    """
    def __init__(self, tensor_training, states_training, tensor_validation, states_validation):
        # Verifico che le i valori siano stati correttamente passati alla funzione
        if len(tensor_training) == 0 or len(states_training) == 0:
            raise ValueError("There are no element in training element")
        if len(tensor_validation) == 0 or len(states_validation) == 0:
            raise ValueError("There are no element in validation element")
        # Imposto il modello come sequenziale.
        self.__model = Sequential()
        # Aggiungo il layers bidirezionali alla rete di tipo LSTM, con i valori di kernel_inizialization, e recurrent_activation
        # impostati in modo da ottenere una distribuzione normale dei valori iniziali.
        self.__model.add(Bidirectional(LSTM(
            units=16,
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
                                          epochs=40,
                                          batch_size=128,
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
    def test_model(self, tensor_test, theoretical_states):
        # Avvio la predizione dei risultati passando alla rete il tensore di src come input.
        predicted_results = np.array(self.__model.predict(tensor_test))
        states_predicted = []
        # Scansiono la lista dei risultati e impostando nella lista degli stati il valore 1 se il risultato supera la
        # soglia di 0.55.
        for result in predicted_results:
            states_predicted.append(0 if result <= CLASS_CHANGE else 1)
        # Trasformo la lista dei risultati in un'array numpay.
        states_predicted = np.array(states_predicted).astype(np.int)
        evaluation_result = self.__model.evaluate(tensor_test, theoretical_states)
        predicted_state_average = 0
        for state in states_predicted:
            predicted_state_average += state
        predicted_state_average = (1 if predicted_state_average / len(states_predicted) > CLASS_CHANGE else 0)
        return states_predicted, evaluation_result, predicted_state_average

    """
        Questi metodo permettono di ottenere le metriche di misurazione delle performance, nel primo abbiamo una tupla a
        quattro elementi:
            - (accuracy, precision, recall, f_score);
            - (accuracy, precision).
        Il secondo metodo non viene mai utilizzato, ma potrebbe essere utile in caso di necessità quando le altre due 
        metriche non possono essere calcolate per mancanza di dati.
    """
    @staticmethod
    def evaluate_results(predicted_states, theoretical_states):
        accuracy = accuracy_score(theoretical_states, predicted_states)
        precision, recall, f_score, _ = get_four_metrics(theoretical_states, predicted_states,
                                                         labels=[0, 1], average="macro")
        return accuracy, precision, recall, f_score

    @staticmethod
    def get_accuracy_precision(predicted_states, theoretical_states):
        accuracy = accuracy_score(theoretical_states, predicted_states)
        precision = precision_score(theoretical_states, predicted_states)
        return accuracy, precision

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
