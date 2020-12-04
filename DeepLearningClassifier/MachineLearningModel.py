# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras import initializers
from keras import regularizers
from sklearn.metrics import precision_recall_fscore_support as get_four_metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from DeepLearningClassifier.FileManager import FileManager

CLASS_CHANGE = 0.55


class MLModel:

    """
        Il metodo di init dell ML-model imposta la rete neurale ricorrente bidirezionale con layer LSTM, la rete è composta
        da 16 layer e da un layer denso che provvederà ad applicare le trasformazioni necessarie provenienti dai layer
        precedenti per influenzare lo stato che verrà trasmesso agli stati successivi, dopo di che il modello avvia il
        training ed il test sui dati passati al costruttore.
        @:param tensor_training: Contiene il tensore tridimensonale che verrà utilizzato come input per la fase di training
            della rete.
        @:param states_training: Contiene la lista degli stati dei pazienti passata alla rete neurale come secondo parametro
            per il training.
        @:param tensor_validation: Contiene il tensore tridimensionale che verrà utilizzato come input per la fase di
            validation.
        @:param states_validation: Contiene la lista degli stati dei pazienti passata alla rete neurale come secondo
            parametro per la validazione.
    """
    def __init__(self, tensor_training, states_training, tensor_validation, states_validation):
        # Imposto il modello come sequenziale.
        self.__model = Sequential()
        # Aggiungo il layers bidirezionali alla rete di tipo LSTM, con i valori di kernel_inizialization, e recurrent_activation
        # impostati in modo da ottenere una distribuzione normale dei valori iniziali.
        self.__model.add(Bidirectional(LSTM(
            units=16,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            recurrent_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))))
        # Aggiungo il layer denso che permetterà di modificare lo stato in ingresso ai layer successivi, utilizzano il
        # regolatore L2 con lambda=0.001
        self.__model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        # Con questo comando vengono impostati i parametri di loss, l'ottimizzatore e la metrica che verrà valutata
        # durante il test e la validazione.
        self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # A questo punto vengono avviati il test e la validazione del modello, impostando le epoche e la demensione del
        # batch.
        self.__history = self.__model.fit(tensor_training, states_training,
                                          epochs=40,
                                          batch_size=128,
                                          validation_data=(tensor_validation, states_validation),
                                          verbose=2)
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
        Questo metodo è quello che sottoporrà la rete al test per valutarne successivamente i risultati.
        @:param tensor_test: Contiene il tensore tridimensionale che verrà passato come parametro per avviare la predizione
            dei risultati
        @:return: metodo restituisce una lista di 0-1 che identificano:
            1: healthy
            2: disease
            Inoltre restituisce una lista di valori che rappresentano i risultati della predizione effettuata dalla rete. 
    """
    def test_model(self, tensor_test):
        # Avvio la predizione dei risultati passando alla rete il tensore di test come input.
        predicted_results = np.array(self.__model.predict(tensor_test))
        states_predicted = []
        # Scansiono la lista dei risultati e impostando nella lista degli stati il valore 1 se il risultato supera la
        # soglia di 0.55.
        for result in predicted_results:
            states_predicted.append(0 if result <= CLASS_CHANGE else 1)
        # Trasformo la lista dei risultati in un'array numpay.
        states_predicted = np.array(states_predicted).astype(np.float)
        return states_predicted, predicted_results

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
            - evaluation_result: contiene i dati di loss e accuracy ottenuti dal test della rete;
            - accuracy: contiene il valore di accuratezza ottenuto dal test della rete;
            - precision: contiene il valore di precisione ottenuto dal test della rete;
            - recall: contiene il valore di recall ottenuto dal test della rete;
            - f_score: contiene il valore di f1_score ottenuto dal test della rete;
            - il numero di pazienti che sono stati classificati male;
            - accuracy_score: contiene i dati di accuratezza calcolati sui singoli file classificati;
            - precision_score: contiene i dati di precisione calcolati sui singoli file classificati;
            - recall_score: contiene i dati di recall calcolati sui singoli file classificati;
            - f1_score: contiene i dati di f1_score calcolati sui singoli file classificati.
            - wrong_paths: contiene la lista dei file classificati erroneamente.
    """
    def classify_results(self, tensor_test, states_test, predicted_results, states_predicted, test_list, test_samples):
        # Valuto il modello sui dati di test
        evaluation_result = self.__model.evaluate(tensor_test, states_test)
        # Calcolo l'accuratezza del test.
        accuracy = accuracy_score(states_test, states_predicted)
        # Calcolo precisione, recall e f_score della classificazione, utilizzando una media binaria in quanto ho due
        # classi di classificazione.
        precision, recall, f_score, _ = get_four_metrics(states_test, states_predicted, labels=['healthy', 'diseased'],
                                                         average='binary')
        avg_predicting_samples = []
        avg_test_samples = []
        # Questo ciclo permette di calcolre la media su ogni sequenza di campioni utilizzati per il test del modello.
        for i in range(len(test_samples)):
            avg_predicting = 0
            avg_test = 0
            # Questo ciclo scansiona tutti i campioni del file calcolado la media di ogni file.
            for j in range(i * test_samples[i], (i * test_samples[i]) + test_samples[i]):
                avg_test += states_test[j]
                avg_predicting += predicted_results[j]
            avg_test_samples.append(avg_test / test_samples[i])
            avg_predicting_samples.append(avg_predicting / test_samples[i])
        half_set = len(test_samples) // 2
        # Genero il vettore di ground truth che mi permetterà di verificare i risultati ottenuti dalla calssificazione
        # di ogni file di test.
        ground_truth = np.concatenate((np.zeros(half_set), np.ones(len(test_samples) - half_set)), axis=None)
        health_ids, _ = FileManager.get_healthy_disease_list()
        healthy_wrong = 0
        disease_wrong = 0
        wrong_paths = []
        # Calcolo il numero di file classificati erroneamente e genero la lista dei file classificati in modo errato.
        for i in range(len(test_samples)):
            if (avg_test_samples[i] == 1 and avg_predicting_samples[i] <= CLASS_CHANGE) \
                    or (avg_test_samples[i] == 0 and avg_predicting_samples[i] >= CLASS_CHANGE):
                wrong_paths.append(test_list[i])
                if FileManager.get_id_from_path(test_list[i]) in health_ids:
                    healthy_wrong += 1
                else:
                    disease_wrong += 1
        # Genero la lista dei dati che verrà confrontata con il ground truth per verificare la classificazione.
        samples_prevision_list = np.concatenate((
            np.zeros(half_set - healthy_wrong),
            np.ones(healthy_wrong),
            np.ones(half_set - disease_wrong),
            np.zeros(disease_wrong)
        ))
        return evaluation_result, accuracy, precision, recall, f_score,\
            healthy_wrong + disease_wrong, \
            accuracy_score(ground_truth, samples_prevision_list), precision_score(ground_truth, samples_prevision_list), \
            recall_score(ground_truth, samples_prevision_list), f1_score(ground_truth, samples_prevision_list), \
            wrong_paths
