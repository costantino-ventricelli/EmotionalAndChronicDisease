# coding=utf-8

import numpy as np
from keras.layers import Layer
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers import dot
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences


class NetworkWithAttention:
    """
        La classe costruisce una rete profonda:
        - LSTM a 32 unità;
        - Denso a 32 unità;
        - Attention a 32 unità;
        - Denso a 3 unità.
    """
    def __init__(self, tensor_training, states_training, tensor_validation, states_validation, tensor_test, states_test):
        # Verifico che le i valori siano stati correttamente passati alla funzione
        if len(tensor_training) == 0 or len(states_training) == 0:
            raise ValueError("There are no element in training element")
        if len(tensor_validation) == 0 or len(states_validation) == 0:
            raise ValueError("There are no element in validation element")
        tensor_training = pad_sequences(tensor_training).astype(np.float32)
        tensor_validation = pad_sequences(tensor_validation).astype(np.float32)
        self.__tensor_test = pad_sequences(tensor_test).astype(np.float32)
        tensor_training, tensor_validation, self.__tensor_test = NetworkWithAttention.__normalize_tensor(tensor_training, tensor_validation, self.__tensor_test)
        states_training = np.array(states_training).astype(np.float32)
        states_validation = np.array(states_validation).astype(np.float32)
        self.__states_test = np.array(states_test).astype(np.float32)
        print("training_tensor:", np.shape(tensor_training), "training_label:", len(states_training))
        print("validation_tensor:", np.shape(tensor_validation), "validation_label:", len(states_validation))
        # Imposto il modello come sequenziale.
        self.__model = Sequential()
        # Aggiungo il layers bidirezionali alla rete di tipo LSTM, con i valori di kernel_inizialization, e recurrent_activation
        # impostati in modo da ottenere una distribuzione normale dei valori iniziali.
        self.__model.add(Bidirectional(LSTM(
            units=32,
            return_sequences=True), input_shape=(tensor_training.shape[1], 2)))
        # Aggiungo il layer denso che permetterà di modificare lo stato in ingresso ai layer successivi, utilizzano il
        # regolatore L2 con lambda=0.001
        self.__model.add(Dense(units=32, activation='relu'))
        # TODO: capire come funziona nel dettaglio questo layer
        self.__model.add(Attention(neurons=32))
        # Questo layer è il layer di output che fornità la distibuzione di classificazione sulle 3 classi
        self.__model.add(Dense(units=3, activation='softmax'))
        # Con questo comando vengono impostati i parametri di loss, l'ottimizzatore e la metrica che verrà valutata
        # durante il src e la validazione.
        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # A questo punto vengono avviati il src e la validazione del modello, impostando le epoche e la demensione del
        # batch.)
        self.__history = self.__model.fit(tensor_training, states_training,
                                          epochs=50,
                                          batch_size=1,
                                          validation_data=(tensor_validation, states_validation), verbose=1)
        # Vengono presentate le informazioni relative allo svolgimento di training e validazione.
        print(self.__model.summary())

    def test_model(self):
        print("test_tensor:", np.shape(self.__tensor_test), "test_label:", len(self.__states_test))
        self.__model.evaluate(self.__tensor_test, self.__states_test)
        predicted = self.__model.predict(self.__tensor_test)
        for i in range(len(predicted)):
            predicted[i] = np.where(predicted[i] < max(predicted[i]), 0, 1)
        return predicted

    @staticmethod
    def __normalize_tensor(training, validation, test):
        max_len = max(training.shape[1], validation.shape[1], test.shape[1])
        training = NetworkWithAttention.__fill_tensor(training, max_len)
        validation = NetworkWithAttention.__fill_tensor(validation, max_len)
        test = NetworkWithAttention.__fill_tensor(test, max_len)
        return training, validation, test

    @staticmethod
    def __fill_tensor(tensor, length):
        tensor = tensor.tolist()
        first_row = tensor[0]
        if len(first_row) < length:
            for _ in range(length - len(first_row)):
                first_row.append([0, 0])
            tensor[0] = first_row
        tensor = pad_sequences(tensor).astype(np.float32)
        return tensor


class Attention(Layer):

    def __init__(self, neurons=128, **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(
            hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(self.neurons, use_bias=False, activation='tanh', name='attention_vector')(
            pre_activation)
        return attention_vector
