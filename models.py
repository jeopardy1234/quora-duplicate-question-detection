from keras.layers import *
from keras.models import Model
import tensorflow as tf
import numpy as np 
from keras import Sequential
from keras.optimizers import adam_v2

class NeuralModels:
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.vocab_size = vocab_size
        self.embedding_matrix = emb_mat
        self.model = Sequential()

class CBOW(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def fit(self,xtrain, xval, ytrain, yval):
        self.model.add(Dense(300, input_shape = (900,), activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(200, activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(100, activation = "sigmoid"))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(2, activation = "softmax"))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)
        self.model.fit(xtrain, ytrain, epochs = self.epochs, validation_data = (xval, yval), batch_size = 64, verbose = 1)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class LsTM(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (128,))
        inp2 = Input(shape = (128,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
        lstm = LSTM(150, return_sequences=False, dropout=0.2, return_state=True)(concat)
        out = Dense(2, activation = "softmax")(lstm[2])
        self.model = Model(inputs = [inp1, inp2], outputs = out)
        self.model.compile(loss = self.loss, optimizer = adam_v2.Adam(learning_rate=0.0008), metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class BiLSTM(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (128,))
        inp2 = Input(shape = (128,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=128)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
        out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, kernel_regularizer='l2', dropout=0.1, return_sequences=True))(concat)
        out = tf.keras.backend.mean(out, axis=1, keepdims=False)
        output = tf.keras.layers.Dense(2, kernel_regularizer='l2', activation='softmax')(out)
        self.model = Model(inputs = [inp1, inp2], outputs = output)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

class LSTM_Attention(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (30,))
        inp2 = Input(shape = (30,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=30)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=30)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
        lstm = tf.keras.layers.LSTM(150, return_sequences=True, dropout=0.1, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)
        attention = tf.keras.layers.Attention()([lstm[0], lstm[0]])
        out = Dense(2, activation = "softmax")(attention)
        self.model = Model(inputs = [inp1, inp2], outputs = out)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)

def BiLSTM_Attention(emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
    inp1 = Input(shape = (30,))
    inp2 = Input(shape = (30,))
    emb1 = Embedding(output_dim=300, weights = [emb_mat], trainable = False, input_dim=vocab_size, input_length=30)(inp1)
    emb2 = Embedding(output_dim=300, weights = [emb_mat], trainable = False, input_dim=vocab_size, input_length=30)(inp2)
    concat = Concatenate(axis = -1)([emb1 + emb2, emb1 - emb2, emb1 * emb2])
    bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True, dropout=0.1, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(concat)
    attention = tf.keras.layers.Attention()([bilstm[0], bilstm[0]])
    out = Dense(2, activation = "softmax")(attention)
    model = Model(inputs = [inp1, inp2], outputs = out)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model