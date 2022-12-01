from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np 
from keras import Sequential

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

class LSTM(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def train_model(self):
        inp1 = Input(shape = (30,))
        inp2 = Input(shape = (30,))
        emb1 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=30)(inp1)
        emb2 = Embedding(output_dim=300, weights = [self.embedding_matrix], trainable = False, input_dim=self.vocab_size, input_length=30)(inp2)
        concat = Concatenate(axis = -1)([emb1 + emb2 , emb1 - emb2, emb1 * emb2])
        lstm = tf.keras.layers.LSTM(150, return_sequences=False, dropout=0.1, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat)
        out = Dense(2, activation = "softmax")(lstm[2])
        self.model = Model(inputs = [inp1, inp2], outputs = out)
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)

    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)
