from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np 

class NeuralModels:
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.vocab_size = vocab_size
        self.embedding_matrix = emb_mat
        self.model = None

class CBOW(NeuralModels):
    def __init__(self, emb_mat, vocab_size = -1, loss = "binary_crossentropy", epochs = 10, optimizer = "adam", metrics = ["accuracy"]):
        super().__init__(emb_mat, vocab_size, loss, epochs, optimizer, metrics)
    
    def fit(self,xtrain, xval, ytrain, yval):
        xtrain1 = xtrain[0]; xtrain2 = xtrain[1]
        xval1 = xval[0]; xval2 = xval[1]

        ip1 = Input(shape=(xtrain1.shape[1],))
        ip2 = Input(shape=(xtrain2.shape[1],))

        emb1 = Embedding(self.vocab_size, 300, input_length=xtrain1.shape[1], weights=[self.embedding_matrix], trainable=True)(ip1)
        emb2 = Embedding(self.vocab_size, 300, input_length=xtrain2.shape[1], weights=[self.embedding_matrix], trainable=True)(ip2)

        concat_layer = concatenate([emb1 + emb2 , emb1 - emb2, emb1 * emb2], axis = -1)
        concat_layer = tf.keras.backend.sum(concat_layer, axis = 1)

        layer1 = Dense(300, activation='relu')(concat_layer)
        layer2 = Dense(200, activation='relu')(layer1)
        layer3 = Dense(100, activation='relu')(layer2)
        output_layer = Dense(1, activation='softmax')(layer3)
        self.model = Model(inputs=[ip1, ip2], outputs=output_layer)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.fit([xtrain1, xtrain2], ytrain, epochs=self.epochs, validation_data=([xval1, xval2], yval))
    
    def get_model_summary(self):
        self.model.summary()
    
    def predict(self, xtest):
        return self.model.predict(xtest)
