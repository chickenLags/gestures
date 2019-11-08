import numpy as np
import tensorflow as tf



class Common:

DATASET = 'mnist.npz'
MODEL_NAME = 'my_numbers_model.h5'

    def __init__(self, dataset='mnist.npz', model_name=my_numbers_model):
        self.DATASET = dataset
        self.MODEL_NAME = model_name + '.h5'


    def create_model(self):
        model = tf.keras.models.Sequential()
    
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    
        return model
    
    
    def load_dataset(self):
        with np.load(DATASET) as data:
          x_train = data['x_train']
          y_train = data['y_train']
          x_test = data['x_test']
          y_test = data['y_test']
    
        return (x_train, y_train), (x_test, y_test)
    
    def restore_model(self, model, x_test, y_test):
        loss, acc = model.evaluate(x_test, y_test, verbose=2)
        print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
    
        model = tf.keras.models.load_model(MODEL_NAME)
    
        loss,acc = model.evaluate(x_test, y_test, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    
        return model
    
    def save_model(self, model):
        model.save(MODEL_NAME)