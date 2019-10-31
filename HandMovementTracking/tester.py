import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model('my_numbers_model.model')

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

predictions = model.predict([x_test])
count_predictions = len(predictions)

running = True

while running:
    print:("the model has " + str(count_predictions) + "predictions. Which one do you want to see. " )
    requested_index = input("====>")

    if requested_index is "quit":
        running = False
    
    requested_index = int(requested_index)
    
    if type(requested_index) is not int:
        print("Please enter a number without other characters.")
    elif requested_index >= count_predictions:
        print("Please enter an index below " + str(count_predictions) + ".")
    else:
        print("The model predicted that the following image is: " + str(np.argmax(predictions[requested_index])))
        plt.imshow(x_test[requested_index])
        plt.show()
