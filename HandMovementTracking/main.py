import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('my_numbers_model.model')


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

