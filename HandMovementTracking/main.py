import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Common




common = Common()

(x_train, y_train), (x_test, y_test) = load_dataset()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = common.create_model()
model.fit(x_train, y_train, epochs=3)

common.save_model(model)

print("model has successfully been created and saved")

