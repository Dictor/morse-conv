import tensorflow as tf
import numpy as np
from keras import optimizers, losses, metrics
from load_data import load_data
from model_tf import MorseCNN

learning_rate = 0.001
training_epochs = 20
batch_size = 100

xtr, ytr, xva, yva, xte, yte = load_data("jeonghyun.npz")
model = MorseCNN()
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss=losses.CategoricalCrossentropy(from_logits=True), metrics=[metrics.CategoricalAccuracy()])


def limit_onehot_array_to_tensor(arr):
    arr = np.argmax(arr, axis=1)
    arr[arr > 36] = 36
    arro = np.zeros((arr.size, arr.max() + 1))
    arro[np.arange(arr.size), arr] = 1
    return arro


ytr = limit_onehot_array_to_tensor(ytr)
yva = limit_onehot_array_to_tensor(yva)

xtr = tf.convert_to_tensor(xtr)
ytr = tf.convert_to_tensor(ytr)
xva = tf.convert_to_tensor(xva)
yva = tf.convert_to_tensor(yva)

model.fit(x=xtr, y=ytr, epochs=training_epochs,
          batch_size=batch_size, validation_data=(xva, yva))


def representative_dataset():
    yield [xte.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

with open('morse_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
