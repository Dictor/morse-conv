from tensorflow import keras
from keras.models import Model
from keras.layers import Conv1D, Activation, MaxPool1D, Dense, Input, Flatten
from keras import Sequential
keras.backend.clear_session()

class MorseCNN(Model):
    def __init__(self):
        super(MorseCNN, self).__init__()

        # L1
        # input : 1 channel, 64 width, N batch
        # after conv : 12 channel, 64 width, N batch
        # after pool : 24 channel, 32 width, N batch
        self.layer1 = Sequential([
            Conv1D(12, 3, strides=1, padding="same", input_shape=(64, 1)),
            Activation("relu"),
            MaxPool1D(2, 2)
        ])

        # L2
        # input : 12 channel, 32 width, N batch
        # after conv : 24 channel, 32 width, N batch
        # after pool : 24 channel, 16 width, N batch
        self.layer2 = Sequential([
            Conv1D(24, 3, strides=1, padding="same"),
            Activation("relu"),
            MaxPool1D(2, 2)
        ])
        
        xavier = keras.initializers.GlorotNormal(seed=None)
        # FC
        # input : 24 channel * 16 height
        # output : 37 (alpha 26 + num 10 + other 1)
        self.fc = Sequential([
            Flatten(),
            Dense(24*16, activation="relu", kernel_initializer=xavier),
            Dense(37, kernel_initializer=xavier)
        ])


    def call(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out