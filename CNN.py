import os
import time
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
import mnist


class CNN:
    def __init__(self, file_name, retrain = False):
        self.file_name = file_name
        self.retrain = retrain


        if not retrain:
            try:
                self.model = load_model(self.file_name)
            except:
                retrain = True

        if retrain:
            self.model = Sequential([
                Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax'),
            ])

            self.model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy']
            )

    def load(self):
        self.model = load_model(self.file_name)

    def save(self):
        self.model.save(self.file_name)

    def get_test(self):
        test_images = mnist.test_images()
        # test_images = (test_images / 255) - 0.5
        test_images = np.expand_dims(test_images, axis=3)
        return test_images

    def predict(self, x):
        x = (x / 255) - 0.5
        x = np.expand_dims(x, axis=3)
        x = np.array([x])
        predictions = self.model.predict(x)
        return np.argmax(predictions[0]), predictions[0]

    def train(self, epochs = 10, batch_size = 10):
        print("Training...")

        start = time.time()

        training_images = mnist.train_images()
        training_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        training_images = (training_images / 255.0) - 0.5
        test_images = (test_images / 255.0) - 0.5

        training_images = np.expand_dims(training_images, axis=3)
        test_images = np.expand_dims(test_images, axis=3)

        hist = self.model.fit(
            training_images,
            to_categorical(training_labels),
            batch_size=batch_size,
            epochs=epochs,
            validation_data = (test_images, to_categorical(test_labels))
        )

        end = time.time()
        elapsed = end - start

        acc = hist.history['acc'][-1]

        return acc, elapsed


        

