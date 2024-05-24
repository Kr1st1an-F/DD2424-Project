import tensorflow as tf
from keras.datasets import cifar10
from keras import regularizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np



def summarize_diagnostics(history):
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="test")
    plt.legend()
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="orange", label="test")
    plt.legend()
    plt.savefig("cnn_plot.png")
    plt.close()


def relu(x):
    return tf.maximum(0.0, x)


def he_initialization(shape, dtype=None):
    fan_in = np.prod(shape[:-1])
    stddev = np.sqrt(2.0 / fan_in)
    return tf.random.normal(shape, mean=0.0, stddev=stddev, dtype=dtype)


def one_hot(labels, dim):
    labels = labels.flatten()
    one_hot_labels = np.zeros((labels.size, dim))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


def softmax(logits):
    exp_logits = tf.exp(logits - tf.reduce_max(logits, axis=-1, keepdims=True))
    sum_exp_logits = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    probabilities = exp_logits / sum_exp_logits
    return probabilities


def normalize(X):
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    stddev = np.std(X, axis=(0, 1, 2), keepdims=True)
    return (X - mean) / stddev


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(
        inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation=relu, kernel_initializer=he_initialization, padding="same")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation=relu, kernel_initializer=he_initialization)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(10, activation=softmax)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[callback])
    _, acc = model.evaluate(X_test, y_test)
    print(f"Accuracy {acc * 100.0:.3f}")

    summarize_diagnostics(history)
