import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPool2D, Dropout, Flatten, Dense, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

image_dimension = (32, 32, 1)


def get_data_list(path):
    return len(os.listdir(path)), sorted(os.listdir(path))


def get_labeled_data(path):
    images = []
    labels = []
    num_classes, data_list = get_data_list(path)

    print("Importing classes...")
    for i in range(num_classes):
        image_list = os.listdir(path + "/" + str(i))
        for j in image_list:
            img = cv2.imread(path + "/" + str(i) + "/" + j)
            img = cv2.resize(img, (image_dimension[0], image_dimension[1]))
            images.append(img)
            labels.append(i)
    print("Classes imported.")
    return images, labels, len(np.unique(labels))


def split_data(images, labels, num_classes, test_size, validation_size):
    images = np.array(images)
    labels = np.array(labels)
    num_samples = []

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    for i in range(num_classes):
        num_samples.append(len(np.where(y_train == i)[0]))
    return X_train, X_test, X_validation, y_train, y_test, y_validation, num_samples


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)  # equalize the lighting of the images
    image = image / 255
    return image


def preprocess(X):
    X = np.array(list(map(preprocess_image, X)))
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))  # required for the neural network
    return X


def augment_data(X_train):
    data_generator = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        shear_range=0.1,
                                        rotation_range=10)
    data_generator.fit(X_train)
    return data_generator


def encode(y, num_classes):
    return to_categorical(y, num_classes)


def cnn_model(num_classes):
    num_filters = 60
    size_first_filter = (5, 5)
    size_second_filter = (3, 3)
    size_pool = (2, 2)
    num_node = 500

    model = Sequential()
    model.add((Conv2D(num_filters, size_first_filter, input_shape=image_dimension, activation='relu')))
    model.add((Conv2D(num_filters, size_first_filter, activation='relu')))
    model.add((MaxPool2D(pool_size=size_pool)))
    model.add((Conv2D(num_filters // 2, size_second_filter, activation='relu')))
    model.add((Conv2D(num_filters // 2, size_second_filter, activation='relu')))
    model.add((MaxPool2D(pool_size=size_pool)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def run_model(model, X_train, y_train, X_validation, y_validation):
    batch_size = 20
    epochs = 10
    data_generator = augment_data(X_train)
    history = model.fit(data_generator.flow(X_train, y_train,
                                            batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_validation, y_validation),
                        shuffle=1
                        )
    return history, model


def plot_results(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.show()


def display_score(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score = ', score[0])
    print('Accuracy = ', score[1])


def save_model(model, model_name):
    model.save(model_name+".model", save_format="h5")

# d, l = get_labeled_data("../resources")
