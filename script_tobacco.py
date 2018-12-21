# coding: utf-8
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint


def confusion_matrix_NN(NN, x_test, y_test):
    """this function compute the confusion matrix for a given keras model (NN)
    """
    y_pred = NN.predict(np.array(x_test))
    y_test_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)
    matrix = confusion_matrix(y_pred_class, y_test_class)
    print(matrix)


def model_creation(dim_data):
    """this function create a neural network model then return it
        inputs:
            dim_data : the dimension of one of the data
    """
    dim = dim_data

    NN = Sequential()

    NN.add(Dense(512, activation="relu", input_shape=(dim,)))
    NN.add(Dropout(0.5))

    NN.add(Dense(10, activation='softmax'))

    NN.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])
    return NN


def main():
    # The file path where to find the data
    path_folder = "data/"

    # Opening metadata
    meta_data = pd.read_csv(path_folder + "Tobacco3482.csv")

    # Here I'm extracting the labels
    labels = np.unique(meta_data["label"])

    # Opening the data
    x = []
    y = []
    label_classes = {}
    i = 0
    for label in labels:
        path = path_folder + label + "/*.txt"
        print("Opening " + label + " data")
        files = glob.glob(path)
        for file in files:
            file_tmp = open(file, 'r')
            x.append(file_tmp.read())
            y.append(label)
            file_tmp.close()
        label_classes[i] = label
        i += 1
    print("Opened " + str(len(x)) + " documents, " +
          str(len(np.unique(y))) + " different classes")

    # Here I'm extracting the label
    labels = np.unique(meta_data["label"])

    # Treating the labels
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Splitting the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42)

    # Transforming the data into token representation
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)

    x_train_counts = vectorizer.transform(x_train)
    x_test_counts = vectorizer.transform(x_test)

    # Bayesian part

    # Creation of the model
    clf = MultinomialNB()
    print("Training Bayesian for baseline")
    # Training
    clf.fit(x_train_counts, y_train)

    print("Printing results for Bayesian")
    # Printing of the results
    print("Accuracy score : ")
    print(clf.score(x_test_counts, y_test))
    y_pred = clf.predict(x_test_counts)
    print("Confusion matrix :")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report :")
    print(classification_report(y_test, y_pred))
    print("Where classes are :")
    for label in label_classes:
        print(str(label) + " : " + label_classes[label])

    # Neural Network part
    # creation of the callbacks to save the best model

    checkpointer = ModelCheckpoint(
        filepath="weights.hdf5", verbose=1, save_best_only=True)
    callbacks = [checkpointer]

    # Extracting the size of the data
    dimension_data = len(x_train_counts.toarray()[0])

    # Creation of the model
    NN = model_creation(dimension_data)

    print("Training neural network, this may take while")
    # Training of the data
    NN.fit(x_train_counts.toarray(), to_categorical(y_train), epochs=10,
           validation_split=0.1, batch_size=128, callbacks=callbacks)

    # Loading the best model
    NN.load_weights('weights.hdf5')

    print("Printing neural network results")
    # Printing the results
    print("Accuracy score :")
    print(NN.evaluate(x_test_counts.toarray(), to_categorical(y_test))[1])

    print("Confusion matrix :")
    confusion_matrix_NN(NN, x_test_counts.toarray(), to_categorical(y_test))

    print("Classification report :")
    y_pred = NN.predict(np.array(x_test_counts.toarray()))
    y_test_class = np.argmax(to_categorical(y_test), axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)
    print(classification_report(y_test_class, y_pred_class))

    print("Where classes are :")
    for label in label_classes:
        print(str(label) + " : " + label_classes[label])

    print("The model is trained and the weights are saved at weights.hdf5, closing script")

if __name__ == "__main__":
    main()
