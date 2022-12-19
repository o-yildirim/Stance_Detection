import pandas as pd
import numpy as np
from numpy import asarray
import re

from keras.preprocessing.text import  Tokenizer
from keras.utils import pad_sequences

def get_data(train, val, test):  # Parameters are paths. Returns the data in the specified paths.
    training_pd = pd.DataFrame()
    for path in train:
        sep = ','
        if path[-3:] == "txt":  # Txt files are seperated by tabs, not commas.
            sep = '\t'
        temp_pd = pd.read_csv(path, sep=sep)
        training_pd = pd.concat([training_pd, temp_pd])
    training_pd = training_pd.where(training_pd[
                                        'Stance'] != 'NONE').dropna()  # We remove NONE labels to be able to compare our results with P-stance.
    X_train = training_pd[['Tweet']].to_numpy().flatten()
    y_train = training_pd[['Stance']].to_numpy().flatten()
    X_train = preprocess_data(X_train)
    y_train = convert_labels(y_train)

    validation_pd = pd.DataFrame()
    for path in val:
        sep = ','
        if path[-3:] == "txt":
            sep = '\t'
        temp_pd = pd.read_csv(path, sep=sep)
        validation_pd = pd.concat([validation_pd, temp_pd])
    validation_pd = validation_pd.where(validation_pd['Stance'] != 'NONE').dropna()
    X_validation = validation_pd[['Tweet']].to_numpy().flatten()
    y_validation = validation_pd[['Stance']].to_numpy().flatten()
    X_validation = preprocess_data(X_validation)
    y_validation = convert_labels(y_validation)

    all_corpus = np.concatenate((X_train, X_validation), axis=0)

    t = Tokenizer()
    t.fit_on_texts(all_corpus)

    encoded_all = t.texts_to_sequences(all_corpus)
    max_length = find_max_sent_length(encoded_all)

    encoded_X_train = t.texts_to_sequences(X_train)
    encoded_X_validation = t.texts_to_sequences(X_validation)
    padded_X_train = pad_sequences(encoded_X_train, maxlen=max_length, padding='post')
    padded_X_validation = pad_sequences(encoded_X_validation, maxlen=max_length, padding='post')

    test_pd = pd.DataFrame()
    for path in test:
        sep = ','
        if path[-3:] == "txt":
            sep = '\t'
        temp_pd = pd.read_csv(path, sep)
        test_pd = pd.concat([test_pd, temp_pd])
    test_pd = test_pd.where(test_pd['Stance'] != 'NONE').dropna()
    X_test = test_pd[['Tweet']].to_numpy().flatten()
    y_test = test_pd[['Stance']].to_numpy().flatten()
    X_test = preprocess_data(X_test)
    y_test = convert_labels(y_test)

    encoded_X_test = t.texts_to_sequences(X_test)
    padded_X_test = pad_sequences(encoded_X_test, maxlen=max_length, padding='post')

    return padded_X_train, y_train, padded_X_validation, y_validation, padded_X_test, y_test, t.word_index, max_length


# Data preprocessing.
def preprocess_data(data):
    cleaned_data = []
    for sent in data:
        cleaned_sent = re.sub("@[A-Za-z0-9_]+", "", sent)  # Removing mentions like @abc.
        cleaned_sent = cleaned_sent.replace('#SemST', '')  # Removing SemEval tag.

        cleaned_data = np.append(cleaned_data, cleaned_sent)
    return cleaned_data


# Finding the length of the longest sentence in "sent_list" in terms of words.
def find_max_sent_length(sent_list):
    max_len = 0
    for sent in sent_list:
        if len(sent) > max_len:
            max_len = len(sent)

    return max_len


# Loads the whole embedding into memory (pre-defined word embeddings from GloVe)
def load_glove_vectors(path,word_dict):
    embeddings_index = {}  # Embeddings will be kept here.
    f = open(path,encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        if word not in word_dict.keys():
            continue
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embeddings_found = len(embeddings_index)
    print('Loaded %s word vectors.' % embeddings_found)

    # The module below assigns 0 vector to every word that does not exist in GloVe word vectors (for instance hashtags etc.)
    embedding_dimension = len(list(embeddings_index.values())[0])  # Getting the first word vector and checking its size to determine embedding dimensions.
    for word in word_dict.keys():
        if word not in embeddings_index.keys():
            embeddings_index[word] = np.zeros(embedding_dimension)  # Setting non-existing words to zero vector.
            # embeddings_index[word] = np.random.rand(1,embedding_dimension) #Setting non-existing words to a random vector.

    print('Initialized zero vectors for %s words.' % (len(embeddings_index) - embeddings_found))

    return embeddings_index, embedding_dimension


# Creates and returns a [vocab_size x embedding dimension] size embedding matrix.
def create_embedding_matrix(embeddings, vocab_size, embedding_dimension, word_dict):
    embedding_matrix = np.zeros((vocab_size, embedding_dimension))
    for word, i in word_dict.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Converting string labels into one-hot encoding format.
def convert_labels(labels):
    label_conversion_dict = {'FAVOR': [1, 0], 'AGAINST': [0, 1]}
    i = 0
    converted_labels = np.zeros(shape=(len(labels), 2))
    for label in labels:
        converted_labels[i] = label_conversion_dict[label]
        i += 1

    return converted_labels