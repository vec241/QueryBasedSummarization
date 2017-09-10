"""
Gathers functions required to perform operations on data : load, clean, create batches ...
"""

import numpy as np
import pandas as pd
import re
import itertools
import sys
from collections import Counter
import pickle
import os.path
import spacy
from tqdm import tqdm


def clean_str(string):
    """
    Tokenization/string cleaning
    Inspired from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def cut(s):
    """
    Cuts questions at max question length
    """
    temp = s.split(" ")
    temp_max_question_length = s[:FLAGS.max_question_length]
    return " ".join(temp_max_question_length)


def load_data_and_labels(FLAGS): #labels, query_CBOW, paragraph_CBOW, embedding_method
    """
    Loads data from file and generates labels.
    """

    if FLAGS.dataset_size == 'short_balanced':
        q_path = FLAGS.short_balanced_query_text
        p_path = FLAGS.short_balanced_paragraph_text
        y_path = FLAGS.short_balanced_labels
    elif FLAGS.dataset_size == 'full_balanced':
        q_path = FLAGS.full_balanced_query_text
        p_path = FLAGS.full_balanced_paragraph_text
        y_path = FLAGS.full_balanced_labels
    elif FLAGS.dataset_size == 'medium_balanced':
        q_path = FLAGS.medium_balanced_query_text
        p_path = FLAGS.medium_balanced_paragraph_text
        y_path = FLAGS.medium_balanced_labels
    else:
        print("Please define size of dataset to use")
        sys.exit()

    if os.path.isfile(q_path):
        print("Loading and cleaning queries...")
        q = np.load(q_path)

        if FLAGS.data_cleaning_flag is True:
            q = [clean_str(x) for x in q]
            print("Queries cleaned !")
    else:
        print("text file not present. Exiting ...")
        sys.exit()

    if os.path.isfile(p_path):
        print("Loading and cleaning paragraphs...")
        p = np.load(p_path)
        if FLAGS.data_cleaning_flag is True:
            p = [clean_str(x) for x in p]
            print("Paragraph cleaned !")
    else:
        print("text file not present. Exiting ...")
        sys.exit()

    if not os.path.isfile(y_path):
        print("File containing labels not present. Please check the directory.")
        sys.exit()
    else:
        y = np.load(y_path)

    print("Finished loading data successfully.")
    return q, p, y


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_embeddings(path,vocab):
    """
    Loads Glove emeddings and returns an embedding matrix corresponding to the vocabulary.
    Embedding matrix is of size vocab_size * embedding_size
    """
    print("Loading embeddings...")
    embs = ([x.split(" ") for x in open(path).read().strip().split("\n")])
    print("Creating embedding mapping matrix...")
    words = np.array([x[0] for x in embs])
    words_list = words.tolist()
    state_index = words_list.index('state')
    print("index of state  : ",state_index)
    print("glove words[:10]  : ", words[:10])
    mat = np.array([x[1:] for x in embs]).astype(float)
    print("glove vector 'word : state' : ", mat[state_index])
    mapped_words = [x[0] for x in vocab.transform(words)]
    print("glove mapped_words",mapped_words[:100])   #lots of 0 for special symbols
    vocab_size = len(vocab.vocabulary_)
    emb_matrix = np.zeros((vocab_size,mat.shape[1]))
    set_words = set(mapped_words)
    print("mapped_words.index[0] :",mapped_words.index(0))
    print("mat[0] :",mat[mapped_words.index(0)])
    print("words[0] :",words[mapped_words.index(0)])
    for i in range(vocab_size):
        if i in set_words:
            emb_matrix[i]=mat[mapped_words.index(i)]
    return emb_matrix
