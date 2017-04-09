import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import pickle
import os.path
import spacy
from tqdm import tqdm

# TO DO : STRING CLEANING, STOP WORDS, ...
'''
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
'''

def load_data_and_labels(FLAGS): #labels, query_CBOW, paragraph_CBOW, embedding_method
    """
    Loads data from file and generates labels.
    """
    if FLAGS.dataset_size == 'short':
        q_path = FLAGS.short_query_text
        p_path = FLAGS.short_paragraph_text
        y_path = FLAGS.short_labels
    elif FLAGS.dataset_size == 'medium':
        q_path = FLAGS.medium_query_text
        p_path = FLAGS.medium_paragraph_text
        y_path = FLAGS.medium_labels
    elif FLAGS.dataset_size == 'full':
        q_path = FLAGS.full_query_text
        p_path = FLAGS.full_paragraph_text
        y_path = FLAGS.full_labels
    else:
        print("Please define size of dataset to use")
        sys.exit()

    if os.path.isfile(q_path):
        print("Loading queries...")
        q = np.load(q_path)
    else:
        print("text file not present. Exiting ...")
        sys.exit()
    if os.path.isfile(p_path):
        print("Loading paragraphs...")
        p = np.load(p_path)
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
    Embedding matrix is of size vocab_size x embedding_size
    """
    print("Loading embeddings...")
    embs = ([x.split(" ") for x in open(path).read().strip().split("\n")])
    print("Creating embedding mapping matrix...")
    words = np.array([x[0] for x in embs])
    mat = np.array([x[1:] for x in embs]).astype(float)
    mapped_words = [x[0] for x in vocab.transform(words)]
    vocab_size = len(vocab.vocabulary_)
    emb_matrix = np.zeros((vocab_size,mat.shape[1]))
    set_words = set(mapped_words)
    for i in tqdm(range(vocab_size)):
        if i in set_words:
            emb_matrix[i]=mat[mapped_words.index(i)]
    return emb_matrix


'''
def embed_batch(q_batch, p_batch, vocab_proc, method_name='CBOW'):
    q_batch_embedding = embed(q_batch, vocab_proc)
    p_batch_embedding = embed(p_batch, vocab_proc)
    return q_batch_embedding,p_batch_embedding


def embed(data, vocab_proc, method_name='CBOW'):
    nlp = spacy.load('en')
    doc_vecs = []
    append_count = 0
    if method_name == 'CBOW':
        doc_embed = 'doc_embed_with_'+method_name
    for document in data:
        document = str(document)
        doc = nlp(document)
        word_vecs = []
        for word in doc:
            word_vec = word_text.vector
            word_vecs.append(word_vec)
        doc_vec = doc_embed_with_CBOW(word_vecs)
        doc_vecs.append(doc_vec)
        append_count+=1
    return doc_vecs


def doc_embed_with_CBOW(word_vecs):
    mean_vec = np.zeros([1, 300])
    for word_vec in word_vecs:
        # compute final vec
        mean_vec += word_vec
    mean_vec = mean_vec.mean(axis=0)
    return mean_vec


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
'''