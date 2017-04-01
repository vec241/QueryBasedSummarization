import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import pickle
import os.path
import spacy


def load_data_and_labels(FLAGS): #labels, query_CBOW, paragraph_CBOW, embedding_method
    """
    Loads data from file and generates labels.
    """

    #CBOW Calculation
    #if FLAGS.embedding_method == 'CBOW':
    #    print("Embedding with CBOW starts...")

    #Query Embedding
    if os.path.isfile(FLAGS.query_text):
        print("CBOW Embedding for query already present. Loading the embeddings directly...")
        q = np.load(FLAGS.query_text).tolist()
        print ("q shape before :",np.shape(q))
        q = [[x] for x in q]
        print ("q shape after : ",np.shape(q))
    else:
        print("text file not present. Exiting ...")
        sys.exit()
    if os.path.isfile(FLAGS.paragraph_text):
        print("CBOW Embedding for query already present. Loading the embeddings directly...")
        p = np.array(np.load(FLAGS.paragraph_text).tolist())
        p = [[x] for x in p]
    else:
        print("text file not present. Exiting ...")
        sys.exit()
    if not os.path.isfile(FLAGS.labels):
        print("File containing labels not present. Please check the directory.")
        sys.exit()
    else:
        y = np.array(np.load(FLAGS.labels).tolist())
    print("Finished loading data successfully.")
    return q, p, y

    #labels

    # Load data
    #y = np.array(np.load(labels).tolist())
    #q = np.array(np.load(query_CBOW).tolist())
    #p = np.array(np.load(paragraph_CBOW).tolist())


#method_name = 'CBOW'
#query_vecs = embed('query',method_name)
#paragraph_vecs = embed('para_text',method_name)



def embed_batch(q_batch,p_batch,method_name='CBOW'):
    print(len(q_batch))
    q_batch_embedding = embed(q_batch)
    p_batch_embedding = embed(p_batch)
    #print("q_batch_embedding shape :  ",np.shape(q_batch_embedding))
    return q_batch_embedding,p_batch_embedding
    #for i in q_batch:
    #    print(i)


def embed(data,method_name='CBOW'):
    nlp = spacy.load('en')
    doc_vecs = []
    word_vecs = []
    append_count = 0
    if method_name == 'CBOW':
            doc_embed = 'doc_embed_with_'+method_name
    print ("inside embed function with :",doc_embed)
    #for qu in tqdm(list(df[column_name])):
    print ("iterating over the data of length :", len(data))
    for document in data: #list(data) #list(short_df_1000[column_name]) #list(df[column_name]) #document in data_list:
        document = str(document)
        doc = nlp(document)
        for word in doc:
            word_vec = word.vector
            word_vecs.append(word_vec)
        doc_vec = doc_embed_with_CBOW(word_vecs)
        doc_vecs.append(doc_vec)
        #print ("append count :", append_count)
        append_count+=1
    return doc_vecs

def doc_embed_with_CBOW(word_vecs):
    mean_vec = np.zeros([1, 300])
    for word_vec in word_vecs:
        # compute final vec
        mean_vec += word_vec
    mean_vec = mean_vec.mean(axis=0)
    return mean_vec



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
