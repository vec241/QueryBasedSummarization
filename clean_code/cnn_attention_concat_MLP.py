"""
Code for the CNN + attention + concat + MLP model (cf report)
"""


import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, num_classes, vocab_size, embedding_size, max_length, vocab_proc, filter_sizes, num_filters,
      l2_reg_lambda=0.0, use_emb=True):

        # Placeholders for input, embedding matrix for unknown words and dropout
        self.input_q = tf.placeholder(tf.int32, [None, max_length], name="input_q")
        self.input_p = tf.placeholder(tf.int32, [None, max_length], name="input_p")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.W_emb = tf.placeholder(tf.float32,[vocab_size, embedding_size], name="emb_pretrained")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding_text"):

            '''
            # Following block only useful if we want to train embeddings for words not in vocab
            if use_emb:
                self.train_W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W_train")
                # Build de matrix of embeddings of words in the vocab.
                # If word exists in pre-trained embeddings, take this embedding.
                # If not, create a random embedding
                self.W = tf.where(tf.equal(tf.reduce_sum(tf.abs(self.W_emb),1),0), self.train_W, self.W_emb)
            else:
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            '''

            self.W = self.W_emb

            # Map word IDs to word embeddings
            self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
            self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)
            self.input_q_emb_expanded = tf.expand_dims(self.input_q_emb, -1)
            self.input_p_emb_expanded = tf.expand_dims(self.input_p_emb, -1)

        # Process p and q through convolution layer
        with tf.name_scope("conv_maxpool_q"):
            # For question, we do convolution + max pooling to get 1 vector per question
            conv_outputs_q = self.convolution(self.input_q_emb_expanded, embedding_size, max_length, filter_sizes, num_filters)
            print('conv_outputs_q', conv_outputs_q)
            self.h_q = self.max_pooling(conv_outputs_q, max_length, filter_sizes, num_filters)
            print('self.h_q', self.h_q)
        with tf.name_scope("conv_att_p"):
            # For paragraph, we do convolution and then apply attention using question vector
            conv_outputs_p = self.convolution(self.input_p_emb_expanded, embedding_size, max_length, filter_sizes, num_filters)
            print('conv_outputs_p', conv_outputs_p)
            self.h_p = self.attention(self.h_q, conv_outputs_p, max_length, filter_sizes, num_filters)
            print('self.h_p', self.h_p)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_q_drop = tf.nn.dropout(self.h_q, self.dropout_keep_prob)
            self.h_p_drop = tf.nn.dropout(self.h_p, self.dropout_keep_prob)

        # Keeping track of l2 regularization loss (optional)
        #l2_loss = tf.constant(0.0)

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = self.h_q_drop.get_shape().as_list()[1] + self.h_p_drop.get_shape().as_list()[1] # we are going to concat paragraph and question
        print('n_input', n_input)
        n_classes = num_classes

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.concatenated_input = tf.concat([self.h_q_drop, self.h_p_drop], 1, name="concatenated_input")
            print("self.concatenated_input : ", self.concatenated_input)
            self.scores = self.multilayer_perceptron(self.concatenated_input,
                            n_input, n_hidden_1, n_hidden_2, n_classes, self.dropout_keep_prob)
            function_to_score = lambda x : x + (10.0**(-4))  # Where `f` instantiates myCustomOp.
            self.scores = tf.map_fn(function_to_score, self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.y_true = tf.argmax(self.input_y, 1, name="y_true")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores, name = "losses")
            self.loss = tf.reduce_mean(losses,0,name = 'loss_sub') #+ l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1),name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def convolution(self, input_expanded, embedding_size, max_length, filter_sizes, num_filters):
        '''
        Given an input of size [batch, doc_length, embedding_size, in_channels (here = 1)],
        Given filter_sizes (default 3,4,5) and num_filters per filter_sizes (default 128),
        Returns a list of num_filters convoluted outputs of size [batch, doc_length, 1, out_channels = num_filters]
        '''
        # Create a convolution layer for each filter size
        conv_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input_expanded,
                    W,
                    #strides=[1, 1, 1, 1],
                    strides=[1, 1, embedding_size, 1],
                    #padding="VALID",
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                conv_outputs.append(h)
        return conv_outputs


    def max_pooling(self, conv_outputs, max_length, filter_sizes, num_filters):
        '''
        Given a list of num_filters convoluted outputs of size [batch, doc_length, 1, out_channels = num_filters],
        Returns tensor pooled over doc length of size [batch, num_filters * number of different filter_sizes]
        '''
        # Create a maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("maxpool-%s" % filter_size):
                # Maxpooling over the outputs
                h = conv_outputs[i]
                pooled = tf.nn.max_pool(
                    h,
                    #ksize=[1, sequence_length - filter_size + 1, 1, 1], #if convolution padding="VALID"
                    ksize=[1, max_length, 1, 1], #if convolution padding="SAME"
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        h_pool = tf.concat(pooled_outputs, 3)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat

    def attention(self, query_vector, paragraph, max_length, filter_sizes, num_filters):
        # Create embedding of paragraph using attention from query (embedded with CBOW)
        paragraph = tf.concat(paragraph, 3)
        num_filters_total = num_filters * len(filter_sizes)
        paragraph = tf.reshape(paragraph, [-1, max_length, num_filters_total])
        query_vector_expanded = tf.expand_dims(query_vector, 1)
        alphas = tf.multiply(query_vector_expanded, paragraph)
        alphas = tf.reduce_sum(alphas, 2, name="alphas")
        norm_alphas = tf.nn.softmax(logits=alphas, name="norm_alphas")
        norm_alphas_expanded = tf.expand_dims(norm_alphas, 2)
        h_attention = tf.multiply(norm_alphas_expanded, paragraph)
        h_attention = tf.reduce_sum(h_attention, 1)
        return h_attention


    def nn_layer(self, x, W_shape, bias_shape, dropout_keep_prob):
        """
        Implements one layer of the MLP
        """
        W = tf.get_variable("weights", W_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", bias_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        W = tf.nn.dropout(W, dropout_keep_prob)
        out_lay = tf.add(tf.matmul(x, W), b)
        return out_lay, W


    def multilayer_perceptron(self, x, n_input, n_hidden_1, n_hidden_2, n_classes, dropout_keep_prob):
        """
        Implements the MLP.
        Defining self.[various variables] for visualization / debugging purpose
        """
        with tf.variable_scope("layer_1"):
            out_lay1,W1 = self.nn_layer(x, [n_input, n_hidden_1], [n_hidden_1], dropout_keep_prob)
            self.W1 = W1
            out_lay1 = tf.nn.relu(out_lay1)
            self.outlay1 = out_lay1
        with tf.variable_scope("layer_2"):
            out_lay2,W2 = self.nn_layer(out_lay1, [n_hidden_1, n_hidden_2], [n_hidden_2], dropout_keep_prob)
            self.W2 = W2
            out_lay2 = tf.nn.relu(out_lay2)
            self.outlay2 = out_lay2
        with tf.variable_scope("out_lay"):
            out_lay,W3 = self.nn_layer(out_lay2, [n_hidden_2, n_classes], [n_classes], dropout_keep_prob)
            self.W3 = W3
        self.outlay = out_lay
        return out_lay
