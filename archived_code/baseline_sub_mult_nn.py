import tensorflow as tf
import numpy as np


class Model(object):
    """
    Baseline for query-based paragraph classification (determine if the paragraph
    is relevant to the query).
    We implement a MLP / single layer on a vector a where a is the concatenation
    of the pointwise multiplication and the pointwise difference of the paragraph
    and the question embeddings.
    """
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters,
      l2_reg_lambda=0.0, vocab_size=None):

        # Placeholders for input and dropout
        self.input_q = tf.placeholder(tf.float32, [None, embedding_size], name="input_q")
        self.input_p = tf.placeholder(tf.float32, [None, embedding_size], name="input_p")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = embedding_size*2 # we are going to concat paragraph and question
        n_classes = num_classes

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size*2, num_classes],  #embedding_size,
                initializer=tf.contrib.layers.xavier_initializer())

            dif = tf.subtract(self.input_p,self.input_q,name="dif")
            print(dif)
            dif_point_mul = tf.multiply(dif,dif,name ="dif_point_mul")
            print(dif_point_mul)
            pq_point_mul = tf.multiply(self.input_p,self.input_q,name="pq_point_mul")
            print(pq_point_mul)
            self.concatenated_input = tf.concat([dif_point_mul,pq_point_mul], 1,name="concatenated_input")
            print(self.concatenated_input)
            #W = tf.nn.dropout(W, self.dropout_keep_prob)
            #self.scores = tf.matmul(self.concatenated_input,W, name='scores')  #tf.reshape(W,[embedding_size,num_classes])

            self.scores = self.multilayer_perceptron(self.concatenated_input,
                n_input, n_hidden_1, n_hidden_2, n_classes, self.dropout_keep_prob)
            print(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.predictions)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            print(self.scores)
            print(self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores, name = "losses")
            print(losses)
            self.loss = tf.reduce_mean(losses,0,name = 'loss_sub') #+ l2_reg_lambda * l2_loss
            print(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1),name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def nn_layer(self, x, W_shape, bias_shape, dropout_keep_prob):
        W = tf.get_variable("weights", W_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", bias_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        W = tf.nn.dropout(W, dropout_keep_prob)
        out_lay = tf.add(tf.matmul(x, W), b)
        return out_lay

    def multilayer_perceptron(self, x, n_input, n_hidden_1, n_hidden_2, n_classes, dropout_keep_prob):
        with tf.variable_scope("layer_1"):
            out_lay1 = self.nn_layer(x, [n_input, n_hidden_1], [n_hidden_1], dropout_keep_prob)
            out_lay1 = tf.nn.relu(out_lay1)
        with tf.variable_scope("layer_2"):
            out_lay2 = self.nn_layer(out_lay1, [n_hidden_1, n_hidden_2], [n_hidden_2], dropout_keep_prob)
            out_lay2 = tf.nn.relu(out_lay2)
        with tf.variable_scope("out_lay"):
            out_lay = self.nn_layer(out_lay2, [n_hidden_2, n_classes], [n_classes], dropout_keep_prob)
        return out_lay
