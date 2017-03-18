import tensorflow as tf
import numpy as np

class Model(object):
    """
    Baseline for query-based paragraph classification.
    Implements a bilinear product between paragraph and query to determine if
    the paragraph is relevant to the query.
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

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size , embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            W = tf.nn.dropout(W, self.dropout_keep_prob)
            qW = tf.matmul(self.input_p, tf.reshape(W,[embedding_size,embedding_size*num_classes]), name='qW')
            qW_reshape = tf.reshape(qW,[-1,embedding_size, num_classes], name='qW_reshape')
            qW_reshape2 = tf.reshape(qW_reshape,[-1,embedding_size], name='qW_reshape2')
            double_p = tf.concat([self.input_p, self.input_p], 0, name='double_p')
            double_qWp = tf.multiply(qW_reshape2, double_p, name='double_qWp')
            double_qWp_reshape = tf.reshape(double_qWp,[-1,embedding_size, num_classes], name='double_qWp_reshape')
            self.scores = tf.reduce_sum(double_qWp_reshape, 1, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses,0) #+ l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
