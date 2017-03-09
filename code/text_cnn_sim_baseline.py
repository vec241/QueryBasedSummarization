import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters,
      l2_reg_lambda=0.0, vocab_size=None):

        # Placeholders for input, output and dropout
        self.input_q = tf.placeholder(tf.float32, [None, embedding_size], name="input_q")
        self.input_p = tf.placeholder(tf.float32, [None, embedding_size], name="input_p")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        '''
        # Preparing inputs in right format
        with tf.name_scope("expanding"):
            self.input_q_expanded = tf.expand_dims(self.input_q, 2, name='input_q_expanded')
            self.input_p_expanded = tf.expand_dims(self.input_p, 2, name='input_p_expanded')
        '''
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            '''W = tf.Variable(
                tf.random_uniform([embedding_size, embedding_size, num_classes], -1.0, 1.0),
                name="W")'''
            W = tf.get_variable(
                "W",
                shape=[embedding_size*2, num_classes],  #embedding_size,
                initializer=tf.contrib.layers.xavier_initializer())
            #W_expanded = tf.expand_dims(W, 0, name='W_expanded')
            #l2_loss += tf.nn.l2_loss(W)
            #Wpp = tf.multiply(W_expanded,self.input_p_expanded, name='Wpp')
            #print(Wpp)
            #Wp = tf.reduce_sum(tf.multiply(W_expanded,self.input_p_expanded),1, name="Wp")



            #new code
            dif = tf.subtract(self.input_p,self.input_q,name="dif")
            print(dif)
            dif_point_mul = tf.multiply(dif,dif,name ="dif_point_mul")
            print(dif_point_mul)
            pq_point_mul = tf.multiply(self.input_p,self.input_q,name="pq_point_mul")
            print(pq_point_mul)
            self.concated_input = tf.concat([dif_point_mul,pq_point_mul], 1,name="concated_input")
            print(self.concated_input)
            W = tf.nn.dropout(W,0.6)
            self.scores = tf.matmul(self.concated_input,W, name='scores')  #tf.reshape(W,[embedding_size,num_classes])
            print(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print(self.predictions)
            #Wp = tf.transpose(tf.reshape(Wp_flat,[-1, embedding_size, num_classes]), perm=[1, 0, 2], name='Wp')
            #print(Wp)
            #qWp = tf.matmul(tf.reshape(self.input_q, [-1, 1]), tf.reshape(Wp, [-1, num_classes]), name='qWp')
            #print(qWp)
            #self.scores = tf.reduce_sum(qWp_reshaped, 1, name="scores")
            #print(self.scores)

            #old code
            #Wp_flat = tf.matmul(self.input_p, tf.reshape(W,[embedding_size,embedding_size*num_classes]), name='Wp_flat')
            #print(Wp_flat)
            #Wp = tf.transpose(tf.reshape(Wp_flat,[-1, embedding_size, num_classes]), perm=[1, 0, 2], name='Wp')
            #print(Wp)
            #qWp = tf.matmul(tf.reshape(self.input_q, [-1, 1]), tf.reshape(Wp, [-1, num_classes]), name='qWp')
            #print(qWp)
            #self.scores = tf.reduce_sum(qWp_reshaped, 1, name="scores")
            #print(self.scores)
            #self.scores = tf.reduce_sum(tf.multiply(self.input_q_expanded, Wp), 0, name="scores")
            #self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #self.scores = tf.matmul(self.input_q,tf.reshape(Wp,[embedding_size,num_classes]), name="scores")
            #self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #print(self.predictions)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            print(self.scores)
            print(self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores, name = "losses")
            print('1')
            print(losses)
            self.loss = tf.reduce_mean(losses,0,name = 'loss_sub') #+ l2_reg_lambda * l2_loss
            print('2')
            print(self.loss)
            print('3')

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1),name='correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        '''
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.scores, logits=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        '''
