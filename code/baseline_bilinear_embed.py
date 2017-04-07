import tensorflow as tf
import numpy as np

# TO DO : OPTIONAL --> ADD MLP ON TOP OF BILINEAR PRODUCT. TO DISCUSS WITH K. CHO

class Model(object):
    """
    Baseline for query-based paragraph classification.
    Implements a bilinear product between paragraph and query to determine if
    the paragraph is relevant to the query.
    """
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

            # Create the matrix of embeddings of all words in vocab
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

            # Map word IDs to word embeddings
            self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
            self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)

            # Transform matrix of word embeddings into CBOW (i.e. average along axis that contain the embedded words)
            self.input_q_CBOW = tf.reduce_mean(self.input_q_emb,1, name="input_q_CBOW")
            self.input_p_CBOW = tf.reduce_mean(self.input_p_emb,1, name="input_p_CBOW")

            # OPTIONAL : add dropout on the embeddings
            #self.input_q_CBOW_dropout = tf.nn.dropout(self.input_q_CBOW,self.dropout_keep_prob)
            #self.input_p_CBOW_dropout = tf.nn.dropout(self.input_p_CBOW,self.dropout_keep_prob)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[embedding_size , embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            W = tf.nn.dropout(W, self.dropout_keep_prob)
            qW = tf.matmul(self.input_q_CBOW, tf.reshape(W,[embedding_size,embedding_size*num_classes]), name='qW')
            qW_reshape = tf.reshape(qW,[-1,embedding_size, num_classes], name='qW_reshape')
            qW_reshape2 = tf.reshape(qW_reshape,[-1,embedding_size], name='qW_reshape2')
            double_p = tf.concat([self.input_p_CBOW, self.input_p_CBOW], 0, name='double_p')
            double_qWp = tf.multiply(qW_reshape2, double_p, name='double_qWp')
            double_qWp_reshape = tf.reshape(double_qWp,[-1,embedding_size, num_classes], name='double_qWp_reshape')
            self.scores = tf.reduce_sum(double_qWp_reshape, 1, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.y_true = tf.argmax(self.input_y, 1, name="y_true")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses,0) #+ l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
