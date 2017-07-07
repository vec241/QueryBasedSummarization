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
          '''

          self.W = self.W_emb
          self.W_2nd_row = tf.gather(self.W , 1)

          # Map word IDs to word embeddings
          self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
          self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)

          # Transform inputs of IDs into FUCK*** CBOW
          self.input_q_CBOW, self.mask_input_q_nonzero = self.embed_CBOW(self.input_q, self.input_q_emb)
          self.input_p_CBOW, self.mask_input_p_nonzero = self.embed_CBOW(self.input_p, self.input_p_emb)
          # That's it :) <3

          # OPTIONAL : add dropout on the embeddings
          #self.input_q_CBOW_dropout = tf.nn.dropout(self.input_q_CBOW,self.dropout_keep_prob)
          #self.input_p_CBOW_dropout = tf.nn.dropout(self.input_p_CBOW,self.dropout_keep_prob)



        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = embedding_size*2 # we are going to concat paragraph and question
        n_classes = num_classes

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
          dif = tf.subtract(self.input_p_CBOW, self.input_q_CBOW,name="dif")
          print(dif)
          dif_point_mul = tf.multiply(dif, dif, name ="dif_point_mul")
          print(dif_point_mul)
          pq_point_mul = tf.multiply(self.input_p_CBOW, self.input_q_CBOW,name="pq_point_mul")
          print(pq_point_mul)
          self.concatenated_input = tf.concat([dif_point_mul, pq_point_mul], 1,name="concatenated_input")
          print(self.concatenated_input)
          self.scores = self.multilayer_perceptron(self.concatenated_input,
              n_input, n_hidden_1, n_hidden_2, n_classes, self.dropout_keep_prob)
          print(self.scores)
          function_to_score = lambda x : x + (10.0**(-4))  # Where `f` instantiates myCustomOp.
          self.scores = tf.map_fn(function_to_score, self.scores)
          self.predictions = tf.argmax(self.scores, 1, name="predictions")
          print(self.predictions)
          self.y_true = tf.argmax(self.input_y, 1, name="y_true")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
          losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores, name = "losses")
          self.loss = tf.reduce_mean(losses,0,name = 'loss_sub') #+ l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
          correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1),name='correct_predictions')
          self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



        # Random stuff to print and delete later
        self.input_q_01 = tf.gather(self.input_q , 1)
        self.input_p_06 = tf.gather(self.input_p , 6)
        self.input_p_07 = tf.gather(self.input_p , 7)
        self.input_p_08 = tf.gather(self.input_p , 8)
        self.input_q_CBOW_new_01 = tf.gather(self.input_q_CBOW, 1)


    def embed_CBOW(self, input_indices, input_emb):
        """
        Create CBOW embedding of input of size batch_size * max_length
        Each input is a batch of sentences of type [ word_i word_j word_k 0 .... 0 ]
        """
        # First, count number of words in each sentence by counting number of non zeros
        zero_t = tf.constant(0, dtype=tf.int32)
        zero_f = tf.constant(0, dtype=tf.float32)
        mask_input_non_zero = tf.reduce_sum(tf.cast(tf.not_equal(input_indices,zero_t), tf.float32),1)
        ones = tf.ones_like(mask_input_non_zero, tf.float32)
        mask_input_non_zero = tf.where(tf.equal(mask_input_non_zero, zero_f), ones, mask_input_non_zero)
        mask_input_non_zero_expanded = tf.expand_dims(mask_input_non_zero,1)
        # Create CBOW embedding by summing words embedding and dividing by number of words
        input_sum = tf.reduce_sum(input_emb, 1)
        input_CBOW =  tf.div(input_sum, mask_input_non_zero_expanded)
        return input_CBOW, mask_input_non_zero

    def nn_layer(self, x, W_shape, bias_shape, dropout_keep_prob):
          W = tf.get_variable("weights", W_shape,
              initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable("biases", bias_shape,
              initializer=tf.contrib.layers.xavier_initializer())
          W = tf.nn.dropout(W, dropout_keep_prob)
          out_lay = tf.add(tf.matmul(x, W), b)
          #out_lay = tf.matmul(x, W)
          #W = tf.Print(W,[W]," nn layer W :")
          #b = tf.Print(b,[b]," nn layer b :")
          #out_lay = tf.Print(out_lay,[out_lay]," nn layer out_lay :")
          return out_lay, W


    def multilayer_perceptron(self, x, n_input, n_hidden_1, n_hidden_2, n_classes, dropout_keep_prob):
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







#### PREVIOUS CODE

"""

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

            dif = tf.subtract(self.input_p_CBOW,self.input_q_CBOW,name="dif")
            print(dif)
            dif_point_mul = tf.multiply(dif,dif,name ="dif_point_mul")
            print(dif_point_mul)
            pq_point_mul = tf.multiply(self.input_p_CBOW,self.input_q_CBOW,name="pq_point_mul")
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
            self.y_true = tf.argmax(self.input_y, 1, name="y_true")

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

"""
