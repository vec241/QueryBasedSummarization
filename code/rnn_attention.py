import tensorflow as tf
import numpy as np


class Model(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, num_classes, vocab_size, embedding_size, max_length, vocab_proc, filter_sizes, num_filters,
      l2_reg_lambda=0.0, use_emb=True):

        # Placeholders for input, embedding matrix for unknown words and dropout
        self.input_q = tf.placeholder(tf.int32, [None, max_length], name="input_q")
        self.input_p = tf.placeholder(tf.int32, [None, max_length], name="input_p")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.W_emb = tf.placeholder(tf.float32,[vocab_size, embedding_size], name="emb_pretrained")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.seq_len = tf.placeholder(tf.int32, [None])

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

            # Map word IDs to word embeddings
            self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
            self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)


        # Process p and q through rnn / attention layer
        with tf.name_scope("rnn_question"):
            with tf.variable_scope("rnn_q"):
                # For question, we do BiLSTM + only take last hidden states
                outputs_q, outputs_states_q = self.BiRNN(embedding_size, self.input_q_emb, max_length, self.seq_len)
                print('outputs_q', outputs_q)
                print('outputs_states_q', outputs_states_q)
                # Concat last hiden spaces of both LSTMs (forward and bacward LSTMs)
                self.h_q = tf.concat([outputs_states_q[0][1], outputs_states_q[1][1]], 1)
                print('h_q', self.h_q)

        with tf.name_scope("rnn_paragraph"):
            with tf.variable_scope("rnn_p"):
                # For paragraph, we do BiLSTM and keep all hidden states associated to each input word,
                # to do attention over using hidden state from question h_q
                outputs_p, outputs_states_p = self.BiRNN(embedding_size, self.input_p_emb, max_length, self.seq_len)
                print('outputs_p', outputs_p)
                print('outputs_states_p', outputs_states_p)
                outputs_p = tf.concat(outputs_p, 2)
                print('outputs_p', outputs_p)
                self.h_p = self.attention(self.h_q, outputs_p, max_length, filter_sizes, num_filters)
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



    def BiRNN(self, embedding_size, input_emb, max_length, seq_len):
        n_cell_dim = 64 #embedding_size
        with tf.name_scope('lstm_q'):
            print('input_emb', input_emb)
            lstm_fw_cell_q = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell_q = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell_q,
                                                                     cell_bw=lstm_bw_cell_q,
                                                                     inputs=input_emb,
                                                                     dtype=tf.float32,
                                                                     time_major=False,
                                                                     sequence_length=seq_len)
        return outputs, output_states


    def attention(self, query_vector, paragraph, max_length, filter_sizes, num_filters):
        # Create embedding of paragraph using attention from query (embedded with CBOW)
        print('query_vector.get_shape()', query_vector.get_shape())
        print('paragraph.get_shape()', paragraph.get_shape())
        query_vector_expanded = tf.expand_dims(query_vector, 1)
        print('query_vector_expanded.get_shape()', query_vector_expanded.get_shape())
        alphas = tf.multiply(query_vector_expanded, paragraph)
        print('alphas.get_shape()', alphas.get_shape())
        alphas = tf.reduce_sum(alphas, 2, name="alphas")
        print('alphas.get_shape()', alphas.get_shape())
        norm_alphas = tf.nn.softmax(logits=alphas, name="norm_alphas")
        print('norm_alphas.get_shape()', norm_alphas.get_shape())
        norm_alphas_expanded = tf.expand_dims(norm_alphas, 2)
        print('norm_alphas_expanded.get_shape()', norm_alphas_expanded.get_shape())
        #self.input_p_emb = tf.transpose(self.input_p_emb, perm=[0, 2, 1])
        #print(self.input_p_emb.get_shape())
        h_attention = tf.multiply(norm_alphas_expanded, paragraph)
        print('h_attention.get_shape()', h_attention.get_shape())
        h_attention = tf.reduce_sum(h_attention, 1)
        print('h_attention.get_shape()', h_attention.get_shape())
        return h_attention


    def nn_layer(self, x, W_shape, bias_shape, dropout_keep_prob):
        W = tf.get_variable("weights", W_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", bias_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        W = tf.nn.dropout(W, dropout_keep_prob)
        out_lay = tf.add(tf.matmul(x, W), b)
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
