import tensorflow as tf
import numpy as np


class Model(object):
    """
    Baseline for query-based paragraph classification (determine if the paragraph
    is relevant to the query).
    We implement a MLP on a vector a where a is the concatenation
    of the paragraph and the question embeddings.
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

            # Map word IDs to word embeddings
            self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
            self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)

            # Create CBOW embeddings of query and paragraph (i.e. average along axis that contain the embedded words)
            self.input_q_CBOW = tf.reduce_mean(self.input_q_emb,1, name="input_q_CBOW")
            self.input_p_CBOW = tf.reduce_mean(self.input_p_emb,1, name="input_p_CBOW")

            # Create embedding of paragraph using attention from query (embedded with CBOW)
            print(self.input_q_CBOW.get_shape())
            print(self.input_p_emb.get_shape())
            input_q_CBOW_expanded = tf.expand_dims(self.input_q_CBOW, 1)
            print(input_q_CBOW_expanded.get_shape())
            alphas = tf.multiply(input_q_CBOW_expanded, self.input_p_emb)
            print(alphas.get_shape())
            alphas = tf.reduce_sum(alphas, 2, name="alphas")
            print(alphas.get_shape())
            norm_alphas = tf.nn.softmax(logits=alphas, name="norm_alphas")
            print(norm_alphas.get_shape())
            norm_alphas_expanded = tf.expand_dims(norm_alphas, 2)
            print(norm_alphas_expanded.get_shape())
            #self.input_p_emb = tf.transpose(self.input_p_emb, perm=[0, 2, 1])
            #print(self.input_p_emb.get_shape())
            input_p_attention = tf.multiply(norm_alphas_expanded, self.input_p_emb)
            print(input_p_attention.get_shape())
            self.input_p_attention = tf.reduce_sum(input_p_attention, 1)
            print(self.input_p_attention.get_shape())

            # OPTIONAL : add dropout on the embeddings
            #self.input_q_CBOW_dropout = tf.nn.dropout(self.input_q_CBOW,self.dropout_keep_prob)
            #self.input_p_CBOW_dropout = tf.nn.dropout(self.input_p_CBOW,self.dropout_keep_prob)


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = embedding_size*4 # we are going to concat paragraph and question
        n_classes = num_classes

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # Compute pointwise square difference and pointwise multiplication between Q CBOW and P embedded with attention
            # Checks how relevant is the paragraph to the question
            dif_p_q = tf.subtract(self.input_q_CBOW, self.input_p_attention, name="dif_p_q")
            dif_p_q_point_mul = tf.multiply(dif_p_q, dif_p_q, name ="dif_p_q_point_mul")
            pq_point_mul = tf.multiply(self.input_q_CBOW, self.input_p_attention, name="pq_point_mul")
            # Compute pointwise square difference and pointwise multiplication between P CBOW and P embedded with attention
            # Checks how much the part of the paragraph relevant to the question is central with respect to the topic of the paragraph
            dif_p_p = tf.subtract(self.input_p_CBOW, self.input_p_attention, name="dif_p_p")
            dif_p_p_point_mul = tf.multiply(dif_p_p, dif_p_p, name ="dif_p_p_point_mul")
            pp_point_mul = tf.multiply(self.input_p_CBOW,self.input_q_CBOW,name="pp_point_mul")

            self.concatenated_input = tf.concat([dif_p_q, pq_point_mul, dif_p_p, pp_point_mul], 1,name="concatenated_input")
            self.scores = self.multilayer_perceptron(self.concatenated_input,
                            n_input, n_hidden_1, n_hidden_2, n_classes, self.dropout_keep_prob)
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
