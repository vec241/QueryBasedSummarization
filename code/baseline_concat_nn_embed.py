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
            """
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
            """
            self.W = self.W_emb
            self.W_2nd_row = tf.gather(self.W , 1)
            """
            #get non zero element
            tf.where(tf.equal(tf.reduce_sum(tf.abs(self.W_emb),1),0), self.train_W, self.W_emb)

            """
            zero_t = tf.constant(0, dtype=tf.int32)
            one_t = tf.constant(1, dtype=tf.int32)
            #mask_ones = tf.ones(self.input_q.get_shape(), tf.int32)
            #mask_zeros = tf.zeros(self.input_q.get_shape(), tf.int32)
            mask_input_q = tf.equal(self.input_q,zero_t)#,zero_t,one_t )
            mask_input_p = tf.equal(self.input_p,zero_t)
            #mask_input_q = tf.where(tf.equal(self.input_q,zero_t),zero_t,one_t )
            mask_input_q_non_zero = tf.reduce_sum(tf.cast(mask_input_q, tf.float32),1)
            mask_input_p_non_zero = tf.reduce_sum(tf.cast(mask_input_p, tf.float32),1)
            mask_input_q_non_zero = tf.expand_dims(mask_input_q_non_zero,1)
            #mask_input_q_non_zero.eval()
            mask_input_p_non_zero = tf.expand_dims(mask_input_p_non_zero,1)
            #mask_input_p_non_zero.eval()
            #mask_input_q_indices = tf.where(mask_input_q)
            print("mask_input_q :", mask_input_q)
            print("mask_input_q_non_zero :", mask_input_q_non_zero)
            # Print elements of q,p
            print("tf.gather(self.input_q, 0) ", tf.gather(self.input_q, 0))
            print("tf.gather(self.input_p, 0) ", tf.gather(self.input_p, 0))

            # Map word IDs to word embeddings
            self.input_q_emb = tf.nn.embedding_lookup(self.W, self.input_q)
            self.input_p_emb = tf.nn.embedding_lookup(self.W, self.input_p)
            #print("sess.run(self.input_q_emb) : ",sess.run(self.input_q_emb))
            # Transform matrix of word embeddings into CBOW (i.e. average along axis that contain the embedded words)
            self.input_q_CBOW = tf.reduce_mean(self.input_q_emb,1, name="input_q_CBOW")
            self.input_p_CBOW = tf.reduce_mean(self.input_p_emb,1, name="input_p_CBOW")

            #New CBOW
            self.input_q_sum = tf.reduce_sum(self.input_q_emb,1, name="input_q_sum")
            self.input_p_sum = tf.reduce_sum(self.input_p_emb,1, name="input_p_sum")
            print("input_q_sum :", self.input_q_sum)
            self.input_q_CBOW_new =  tf.div(self.input_q_sum ,mask_input_q_non_zero)
            self.input_p_CBOW_new =  tf.div(self.input_p_sum ,mask_input_p_non_zero)
            print("input_q_CBOW_new :", self.input_q_CBOW_new)
            print("input_p_CBOW_new :", self.input_p_CBOW_new)
            #self.input_p_CBOW_new




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
            self.concatenated_input = tf.concat([self.input_q_CBOW_new, self.input_p_CBOW_new], 1,name="concatenated_input")
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
        #v.name == "foo/v:0"
        self.W3 = W
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
