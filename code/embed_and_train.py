#! /usr/bin/env python

# ==================================================
# To VISUALIZE graph in TENSORBOARD :
# In terminal :
# tensorboard --logdir="./graphs" --port 6006
# In browser
# http://localhost:6006/
# ==================================================

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers_embed
from tensorflow.contrib import learn
from sklearn.metrics import precision_score, recall_score

# Parameters
# ==================================================

# Which model, which embedding method and which data size to use
tf.flags.DEFINE_string("model", "rnn_attention", "Specify which model to use") #rnn_attention, cnn_attention_concat_MLP, cnn_att_sub_mult, cnn_att_comp_agr baseline_concat_nn_embed , baseline_sub_mult_nn_embed, cnn_attention_concat_MLP
tf.flags.DEFINE_string("embedding_method", "CBOW", "embedding_method")
tf.flags.DEFINE_string("dataset_size", "short_balanced", "short_balanced, medium_balanced, or full_balanced")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("emb_path", "../../glove/glove.6B.50d.txt","Path to word embeddings")
tf.flags.DEFINE_string("short_balanced_labels", "../../data/clean_balanced_short_fold0_600K_labels.csv", "labels")
tf.flags.DEFINE_string("short_balanced_query_text", "../../data/clean_balanced_short_fold0_600K_query_text.csv", "query_text")
tf.flags.DEFINE_string("short_balanced_paragraph_text", "../../data/clean_balanced_short_fold0_600K_paragraph_text.csv", "paragraph_text")
tf.flags.DEFINE_string("medium_balanced_labels", "../../data/clean_balanced_medium_fold0_600K_labels.csv", "labels")
tf.flags.DEFINE_string("medium_balanced_query_text", "../../data/clean_balanced_medium_fold0_600K_query_text.csv", "query_text")
tf.flags.DEFINE_string("medium_balanced_paragraph_text", "../../data/clean_balanced_medium_fold0_600K_paragraph_text.csv", "paragraph_text")
tf.flags.DEFINE_string("full_balanced_labels", "../../data/300K_clean_balanced_full_fold0_600K_labels.csv", "labels")
tf.flags.DEFINE_string("full_balanced_query_text", "../../data/300K_clean_balanced_full_fold0_600K_query_text.csv", "query_text")
tf.flags.DEFINE_string("full_balanced_paragraph_text", "../../data/300K_clean_balanced_full_fold0_600K_paragraph_text.csv", "paragraph_text")
tf.flags.DEFINE_string("data_cleaning_flag", True, "data_cleaning_flag")

# Model Hyperparameters
#tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 150, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("vocab_freq", 3, "Min word frequency to appear in vocab. Default : 5")
tf.flags.DEFINE_integer("max_doc_length", 50, "Max document length. Truncates documents longer than x words. Default : 1000")
tf.flags.DEFINE_integer("max_question_length", 5, "Max document length. Truncates documents longer than x words. Default : 1000")
tf.flags.DEFINE_boolean("separate_question_length", False, "separate_question_length")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

timestamp_val = str(time.time())
#create output files
dev_opt_filename = "dev_"+FLAGS.model + timestamp_val + ".csv"
train_opt_filename =  "train_"+FLAGS.model + timestamp_val + ".csv"
#dev_output = open("../../runs/"+dev_opt_filename,"a")
#train_output = open("../../runs/"+train_opt_filename,"a")
#dev_output.write("step,dev_acc,dev_precision,dev_recall\n")
#train_output.write("step,train_acc,train_precision,train_recall\n")
dev_opt_file = "../../runs/"+dev_opt_filename
train_opt_file = "../../runs/"+train_opt_filename

with open(dev_opt_file,"a") as dev_opt:
        dev_opt.write("step,dev_acc,dev_precision,dev_recall\n")


with open(train_opt_file,"a") as train_opt:
        train_opt.write("step,train_acc,train_precision,train_recall\n")


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Import model
print("importing model %s ..." %(FLAGS.model))
if FLAGS.model == "baseline_bilinear_embed":
    from baseline_bilinear_embed import Model
elif FLAGS.model == "baseline_sub_mult_nn_embed":
    from baseline_sub_mult_nn_embed import Model
elif FLAGS.model == "baseline_concat_nn_embed":
    from baseline_concat_nn_embed import Model
elif FLAGS.model == "simple_attention_concat_nn_embed":
    from simple_attention_concat_nn_embed import Model
elif FLAGS.model == "cnn_att_sub_mult":
    from cnn_attention_sub_mult_nn import Model
elif FLAGS.model == "cnn_att_comp_agr":
    from cnn_att_comp_agr import Model
elif FLAGS.model == "cnn_attention_concat_MLP":
    from cnn_attention_concat_MLP import Model
elif FLAGS.model == "rnn_attention":
    from rnn_attention import Model
else:
    print("wrong model defined")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
q_text, p_text, y = data_helpers_embed.load_data_and_labels(FLAGS)

#print(q_text[0])
#print(p_text[0])

# Build vocabulary
print("Finding maximum length in train data...")
max_document_length = max([len(p.split(" ")) for p in p_text])
print("Maximum document length is %d words" %max_document_length)
print("Building vocabulary...")
vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_doc_length, min_frequency=FLAGS.vocab_freq)
print("Processing text data...")
vocab_processor.fit(np.append(p_text,q_text))
q = np.array(list(vocab_processor.transform(q_text)))
p = np.array(list(vocab_processor.transform(p_text)))

#cutting the question at separate length
if FLAGS.separate_question_length:
    print("Cutting queries at separate length...")
    #q = [data_helpers_embed.cut(x) for x in q]
    q = [x[:FLAGS.max_question_length] for x in q]
    print("Queries cut operation done !")

#print("q[0] : ",q[0])
#print("p[0] : ",p[0])
vocab_dict = vocab_processor.vocabulary_._mapping
#print("vocab_dict",vocab_dict)
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
#print("sorted_vocab[0] : ",sorted_vocab[0])
vocabulary = list(list(zip(*sorted_vocab))[0])
#print("vocabulary[:3] : ",vocabulary[:3])

# Load embeddings
embeddings = data_helpers_embed.load_embeddings(FLAGS.emb_path, vocab_processor)
#print("embeddings[0] before : ",embeddings[0])
#print("embeddings 143 'state' : ",embeddings[143])
embeddings[0]=  np.zeros((1,embeddings.shape[1]))#mat[mapped_words.index(i)]

#print("embeddings.shape : ",embeddings.shape)

#print("embeddings[0] after : ",embeddings[0])
#print("embeddings[1] : ",embeddings[1])
# Randomly shuffle data
print("Randomly shuffling the data.../n")
c = list(zip(q, p, y))
np.random.shuffle(c)
q_shuffled, p_shuffled, y_shuffled = zip(*c)
q_shuffled, p_shuffled, y_shuffled = np.array(q_shuffled), np.array(p_shuffled), np.array(y_shuffled)
print("data shuffled !/n")

# Split train/test set
# TODO: This is very crude, should use cross-validation
print("Splitting into train and dev \n")

if FLAGS.dataset_size == "full_balanced":
    dev_sample_index = -5000
else:
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
print("Dev Samples : ",dev_sample_index )
q_train, q_dev = q_shuffled[:dev_sample_index], q_shuffled[dev_sample_index:]
p_train, p_dev = p_shuffled[:dev_sample_index], p_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print ("q_train.shape :",np.shape(q_train))
print ("p_train.shape :",np.shape(p_train))
print ("y_train.shape :",np.shape(y_train))
print ("q_dev.shape :",np.shape(q_dev))
print ("p_dev.shape :",np.shape(p_dev))
print ("y_dev.shape :",np.shape(y_dev))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

print("Entering into graph\n")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("Entering into session\n")
    with sess.as_default():

        model = Model(
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size = embeddings.shape[1],
            max_length = FLAGS.max_doc_length,
            vocab_proc = vocab_processor,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)


        print("Defined Model\n")
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss) # ,tf.trainable_variables()
        def ClipIfNotNone(grad):
            if grad is None:
                print("NAN----------------------------------------------------")
                return grad
            return tf.clip_by_value(grad, -5, 5)
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]

        print("grads_and_vars")
        #tf.Print([gv[0] for gv in grads_and_vars],[gv[0] for gv in grads_and_vars],"[gv[0] for gv in grads_and_vars]")

        # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
        # need to the 'gradient' part, for example cap them, etc.
        #capped_grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]

        # Ask the optimizer to apply the capped gradients.
        train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)"""

        #train_op = optimizer.minimize(model.loss, global_step=global_step)

        """
        if gradient_clipping:
            `gradients = optimizer.compute_gradients(loss)

            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1, 1)
            clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
            opt = optimizer.apply_gradients(clipped_gradients, global_step=global_step)
        else:
            opt = optimizer.minimize(loss, global_step=global_step)`

        """

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        #grad_summaries_merged = tf.constant(1)                     #updated


        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "../../runs", timestamp_val))
        print("Writing to {}\n".format(out_dir))


        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        #train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        print ("global variable initialized")

        def train_step(q_batch, p_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              model.input_q: q_batch,
              model.input_p: p_batch,
              model.input_y: y_batch,
              model.W_emb: embeddings,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            """
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, model.loss, model.accuracy],
                feed_dict)
            """
            #_,step, loss, accuracy, y_true, y_pred, input_q_CBOW_new_01, w1, q01, p_nonzero, outlay1, scores, pred , loss, concatenated_input= sess.run(
            #    [train_op,global_step,  model.loss, model.accuracy, model.y_true, model.predictions, model.input_q_CBOW_new_01,model.W1,model.input_q_01, model.mask_input_p_nonzero, model.outlay1,model.scores,model.predictions, model.loss, model.concatenated_input],
            #    feed_dict)  #,model.W2,model.W3 w2,w3,, outlay1, outlay2  , model.outlay1, model.outlay2
            _,step, loss, accuracy, y_true, y_pred, scores= sess.run(
                [train_op, global_step, model.loss, model.accuracy, model.y_true, model.predictions, model.scores],
                feed_dict)
            #print("sess.run(self.input_q_emb) : ",sess.run(self.input_q_emb))
            time_str = datetime.datetime.now().isoformat()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            opt_line = str(step) + "," + str(accuracy) + "," + str(precision) +","+str(recall)+","+ str(loss)+"\n"
            with open(train_opt_file,"a") as train_opt:
                    train_opt.write(opt_line)
	    #train_output.write(opt_line)
            #print("p_nonzero : ", p_nonzero)
            '''
            #print("W_2nd_row : ", W_2nd_row)
            print("====================================TRAIN STEP ========================================")
            print("W1_learned : ", w1)
            #print("W2_learned : ", w2)
            #print("W3_learned : ", w3)
            print("q01 : ", q01)
            print("q01_emb : ", q01_emb)
            print("q01_sum : ", q_sum)
            print("q01_nonzero : ", q_nonzero)
            print("input_q_CBOW_new_01 : ",input_q_CBOW_new_01)
            print("outlay1 before: ",outlay1_before)
            print("outlay1 after: ",outlay1)
            #print("outlay2 : ",outlay2)
            print("train scores : ",scores)
            print("train  Predictions : ", pred)
            print("train  loss : ", loss)
            '''
            #for i in range(len(concatenated_input)):
            #    print('concatenated_input', concatenated_input[i])
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step) #train_summary_op, # summaries,

        def dev_step(q_batch, p_batch, y_dev, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              model.input_q: q_batch,
              model.input_p: p_batch,
              model.input_y: y_dev,
              model.W_emb: embeddings,
              model.dropout_keep_prob: 1.0
            }
            #step, summaries, loss, accuracy, y_true, y_pred, input_q_CBOW_new_01,w1, q01,p06,p07,p08, p_nonzero, outlay1, scores, pred , loss, concatenated_input, input_p, outlay= sess.run(
            #    [global_step, dev_summary_op, model.loss, model.accuracy, model.y_true, model.predictions, model.input_q_CBOW_new_01,model.W1,model.input_q_01,model.input_p_06,model.input_p_07,model.input_p_08, model.mask_input_p_nonzero, model.outlay1,model.scores,model.predictions, model.loss, model.concatenated_input, model.input_p, model.outlay],
            #    feed_dict)  #,model.W2,model.W3 w2,w3, , outlay1, outlay2 , model.outlay1, model.outlay2
            step, summaries, loss, accuracy, y_true, y_pred, scores= sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy, model.y_true, model.predictions, model.scores],
                feed_dict)
            #print("sess.run(self.input_q_emb) : ",sess.run(self.input_q_emb))
            time_str = datetime.datetime.now().isoformat()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            opt_line = str(step) + "," + str(accuracy) + "," + str(precision) +","+str(recall)+","+ str(loss)+"\n"
            with open(dev_opt_file,"a") as dev_opt:
                    dev_opt.write(opt_line)
	    #dev_output.write(opt_line)
            '''
            #print("W_2nd_row : ", W_2nd_row)
            print("====================================DEV STEP ========================================")
            print("W1_learned : ", w1)
            #print("W2_learned : ", w2)
            #print("W3_learned : ", w3)
            print("q01 : ", q01)
            print("q01_emb : ", q01_emb)
            print("q01_sum : ", q_sum)
            print("q01_nonzero : ", q_nonzero)
            print("dev input_q_CBOW_new_01 : ",input_q_CBOW_new_01)
            print("outlay1 before: ",outlay1_before)
            print("outlay1 : ",outlay1)
            #print("outlay2 : ",outlay2)
            print("dev loss : ", loss)
            '''
            #print("p_nonzero : ", p_nonzero)

            #print("input p :", input_p)
            #print("input p06 :", p06)
            #print("input p07 :", p07)
            #print("input p08 :", p08)
            #for i in range(len(input_p)):
            #    print("input p", i ," : ", input_p[i] )
            #for i in range(len(concatenated_input)):
                #print("input p06 :", p06)
                #print("input p07 :", p07)
                #print("input p :", input_p)

            #    print('concatenated_input', concatenated_input[i])
            #print("q01 : ", q01)
            #print("dev input_q_CBOW_new_01 : ",input_q_CBOW_new_01)
            #print("dev outlay1 : ", outlay1)
            #print("50 first dev scores : ", scores[:50])
            #print("dev Predictions : ")
            #for i in range(len(scores)):
            #    print(scores[i])

            print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}".format(time_str, step, loss, accuracy, precision, recall))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers_embed.batch_iter(
            list(zip(q_train, p_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print("data batches prepared for the model : ")


        # Training loop. For each batch...
        for batch in batches:
            q_batch, p_batch, y_batch = zip(*batch)
            #print ("q_batch.shape[1] :",q_batch.shape[1])
            #q_batch, p_batch = data_helpers_embed.embed_batch(q_batch, p_batch)
            #print ("q_batch.shape :",np.shape(q_batch))
            #print ("p_batch.shape :",np.shape(p_batch))
            #print ("y_batch.shape :",np.shape(y_batch))
            train_step(q_batch, p_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(q_dev, p_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        #dev_output.close()
        #train_output.close()
