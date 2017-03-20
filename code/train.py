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
import data_helpers
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Which model to use
tf.flags.DEFINE_string("model", "baseline_sub_mult_nn", "Specify which model to use")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("labels", "../../data/short_fold0_600K_labels.csv", "labels")
tf.flags.DEFINE_string("query_CBOW", "../../data/short_fold0_600K_query_CBOW.csv", "query_CBOW")
tf.flags.DEFINE_string("paragraph_CBOW", "../../data/short_fold0_600K_paragraph_CBOW.csv", "paragraph_CBOW")
tf.flags.DEFINE_string("query_CNN", "../../data/short_fold0_600K_query_CNN.csv", "query_CNN")
tf.flags.DEFINE_string("paragraph_CNN", "../../data/short_fold0_600K_paragraph_CNN.csv", "paragraph_CNN")
tf.flags.DEFINE_string("query_RNN", "../../data/short_fold0_600K_query_RNN.csv", "query_RNN")
tf.flags.DEFINE_string("paragraph_RNN", "../../data/short_fold0_600K_paragraph_RNN.csv", "paragraph_RNN")
tf.flags.DEFINE_string("query_text", "../../data/short_fold0_600K_query_text.csv", "query_text")
tf.flags.DEFINE_string("paragraph_text", "../../data/short_fold0_600K_paragraph_text.csv", "paragraph_text")
tf.flags.DEFINE_string("embedding_method", "CBOW", "embedding_method")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Import model
print("importing model %s ..." %(FLAGS.model))
if FLAGS.model == "baseline_bilinear":
    from baseline_bilinear import Model
elif FLAGS.model == "baseline_sub_mult_nn":
    from baseline_sub_mult_nn import Model
elif FLAGS.model == "baseline_concat_nn":
    from baseline_concat_nn import Model
else:
    print("wrong model defined")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
if FLAGS.model in ["baseline_bilinear", "baseline_sub_mult_nn", "baseline_concat_nn"]:
    q, p, y = data_helpers.load_data_and_labels(FLAGS)
    #q, p, y = data_helpers.load_data_and_labels(FLAGS.labels, FLAGS.query_CBOW, FLAGS.paragraph_CBOW, FLAGS.embedding_method)
else:
    q, p, y = data_helpers.load_data_and_labels(FLAGS.labels, FLAGS.query_text, FLAGS.paragraph_text, FLAGS.embedding_method)

# Randomly shuffle data
c = list(zip(q, p, y))
np.random.shuffle(c)
q_shuffled, p_shuffled, y_shuffled = zip(*c)
q_shuffled, p_shuffled, y_shuffled = np.array(q_shuffled), np.array(p_shuffled), np.array(y_shuffled)

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
q_train, q_dev = q_shuffled[:dev_sample_index], q_shuffled[dev_sample_index:]
p_train, p_dev = p_shuffled[:dev_sample_index], p_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
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
            #sequence_length=q_train.shape[1],
            num_classes=y_train.shape[1],
            #vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=q_train.shape[1],
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join("../../runs", timestamp))  #os.path.curdir,"runs", timestamp
        print("Writing to {}\n".format(out_dir))

        print("Defined Model\n")
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss, tf.trainable_variables())
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        #out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        #print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
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
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(q_batch, p_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              model.input_q: q_batch,
              model.input_p: p_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(q_batch, p_batch, y_dev, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              model.input_q: q_batch,
              model.input_p: p_batch,
              model.input_y: y_dev,
              model.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(q_train, p_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            q_batch, p_batch, y_batch = zip(*batch)
            train_step(q_batch, p_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(q_dev, p_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
