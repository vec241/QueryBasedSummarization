# QueryBasedSummarization
Query based summarization for a NYU research project

This repository gathers the code of the research study described in the .pdf report.

You must first clone this repository and then organize a root repository as follow:

- Root repo
  - This Github repo
  - Data
  - glove
  - runs

The Data repository should contain the data. The data was originally extracted from the TREC CAR dataset (http://trec-car.cs.unh.edu/). You can ask us to directly obtain the .csv files we use in our code (vec241@nyu.edu, up276@nyu.edu).

The glove repository gathers the GloVe vectors. Go to https://nlp.stanford.edu/projects/glove/ and download glove6B.zip. Put glove.6B.50d in the glove repo.

The runs repository is just an empty repository where the model will save the runs.

To run the code, just do python3 embed_and_train.py (for all the models excepts rnn based models) or embed_and_train_rnn_attention.py (for the rnn based models).

In embed_and_train.py or embed_and_train_rnn_attention.py, many parameters you can tuned (via the FLAGS). Most importantly, chose the model and the dataset you want to use by modifying the two following flags :
- tf.flags.DEFINE_string("model", "rnn_attention", "Specify which model to use")
- tf.flags.DEFINE_string("dataset_size", "short_balanced", "short_balanced, medium_balanced, or full_balanced")

To initiate a job on GPU, please use "initiate_job_to_run_required_script.sh" and change the parameters as per your requirement. Such as you can specify memory limit, name of GPU instance available, max running time etc. Once you are done with that, please specify correct path and script name which you would like to initiate. Please note that, #SBATCH is the local command for NYU Prince Server, you might would have to change these command with your server specific command.
