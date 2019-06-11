from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
import codecs
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import models.baseline_model
import time
from matplotlib import pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
import doc2vec.preprocessor as pre
import tensorflow as tf
import tensorflow_hub as hub

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
# This is a path to an uncased (all lowercase) version of BERT


def get_data_from_article(article):
    return {'id': article['@id'], 'title': article['@title']}


def get_labels(article):
    return {'id': article['@id'], 'label': article['@hyperpartisan'], 'url': article['@url']}

def get_labels_only(article):
    return {'id': article['@id'], 'label': article['@hyperpartisan']}


def extract_training_data(articles_file: str, classified_articles_file: str):
    articles_raw = pre.read_file(articles_file)
    labels_raw = pre.read_file(classified_articles_file)

    title = [get_data_from_article(article) for article in articles_raw['articles']['article']]
    labels = [get_labels(article) for article in labels_raw['articles']['article']]
    df_titles = pd.DataFrame(title)
    df_labels = pd.DataFrame(labels)
    df = df_titles.merge(df_labels, on='id')
    df.label = df.label.astype('int')

    df['https'] = df.url.map(lambda x: urlsplit(x).scheme == 'https').astype('int')
    df['domain'] = df.url.map(lambda x: urlsplit(x).netloc)
    df['country'] = df.domain.map(lambda x: x.split('.')[-1])
    print("Correlation between https and Hyperpartisan", df.corr())
    df_train, df_test, _, _ = train_test_split(df, df.label, stratify=df.label, test_size=0.2)
    vectorizer = TfidfVectorizer(stop_words='english', )
    df_train.title = df_train.title.apply(str)
    df_test.title = df_test.title.apply(str)
    df.title = df.title.apply(str)
    vectorizer.fit(df_train.title)
    train_x = np.concatenate((
        vectorizer.transform(df_train.title).toarray(),
        df_train.https.values.reshape(-1, 1),
    ), axis=1)
    # train_x = vectorizer.transform(df_train.title).toarray()
    test_x = np.concatenate((
        vectorizer.transform(df_test.title).toarray(),
        df_test.https.values.reshape(-1, 1),
    ), axis=1)
    # test_x = vectorizer.transform(df_test.title).toarray()

    train_y = df_train.label.values
    test_y = df_test.label.values
    return train_x, test_x, train_y, test_y


def extract_data_bert(articles_file: str, classified_articles_file: str):
    articles_raw = pre.read_file(articles_file)
    labels_raw = pre.read_file(classified_articles_file)
    title = [get_data_from_article(article) for article in articles_raw['articles']['article']]
    labels = [get_labels_only(article) for article in labels_raw['articles']['article']]
    df_titles = pd.DataFrame(title)
    df_labels = pd.DataFrame(labels)
    df = df_titles.merge(df_labels, on='id')
    df = df.drop(columns=['id'])
    tmp_true = df.label[27]
    df.label = (df.label == tmp_true).astype(int)
    df_train, df_test, _, _ = train_test_split(df, df.label, stratify=df.label, test_size=0.2)
    df_train.title = df_train.title.apply(str)
    df_test.title = df_test.title.apply(str)
    return df_train, df_test




def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', action='store_true', default=False,
                            help='Train Keras baseline model, if false then use logistic regression')
    arg_parser.add_argument('-a', '--articles', nargs='?', type=str, default='data/articles.xml',
                            const='data/articles.xml', help='Path to XML file of articles')
    arg_parser.add_argument('-c', '--classified-articles', nargs='?', type=str, default='data/classifiedArticles.xml',
                            const='data/classifiedArticles.xml', help='Path to XML file of classified articles')
    args = vars(arg_parser.parse_args())
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    # pre.analyze_articles(args['articles'])
    train, test = extract_data_bert(args['articles'], args['classified_articles'])
    DATA_COLUMN = 'title'
    LABEL_COLUMN = 'label'
    label_list = [0, 1]
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                                 text_a=x[DATA_COLUMN],
                                                                                 text_b=None,
                                                                                 label=x[LABEL_COLUMN]), axis=1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a=x[DATA_COLUMN],
                                                                   text_b=None,
                                                                   label=x[LABEL_COLUMN]), axis=1)

    tokenizer = create_tokenizer_from_hub_module()
    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128
    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                      tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH,
                                                                     tokenizer)
    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    OUTPUT_DIR="./OUTPUT"
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    model_fn = model_fn_builder(
        num_labels=len(label_list),
        learning_rate=LEARNING_RATE,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": BATCH_SIZE})
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)
    estimator.evaluate(input_fn=test_input_fn, steps=None)



