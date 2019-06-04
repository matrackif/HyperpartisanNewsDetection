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





def train_using_keras(train_x, test_x, train_y, test_y):
    print(train_x.shape[1])
    model = models.baseline_model.create_baseline2(input_size=train_x.shape[1])
    model.summary()
    start_time = time.time()
    history = model.fit(train_x, train_y, epochs=30, validation_data=(test_x, test_y),
                        verbose=2)
    print('Keras model finished training in %s seconds' % (time.time() - start_time))
    # plot history

    plt.figure(0)
    plt.plot(history.history['loss'], label='Training loss (error)')
    plt.plot(history.history['val_loss'], label='Test loss (error)')
    plt.title('Training/test loss of article classifier')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def fit_using_logistic_regression(train_x, test_x, train_y, test_y):
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(train_x, train_y)
    predictions_train = clf.predict(train_x)
    predictions_test = clf.predict(test_x)
    print("LogisticRegression classification report:")
    print(classification_report(train_y, predictions_train))
    # plot_confusion_matrix(train_y, predictions_train)
    plot_confusion_matrix(test_y, predictions_test)
    plt.show()


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





if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', action='store_true', default=False,
                            help='Train Keras baseline model, if false then use logistic regression')
    arg_parser.add_argument('-a', '--articles', nargs='?', type=str, default='data/articles.xml',
                            const='data/articles.xml', help='Path to XML file of articles')
    arg_parser.add_argument('-c', '--classified-articles', nargs='?', type=str, default='data/classifiedArticles.xml',
                            const='data/classifiedArticles.xml', help='Path to XML file of classified articles')
    args = vars(arg_parser.parse_args())
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



