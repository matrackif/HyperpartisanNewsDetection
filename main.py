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


def get_data_from_article(article):
    return {'id': article['@id'], 'title': article['@title']}


def get_labels(article):
    return {'id': article['@id'], 'label': article['@hyperpartisan'], 'url': article['@url']}


def extract_training_data():
    with codecs.open("data/classifiedArticles.xml", 'r', encoding='utf8') as f:
        text = f.read()
        labels_raw = bf.data(fromstring(text))
    with codecs.open("data/articles.xml", 'r', encoding='utf8') as f:
        text = f.read()
        articles_raw = bf.data(fromstring(text))
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
    vectorizer.fit(df_train.title)
    train_x = np.concatenate((
        vectorizer.transform(df_train.title).toarray(),
        df_train.https.values.reshape(-1, 1),
    ), axis=1)
    test_x = np.concatenate((
        vectorizer.transform(df_test.title).toarray(),
        df_test.https.values.reshape(-1, 1),
    ), axis=1)

    train_y = df_train.label.values
    test_y = df_test.label.values
    return train_x, test_x, train_y, test_y


def train_using_keras(train_x, test_x, train_y, test_y):
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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', action='store_true', default=False, help='Train Keras baselinem model, if false then use logistic regression')
    args = vars(arg_parser.parse_args())
    train_x, test_x, train_y, test_y = extract_training_data()
    print("train_x.shape:", train_x.shape, "test_x.shape", test_x.shape)
    print("train_y.shape:", train_y.shape, "test_y.shape", test_y.shape)
    if args['k']:
        train_using_keras(train_x, test_x, train_y, test_y)
    else:
        # use logistic regression
        fit_using_logistic_regression(train_x, test_x, train_y, test_y)
