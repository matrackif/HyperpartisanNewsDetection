from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import models.models
import time
from matplotlib import pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
import doc2vec.preprocessor as pre
import keras
from typing import Union


def get_article_data_score(articles_file: str, classified_articles_file: str):
    labels_raw = pre.read_file(classified_articles_file)
    articles_raw = pre.read_file(articles_file)
    articles, ids = pre.get_articles(articles_raw)
    corpus = pre.create_corpus(articles, False)
    emotions_data = pre.read_emotions_data()
    political_data = pre.read_political_data()
    scores = []
    for i in range(0, 100):
        scores.append(pre.get_article_score(corpus[i], emotions_data, political_data))
        print(i)
    df_articles = pd.DataFrame(scores)
    df_articles['id'] = ids[0:100]
    labels = [get_labels(article) for article in labels_raw['articles']['article']]
    df_labels = pd.DataFrame(labels)
    df = df_articles.merge(df_labels, on='id')
    x_train, x_test, y_train, y_test = train_test_split(df, df.label, test_size=0.1)
    return {'train_x': x_train.iloc[:, 0:11], 'test_x': x_test.iloc[:, 0:11], 'train_y': y_train, 'test_y': y_test}


def get_data_from_article(article):
    return {'id': article['@id'], 'title': article['@title']}


def get_labels(article):
    return {'id': article['@id'], 'label': article['@hyperpartisan'], 'url': article['@url']}


def get_df_from_articles_file(articles_file: str, classified_articles_file: str):
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
    # print("Correlation between https and Hyperpartisan", df.corr())
    df.title = df.title.apply(str)
    article_titles = df.title.values.tolist()
    preprocessed_titles = pre.create_corpus(article_titles, False)
    df.title = preprocessed_titles
    return df


def transform_to_tfid(articles_file: str, classified_articles_file: str):
    df = get_df_from_articles_file(articles_file, classified_articles_file)
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
    return {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y, 'input_length': train_x.shape[1], 'vocab_size': train_x.shape[1]}


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://nlpforhackers.io/keras-intro/
def transform_to_dense_representation(articles_file: str, classified_articles_file: str):
    MAX_TITLE_LENGTH = 10  # max of 15 words in title
    df = get_df_from_articles_file(articles_file, classified_articles_file)

    # TODO use tf_id representation instead of padded vectors as lstm input
    # For now  we only use it to count number of unique words
    vectorizer = TfidfVectorizer(stop_words='english', )
    vectorizer.fit_transform(df.title.values)
    num_unique_words = len(vectorizer.get_feature_names())

    # integer encode the documents
    encoded_docs = [one_hot(d, num_unique_words) for d in df.title.values]
    # pad documents to a max length of MAX_TITLE_LENGTH words
    ps = np.array(pad_sequences(encoded_docs, maxlen=MAX_TITLE_LENGTH, padding='post'))

    train_x, test_x = None, None
    train_y, test_y = None, None
    # TODO figure out how to use train_test_split() provided by pandas for this case instead of using StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_indexes, test_indexes in sss.split(ps, df.label):
        # print("TRAIN indexes:", train_indexes, "TEST indexes:", test_index)
        train_x, test_x = ps[train_indexes], ps[test_indexes]
        train_y, test_y = df.label[train_indexes], df.label[test_indexes]
        break

    return {'train_x': train_x, 'test_x': test_x, 'train_y': train_y, 'test_y': test_y,
            'input_length': MAX_TITLE_LENGTH, 'vocab_size': num_unique_words}


def initialize_training_test_data(articles_file: str, classified_articles_file: str, model_type: str):
    TRAINING_DATA_EXTRACTOR_FUNCS = {'logistic': transform_to_tfid,
                                     'dense': transform_to_tfid,
                                     'lstm': transform_to_dense_representation,
                                     'sentiment': get_article_data_score}
    if model_type in TRAINING_DATA_EXTRACTOR_FUNCS.keys() and TRAINING_DATA_EXTRACTOR_FUNCS[model_type] is not None:
        return TRAINING_DATA_EXTRACTOR_FUNCS[model_type](articles_file, classified_articles_file)
    return None


def train_and_predict(articles_file: str, classified_articles_file: str, model_type: str):
    ret = initialize_training_test_data(articles_file, classified_articles_file, model_type)
    train_x = ret['train_x']
    train_y = ret['train_y']
    test_x = ret['test_x']
    test_y = ret['test_y']
    print("train_x.shape:", train_x.shape, "test_x.shape", test_x.shape)
    print("train_y.shape:", train_y.shape, "test_y.shape", test_y.shape)
    model = models.models.get_model(model_type, ret)
    if model_type == 'logistic':
        model.fit(train_x, train_y)
    elif model_type == 'sentiment':
        predict(train_x, train_y)
    else:
        # Keras model
        model.summary()
        start_time = time.time()
        history = model.fit(train_x, train_y, epochs=256, validation_data=(test_x, test_y),
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
    if model_type != 'sentiment':
        print(model_type, 'train data statistics:')
        print_and_plot_statistics(model, train_x, train_y)
        print(model_type, 'test data statistics:')
        print_and_plot_statistics(model, test_x, test_y)
    return model


def predict(train_x, train_y):
    prediction = []
    for i in range(0, train_x.shape[0]):
        if ((train_x.iloc[i, 5] / train_x.iloc[i, 6] > 1.35 or
             train_x.iloc[i, 5] / train_x.iloc[i, 6] < 0.65) and
                train_x.iloc[i, 10] > 0.01):
            prediction.append(True)
        else:
            prediction.append(False)
    analyze_results(prediction, train_y)


def analyze_results(pred, train_y):
    num_actual_hyperpartisan = np.sum(train_y)
    num_actual_nonhyperpartisan = np.count_nonzero(train_y == 0)

    num_pred_hyperpartisan = np.sum(pred)
    num_pred_nonhyperpartisan = np.count_nonzero(pred == 0)

    num_correct_pred = np.sum(pred == train_y)
    accuracy = (num_correct_pred / float(len(pred))) * 100
    print('------PREDICTION STATISTICS------')
    print('Total number of predictions:', len(pred))
    print('Actual number of hyperpartisan articles:', num_actual_hyperpartisan)
    print('Actual number of non-hyperpartisan articles:', num_actual_nonhyperpartisan)
    print('Predicted number of hyperpartisan articles:', num_pred_hyperpartisan)
    print('Predicted number of non-hyperpartisan articles:', num_pred_nonhyperpartisan)
    print('Number of correct predictions:', num_correct_pred)
    print('Prediction accuracy: {}%'.format(accuracy))
    plot_confusion_matrix(train_y, pred)
    plt.show()
    print(classification_report(train_y, pred))


def print_and_plot_statistics(model: Union[keras.Model, LogisticRegression], input_x: np.ndarray, actual_y: np.ndarray):
    pred = np.round(model.predict(input_x), decimals=0).reshape(-1)
    num_actual_hyperpartisan = np.sum(actual_y)
    num_actual_nonhyperpartisan = np.count_nonzero(actual_y == 0)

    num_pred_hyperpartisan = np.sum(pred)
    num_pred_nonhyperpartisan = np.count_nonzero(pred == 0)

    num_correct_pred = np.sum(pred == actual_y)
    accuracy = (num_correct_pred / float(pred.shape[0])) * 100
    print('------PREDICTION STATISTICS------')
    print('Total number of predictions:', pred.shape[0])
    print('Actual number of hyperpartisan articles:', num_actual_hyperpartisan)
    print('Actual number of non-hyperpartisan articles:', num_actual_nonhyperpartisan)
    print('Predicted number of hyperpartisan articles:', num_pred_hyperpartisan)
    print('Predicted number of non-hyperpartisan articles:', num_pred_nonhyperpartisan)
    print('Number of correct predictions:', num_correct_pred)
    print('Prediction accuracy: {}%'.format(accuracy))
    plot_confusion_matrix(actual_y, pred)
    plt.show()
    print(classification_report(actual_y, pred))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-a', '--articles', nargs='?', type=str, default='data/articles.xml',
                            const='data/articles.xml', help='Path to XML file of articles')
    arg_parser.add_argument('-c', '--classified-articles', nargs='?', type=str, default='data/classifiedArticles.xml',
                            const='data/classifiedArticles.xml', help='Path to XML file of classified articles')
    arg_parser.add_argument('-m', '--model', nargs='?', type=str, default='sentiment',
                            const='logistic',
                            help='This argument allows us to choose the model type. Supported values: dense (Keras), LSTM (Keras), logistic (sklearn Logistic Regression Classifier)')
    args = vars(arg_parser.parse_args())
    # print(args)
    # get_article_data_score(args['articles'], args['classified_articles'])
    # pre.analyze_articles(args['articles'])
    train_and_predict(args['articles'], args['classified_articles'], args['model'])
