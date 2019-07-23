import re
import codecs
import pandas as pd
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('wordnet')
nltk.download('stopwords')


def preprocess(text):
    # Replace any non-alpha numeric character with empty string
    # TODO remove URLs
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    # Remove stopwords
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def preprocess_Bert(text):
    # Replace any non-alpha numeric character with empty string

    text = re.sub(r"[^A-Za-z0-9!?]", " ", text)
    return text


def analyze_articles(article_file):
    articles_raw = read_file(article_file)
    describe_dataset(articles_raw)
    corpus = create_corpus(get_articles(articles_raw), False)
    print(get_most_frequent_unigrams(corpus, 20))
    print(get_most_frequent_bigrams(corpus, 20))
    print(get_most_frequent_trigrams(corpus, 20))


def read_file(filename):
    with codecs.open(filename, 'r', encoding='utf8') as f:
        text = f.read()
        file_raw = bf.data(fromstring(text))
    return file_raw


def describe_dataset(articles_raw):
    articles = get_articles(articles_raw)
    word_count = get_articles_word_count(articles_raw)
    df_counts = pd.DataFrame(word_count)
    print("\nNumber of words distribution")
    print(df_counts.describe())
    print("\nTop 20 frequent words")
    freq = pd.Series(' '.join(articles).split()).value_counts()[:20]
    print(freq)


def get_articles(articles_raw):
    articles = []
    for article in articles_raw['articles']['article']:
        articles.append(get_article_text(article, ''))
    return articles


def get_articles_word_count(articles_raw):
    word_count = []
    for article in articles_raw['articles']['article']:
        word_count.append(len(str(article).split(" ")))
    return word_count


def get_article_text(article, txt):
    for item in article.items():
        if not item[0].startswith('@'):
            if type(item[1]) in [str, int, bool]:
                txt += str(item[1]) + ' '
            else:
                if type(item[1]) is list:
                    for l in item[1]:
                        txt += get_article_text(l, '')
                else:
                    txt += get_article_text(item[1], '')
    return txt


def create_corpus(articles, should_treat_each_article_separately: bool):
    corpus = []
    stop_words = get_stop_words()
    lem = WordNetLemmatizer()
    for i in range(0, len(articles)):
        text = re.sub('[^a-zA-Z]', ' ', articles[i])
        text = re.sub("(\\d|\\W)+", " ", text)
        text = text.lower().split()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        if should_treat_each_article_separately:
            text = text.split()
        corpus.append(text)
    return corpus



def create_corpus_Bert(articles, should_treat_each_article_separately: bool):
    corpus = []
    stop_words = get_stop_words()
    lem = WordNetLemmatizer()
    for i in range(0, len(articles)):
        text = re.sub('[^a-zA-Z!?]', ' ', articles[i])
        text = re.sub("(\\d|\\W)+", " ", text)
        text = text.lower().split()
        text = [lem.lemmatize(word) for word in text if not word in stop_words]
        text = " ".join(text)
        if should_treat_each_article_separately:
            text = text.split()
        corpus.append(text)
    return corpus

def get_stop_words():
    stop_words = set(stopwords.words("english"))
    new_words = ["one", "would", "also"]
    return stop_words.union(new_words)


def get_most_frequent_unigrams(corpus, n: int):
    vec = CountVectorizer().fit(corpus)
    return get_most_frequent(vec, corpus, n)


def get_most_frequent_bigrams(corpus, n: int):
    vec = CountVectorizer(ngram_range=(2, 2),
                           max_features=2000).fit(corpus)
    return get_most_frequent(vec, corpus, n)


def get_most_frequent_trigrams(corpus, n: int):
    vec = CountVectorizer(ngram_range=(3, 3),
                           max_features=2000).fit(corpus)
    return get_most_frequent(vec, corpus, n)


def get_most_frequent(vec: CountVectorizer, corpus, n: int):
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]
