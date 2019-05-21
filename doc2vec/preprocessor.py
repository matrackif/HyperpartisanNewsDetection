import re
import codecs
import pandas as pd
from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring
from nltk.corpus import stopwords


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


def analyze_articles(article_file):
    articles_raw = read_file(article_file)
    describe_dataset(articles_raw)


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