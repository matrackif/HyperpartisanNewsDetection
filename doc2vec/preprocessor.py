import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords') Use this when running for first time
import re


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


