from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from xml.dom import minidom
import codecs
import re


def strip_html_tags(text: str):
    p = re.compile(r'</*(p|q|a(?!rticle))+.*?>')
    return p.sub('', text)


class Doc2VecModelBuilder:
    def __init__(self, articles_file: str):
        dom = None
        self.model = None
        with codecs.open(articles_file, 'r', encoding='utf8') as f:
            text = f.read()
            dom = minidom.parseString(strip_html_tags(text))
        articles_list = dom.getElementsByTagName('article')
        # TODO use from nltk.tokenize import word_tokenize instead of splitting string
        # print(articles_list[0].childNodes[0].nodeValue)
        self.tagged_docs = [
            TaggedDocument(words=article.childNodes[0].nodeValue.split(), tags=article.attributes['id'].value) for
            article in articles_list]

    def train(self):
        MAX_EPOCHS = 100
        VEC_SIZE = 20
        ALPHA = 0.025
        MIN_ALPHA = 0.00025
        LEARNING_RATE_DECAY = 0.0002
        self.model = Doc2Vec(vector_size=VEC_SIZE,
                             alpha=ALPHA,
                             min_alpha=MIN_ALPHA,
                             min_count=1,
                             dm=1)
        self.model.build_vocab(self.tagged_docs)
        for epoch in range(MAX_EPOCHS):
            print('iteration {0}'.format(epoch))
            self.model.train(self.tagged_docs,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
            # decrease the learning rate
            self.model.alpha -= LEARNING_RATE_DECAY
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha

    def save_model(self):
        self.model.save("d2v_news_articles.model")
        print("Model Saved")
