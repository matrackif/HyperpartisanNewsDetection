from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from xml.dom import minidom
import codecs
import re
from doc2vec.preprocessor import preprocess
TRAIN_PERCENTAGE = 0.8
VEC_SIZE = 100


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
        # print(preprocess(articles_list[0].childNodes[0].nodeValue).split())
        self.tagged_docs = [
            TaggedDocument(words=article.attributes['title'].value.split(), tags=[int(article.attributes['id'].value)]) for
            article in articles_list]

    def train(self):
        MAX_EPOCHS = 20
        ALPHA = 0.025
        MIN_ALPHA = 0.00025
        LEARNING_RATE_DECAY = 0.0002
        """
        If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        """
        self.model = Doc2Vec(vector_size=VEC_SIZE,
                             alpha=ALPHA,
                             min_alpha=MIN_ALPHA,
                             min_count=1,
                             dm=1)
        self.model.build_vocab(self.tagged_docs)
        print("Corpus count:", self.model.docvecs.count)
        for epoch in range(MAX_EPOCHS):
            print("iteration: {0}/{1}".format(epoch + 1, MAX_EPOCHS))
            self.model.train(self.tagged_docs[:int(TRAIN_PERCENTAGE * len(self.tagged_docs))],
                             total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
            # decrease the learning rate
            self.model.alpha -= LEARNING_RATE_DECAY
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha
        print(self.model.docvecs[11], self.model.docvecs[11].shape)

    def save_model(self):
        self.model.save("news_articles.d2v")
        print("Model Saved")
