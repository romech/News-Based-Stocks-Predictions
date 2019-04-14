import logging
from datetime import datetime
import os

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim

from utils.config import Config
from utils.iterators import grouper

config = Config.open("../config.yml")
lda_cfg = config("lda")


def train(docs):
    num_topics = lda_cfg("topics")
    epochs = lda_cfg("epochs")
    label = f'{datetime.now().isoformat(".", timespec="minutes")}({num_topics}-topics,{epochs}-epochs)'

    log_path = config("path.lda-log").format(label)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=lda_cfg("word-extremes.min-count"),
                               no_above=lda_cfg("word-extremes.max-freq"))
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    model = LdaMulticore(corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         passes=epochs,
                         eval_every=lda_cfg.dict_like.get("eval-every"))

    return label, model, dictionary, corpus


def train_save_trace(docs):
    label, model, dictionary, corpus = train(docs)
    model.save(config("path.lda-model").format(label))
    with open(config("path.lda-pointer"), "w") as f:
        f.write(label)

    top_topics = model.top_topics(corpus, topn=20)
    _output_summary(top_topics, config("path.lda-summary").format(label))
    if lda_cfg("visualization"):
        _output_visualization(config("path.lda-vis").format(label), model, corpus, dictionary)


def _output_summary(top_topics, path):
    with open(path, "w") as f:
        for i, topic_repr in enumerate(top_topics):
            words = _topic_repr_to_word(topic_repr)
            print(f"Topic {i}:", file=f)
            print(*map('\t'.join, grouper(words, 5)), sep='\n', file=f)


def _topic_repr_to_word(topic_repr):
    return [word[1] for word in topic_repr[0]]


def _output_visualization(path, model, corpus, dictionary):
    prepared_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    with open(path, "w") as f:
        pyLDAvis.save_html(prepared_data, f)


def run():
    with open(config("path.stems-split").format("train"), "r") as f:
        news = [line.strip().split(' ') for line in f]
        train(news)


if __name__ == '__main__':
    run()
