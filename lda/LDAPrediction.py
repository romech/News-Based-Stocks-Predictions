from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim

from utils.config import Config

config = Config.open("../config.yml")


def load_model_dict():
    model_path = config("path.lda-model").format(_get_latest_model_label())

    return LdaMulticore.load(model_path), Dictionary.load(model_path + '.id2word')


def get_topics_of_doc(model: LdaMulticore, dictionary: Dictionary, words: str):
    bow = dictionary.doc2bow(words.split(' '))
    return model[bow]


def get_topics_of_corpus(model: LdaMulticore, dictionary: Dictionary, corpus: list):
    return [get_topics_of_doc(model, dictionary, doc) for doc in corpus]


def topics_repr_to_string(topics_of_doc):
    return str(list(list(t) for t in topics_of_doc))


def run(parts_names=('train',)):
    model, dictionary = load_model_dict()

    for part in parts_names or config("splits").dict_like.keys():
        with open(config("path.stems-split").format(part), "r") as test_file, \
             open(config("path.topic-distr").format(part), "w") as distr_file:

            for doc in test_file:
                topics = get_topics_of_doc(model, dictionary, doc)
                print(topics_repr_to_string(topics), file=distr_file)


def build_visualization(path: str, model: LdaMulticore, dictionary: Dictionary, corpus: list):
    prepared_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    with open(path, "w") as f:
        pyLDAvis.save_html(prepared_data, f)


def run_visualization():
    model, dictionary = load_model_dict()
    label = _get_latest_model_label()
    with open(config("path.stems-split").format("train"), "r") as f:
        docs = [line.strip().split(' ') for line in f]
        for doc in docs:
            assert isinstance(doc, list)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
    build_visualization(config("path.lda-vis").format(label), model, dictionary, corpus)


def _get_latest_model_label():
    with open(config("path.lda-pointer"), "r") as f:
        return f.readlines()[0]


if __name__ == '__main__':
    # run(['holdout'])
    run_visualization()
