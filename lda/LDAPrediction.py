from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from utils.config import Config

config = Config.open("../config.yml")


def load_model_dict():
    with open(config("path.lda-pointer"), "r") as f:
        label = f.readlines()[0]
    model_path = config("path.lda-model").format(label)

    return LdaMulticore.load(model_path), Dictionary.load(model_path + '.id2word')


def get_topics_of_doc(model: LdaMulticore, dictionary: Dictionary, words: str):
    bow = dictionary.doc2bow(words.split(' '))
    return model[bow]


def get_topics_of_corpus(model: LdaMulticore, dictionary: Dictionary, corpus: list):
    return [get_topics_of_doc(model, dictionary, doc) for doc in corpus]


def topics_repr_to_string(topics_of_doc):
    return str(list(list(t) for t in topics_of_doc))


def run(parts_names=None):
    model, dictionary = load_model_dict()

    for part in parts_names or config("splits").dict_like.keys():
        with open(config("path.stems-split").format(part), "r") as test_file, \
             open(config("path.topic-distr").format(part), "w") as distr_file:

            for doc in test_file:
                topics = get_topics_of_doc(model, dictionary, doc)
                print(topics_repr_to_string(topics), file=distr_file)


if __name__ == '__main__':
    run(['holdout'])
