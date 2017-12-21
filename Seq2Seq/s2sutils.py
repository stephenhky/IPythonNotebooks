
from functools import partial

import numpy as np
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder


class SentenceToCharVecEncoder:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        numchars = len(self.dictionary)
        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(np.arange(numchars).reshape((numchars, 1)))

    def calculate_prelim_vec(self, cor_sent):
        return self.onehot_encoder.transform(
            np.array([self.dictionary.token2id[c] for c in cor_sent]).reshape((len(cor_sent), 1))
        )

    def encode_sentence(self, sent, startsig=False, endsig=False, maxlen=20):
        cor_sent = ('\n' if startsig else '') + sent[:min(maxlen, len(sent))] + ('\n' if endsig else '')
        sent_vec = self.calculate_prelim_vec(cor_sent).tocsc()
        if sent_vec.shape[0] == maxlen + startsig + endsig:
            return sent_vec
        else:
            return csc_matrix((sent_vec.data, sent_vec.indices, sent_vec.indptr),
                              shape=(maxlen + startsig + endsig, sent_vec.shape[1]),
                              dtype=np.float64)

    def encode_sentences(self, sentences, sparse=True, startsig=False, endsig=False, maxlen=20):
        encode_sent_func = partial(self.encode_sentence, startsig=startsig, endsig=endsig, maxlen=maxlen)
        list_encoded_sentences = map(encode_sent_func, sentences)
        if sparse:
            return list_encoded_sentences
        else:
            return np.array(map(lambda sparsevec: sparsevec.toarray(), list_encoded_sentences))


def textfile_generator(textfile, linebreak=True):
    for t in textfile:
        if len(t) > 0:
            yield t.strip() + ('\n' if linebreak else '')


def initSentenceToCharVecEncoder(textfile):
    # text = [t.strip()+'\n' for t in textfile if len(t) > 0]
    dictionary = Dictionary(map(lambda line: [c for c in line], textfile_generator(textfile)))
    return SentenceToCharVecEncoder(dictionary)