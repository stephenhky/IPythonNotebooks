
import numpy as np
import shorttext

import urllib2


# load file
textfile = urllib2.urlopen('http://norvig.com/big.txt', 'r')
text = filter(lambda t: len(t)>0, [t.strip() for t in textfile])

# encoder
chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(text)

# hyperparameters
numchars = len(chartovec_encoder.dictionary)
latent_dim = numchars + 20

# training
seq2seqer = shorttext.generators.CharBasedSeq2SeqGenerator(chartovec_encoder, latent_dim, 120)
seq2seqer.train(text, epochs=100)

# save the model
seq2seqer.save_compact_model('norvigtxt_iter5model_big.bin')
