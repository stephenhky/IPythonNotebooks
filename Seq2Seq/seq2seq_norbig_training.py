
import urllib2

import numpy as np
import keras

from s2sutils import initSentenceToCharVecEncoder

# read the corpus
textfile = urllib2.urlopen('http://norvig.com/big.txt', 'r')
text = filter(lambda t: len(t)>0, [t.strip() for t in textfile])

# initialize one hot encoder
chartovec_encoder = initSentenceToCharVecEncoder(text)


numchars = len(chartovec_encoder.dictionary)
latent_dim = numchars + 20

print "numchars = ", numchars
print "latent_dim = ", latent_dim

# define training model
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, numchars))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, numchars))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(numchars, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# preparing training data
encoder_input = chartovec_encoder.encode_sentences(text[:-1], startsig=True, maxlen=20, sparse=False)
decoder_input = chartovec_encoder.encode_sentences(text[1:], startsig=True, maxlen=20, sparse=False)
decoder_output = chartovec_encoder.encode_sentences(text[1:], endsig=True, maxlen=20, sparse=False)

# compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Training
model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=64,
          epochs=100)

# persist the model
chartovec_encoder.dictionary.save('chartovec_big.dict')

model.save('s2s_big.h5')
open('s2s_big.json', 'wb').write(model.to_json())

encoder_model.save('s2s_encoder_big.h5')
open('s2s_encoder_big.json', 'wb').write(encoder_model.to_json())
decoder_model.save('s2s_decoder_big.h5')
open('s2s_decoder_big.json', 'wb').write(decoder_model.to_json())
