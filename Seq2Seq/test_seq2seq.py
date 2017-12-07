from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

timesteps = 10
input_dim = 2

latent_dim = 5

# model
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

