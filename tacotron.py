import tensorflow as tf

class Tacotron2(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, **kwargs):
        super(Tacotron2, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.decoder_lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(80)  # Assuming 80 mel bands

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        encoder_output, state_h, state_c = self.encoder_lstm(x, training=training)
        decoder_output, _, _ = self.decoder_lstm(encoder_output, initial_state=[state_h, state_c])
        mel_output = self.dense(decoder_output)
        return mel_output

    def get_config(self):
        config = super(Tacotron2, self).get_config()
        config.update({
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'lstm_units': self.encoder_lstm.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            lstm_units=config['lstm_units']
        )
