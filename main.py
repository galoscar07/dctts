import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import time
import matplotlib.pyplot as plt

from tacotron import Tacotron2
from utils import load_audio, extract_mel_spectrogram, measure_time, generate_audio

# Paths
data_path = "LJSpeech-1.1"
model_path = "tacotron2_model.keras"
metadata_path = os.path.join(data_path, 'metadata.csv')

# Hyperparams
embedding_dim = 256
lstm_units = 512

# Global vars
audio_dir = os.path.join(data_path, 'wavs')
mel_spectrograms = []

print("START")
# 1. Data Preprocessing
start_time = time.time()
print("1. Preprocessing Data")

# 1.1. Load transcripts
print("1.1. Loading transcripts")
with open(metadata_path, 'r') as file:
    lines = file.readlines()

# 1.2. Extract pairs of (filename, transcript)
print("1.2. Extracted pairs")
pairs = [(line.split('|')[0], line.split('|')[2].strip()) for line in lines]

# 1.3. Text Tokenization
print("1.3. Text Tokenization")
tokenizer = Tokenizer(char_level=True)
texts = [pair[1] for pair in pairs]
tokenizer.fit_on_texts(texts)

# 1.4. Transform text into ID sequences
print("1.4. Transforming text into ID sequences")
sequences = tokenizer.texts_to_sequences(texts)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
vocab_size = len(tokenizer.word_index) + 1

# 1.5. Create mel spectrograms
print("1.5. Creating mel spectrograms")
for pair in pairs:
    file_path = os.path.join(audio_dir, pair[0] + '.wav')
    audio = load_audio(file_path)
    mel_spec = extract_mel_spectrogram(audio)
    mel_spectrograms.append(mel_spec)

# 1.6 Padding spectrograms to the same length
print("1.6 Padding spectrograms to the same length")
max_len = max(mel_spec.shape[0] for mel_spec in mel_spectrograms)
mel_spectrograms = [np.pad(mel_spec, ((0, max_len - mel_spec.shape[0]), (0, 0)), mode='constant') for mel_spec in mel_spectrograms]
mel_spectrograms = np.array(mel_spectrograms)

# Ensure sequences are padded to the same length
max_sequence_length = sequences.shape[1]
mel_spectrograms = mel_spectrograms[:, :max_sequence_length, :]  # Truncate/pad to the same length

print("Shape of sequences:", sequences.shape)
print("Shape of mel spectrograms:", mel_spectrograms.shape)

data_preprocessing_time = measure_time(start_time)
print(f"Data Preprocessing Time: {data_preprocessing_time:.2f} seconds")

# 2. Model Construction
start_time = time.time()
print("2. Model Initialization")

# Load existing model if available
if os.path.exists(model_path):
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_path, custom_objects={'Tacotron2': Tacotron2})
else:
    print("Creating new model...")
    model = Tacotron2(vocab_size, embedding_dim, lstm_units)

    model_construction_time = measure_time(start_time)
    print(f"Model Construction Time: {model_construction_time:.2f} seconds")

    # 3. Model Training
    start_time = time.time()
    print("3. Model Training")

    # 3.1 Defining optimizer and loss function
    print("3.1 Defining optimizer and loss function")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # 3.2 Compiling model
    print("3.2 Compiling model")
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Callback to save the loss values
    class LossHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    loss_history = LossHistory()

    # 3.3 Training
    print("3.3 Training")
    history = model.fit(sequences, mel_spectrograms, epochs=10, batch_size=16, callbacks=[loss_history])
    model_training_time = measure_time(start_time)
    print(f"Model Training Time: {model_training_time:.2f} seconds")

    # Save the model after training
    model.save(model_path)

    # Plotting the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history.losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.show()

text = "Hello, this is a test of the Tacotron 2 model."
generate_audio(model, tokenizer, max_sequence_length, text)


print("END")

