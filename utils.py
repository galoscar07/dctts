import time

import librosa
import numpy as np
import tensorflow as tf
from librosa import griffinlim
from librosa.feature.inverse import mel_to_stft
from scipy.io.wavfile import write


def load_audio(file_path, sample_rate=22050):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def extract_mel_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram.T

def generate_spectrogram(model, text, tokenizer, max_sequence_length):
    # Tokenize input text
    sequence = tokenizer.texts_to_sequences([text])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=max_sequence_length)
    sequence = np.array(sequence)

    # Predict the mel spectrogram
    spectrogram = model.predict(sequence)
    print("Generated spectrogram shape:", spectrogram.shape)
    print("Spectrogram data (first 5 elements):", spectrogram[0][:5])
    return spectrogram[0]

def spectrogram_to_audio(spectrogram, sr=22050, n_fft=1024, hop_length=256):
    # Convert the log-mel spectrogram to a mel spectrogram
    mel_spec = librosa.db_to_power(spectrogram.T, ref=1.0)

    # Convert mel spectrogram to STFT
    stft = mel_to_stft(mel_spec, sr=sr, n_fft=n_fft)

    # Reconstruct audio using Griffin-Lim
    audio = griffinlim(stft, hop_length=hop_length)
    print("Reconstructed audio shape:", audio.shape)
    return audio

def save_audio(audio, filename, sr=22050):
    # Normalize audio to -1 to 1
    audio = audio / np.max(np.abs(audio))
    write(filename, sr, audio.astype(np.float32))
    print(f"Audio saved to {filename}")

def measure_time(start):
    end = time.time()
    return end - start

def generate_audio(model, tokenizer, max_sequence_length, text):
    # 4. Interface
    start_time = time.time()
    print("4. Interface")

    # 4.1 Text for testing
    print("4.1 Text for testing")

    # 4.2 Generate spectrogram
    print("4.2 Generate spectrogram")
    spectrogram = generate_spectrogram(model, text, tokenizer, max_sequence_length)

    # 4.3 Convert spectrogram to audio
    print("4.3 Convert spectrogram to audio")
    audio = spectrogram_to_audio(spectrogram)

    # 4.4 Save output
    print("4.4 Save output")
    save_audio(audio, 'output.wav')

    interface_time = measure_time(start_time)
    print(f"Interface Time: {interface_time:.2f} seconds")
