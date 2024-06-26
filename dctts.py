import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa
import soundfile as sf


# Constants
sampling_rate = 22050  # Sampling rate of LJSpeech dataset
n_fft = 1024  # FFT window size
hop_length = 256  # Hop length for STFT
n_mels = 80  # Number of Mel filterbanks
text_dim = 80  # Dimensionality of input text
mel_dim = 80  # Dimensionality of Mel spectrogram
hidden_dim = 256  # Hidden dimension for encoder and decoder


# Encoder Block (modificat)
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2, 2)  # Adăugare MaxPool1d
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.pool(x)  # Aplicare MaxPool1d
        x = self.norm2(F.relu(self.conv2(x)))
        return x


# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        return x

# Text2Mel Network
class Text2Mel(nn.Module):
    def __init__(self, input_dim, hidden_dim, mel_dim):
        super(Text2Mel, self).__init__()
        self.encoder = EncoderBlock(input_dim, hidden_dim)
        self.decoder = DecoderBlock(hidden_dim, hidden_dim)
        self.mel_linear = nn.Conv1d(hidden_dim, mel_dim, kernel_size=1) # Adăugat strat liniar

    def forward(self, text):
        encoded_text = self.encoder(text)
        decoded_text = self.decoder(encoded_text)
        mel_spectrogram = self.mel_linear(decoded_text)
        return mel_spectrogram


# Super-resolution Network
class SuperResolution(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SuperResolution, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim * 2, kernel_size=5, padding=2)  # Output 2x channels for upsampling
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(input_dim * 2)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        return x

class WaveNet(nn.Module):
    def __init__(self, input_channels=80, residual_channels=64, dilation_channels=32, skip_channels=64,
                 num_blocks=2, num_layers_per_block=10, quantize=256):
        super(WaveNet, self).__init__()

        self.input_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.dilated_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.num_layers = num_blocks * num_layers_per_block

        for block in range(num_blocks):
            for layer in range(num_layers_per_block):
                dilation = 2**layer
                # Calculate padding to maintain output length
                padding = (2 - 1) * dilation
                self.dilated_layers.append(
                    nn.Conv1d(residual_channels, 2 * dilation_channels, kernel_size=2, dilation=dilation, padding=padding)
                )
                self.skip_layers.append(
                    nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)
                )

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, quantize, kernel_size=1)
        self.final_conv = nn.Conv1d(quantize, n_mels, kernel_size=1)  # Added

    def forward(self, x):
        x = self.input_conv(x)
        skip_outputs = []
        for i in range(self.num_layers):
            dilated_output = self.dilated_layers[i](x)
            tanh_out, sig_out = torch.chunk(dilated_output, 2, dim=1)
            gated_output = torch.tanh(tanh_out) * torch.sigmoid(sig_out)
            skip_output = self.skip_layers[i](gated_output)

            # Dynamically adjust skip_output length
            skip_output = skip_output[:, :, :x.size(2)]  # Trim or pad to match x

            x = skip_output + x
            skip_outputs.append(skip_output)

        output = torch.sum(torch.stack(skip_outputs), dim=0)
        output = F.relu(output)
        output = self.output_conv1(output)
        output = F.relu(output)
        output = self.output_conv2(output)
        output = self.final_conv(output)
        return output

# DCTTS Model (modificat)
class DCTTS(nn.Module):
    def __init__(self, text_dim, mel_dim, hidden_dim):
        super(DCTTS, self).__init__()
        self.text2mel = Text2Mel(text_dim, hidden_dim, mel_dim)
        self.channel_reducer = nn.Conv1d(mel_dim * 2, mel_dim, kernel_size=1)
        self.super_res = SuperResolution(mel_dim, hidden_dim)
        self.wavenet = WaveNet()

    def forward(self, text, target_mels=None):
        mel_spectrogram = self.text2mel(text)
        high_res_mel = self.super_res(mel_spectrogram)
        high_res_mel = self.channel_reducer(high_res_mel)
        waveform = self.wavenet(high_res_mel)

        # Determine the desired output length
        target_length = target_mels.size(2) if target_mels is not None else mel_spectrogram.size(2) * 2  # For inference

        # Pad or trim the waveform
        if waveform.size(2) < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.size(2)), 'constant')  # Pad
        else:
            waveform = waveform[:, :, :target_length]  # Trim

        return waveform

# Dataset class for LJSpeech
class LJDataset(Dataset):
    def __init__(self, root_dir, metadata_file, max_sequence_length=100, num_samples=None):
        self.root_dir = root_dir
        self.max_sequence_length = max_sequence_length
        self.data = self.load_metadata(metadata_file)
        if num_samples is not None:
            self.data = self.data[:num_samples]

    def load_metadata(self, metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = [line.strip().split('|') for line in f]
        return metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.root_dir, self.data[idx][0] + ".wav")
        text = self.data[idx][2]

        waveform, _ = torchaudio.load(wav_file)
        waveform = waveform.numpy()[0]  # Convert to numpy array

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate, n_fft=n_fft,
                                                         hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Trim or pad sequence to fixed length
        if mel_spectrogram.shape[1] > self.max_sequence_length:
            mel_spectrogram = mel_spectrogram[:, :self.max_sequence_length]
        else:
            pad_width = self.max_sequence_length - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant',
                                     constant_values=-80.0)

        # Convert to PyTorch tensor
        mel_spectrogram = torch.FloatTensor(mel_spectrogram)

        # Convert text to one-hot encoding (dummy representation)
        # Replace with actual text processing logic
        text_tensor = torch.zeros(text_dim, dtype=torch.float32)
        for i, char in enumerate(text):
            if i < text_dim:
                text_tensor[i] = ord(char) / 256.0  # Normalize ASCII values

        return text_tensor, mel_spectrogram


# Function to collate batch data (pad sequences)
def collate_fn(batch):
    texts, mels = zip(*batch)
    max_text_len = max(text.size(0) for text in texts)
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.float32)
    for i, text in enumerate(texts):
        padded_texts[i, :text.size(0)] = text

    # 3. Trim all mels to the shortest mel in the batch
    min_mel_len = min(mel.size(1) for mel in mels)
    padded_mels = torch.zeros(len(mels), n_mels, min_mel_len, dtype=torch.float32)
    for i, mel in enumerate(mels):
        padded_mels[i, :, :] = mel[:, :min_mel_len]

    return padded_texts, padded_mels


def train(model, train_loader, criterion, optimizer, num_epochs, model_path):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for texts, mels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            texts, mels = texts.to(device), mels.to(device)
            outputs = model(texts, mels)  # Pass mels to the model
            loss = criterion(outputs, mels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f'Train Loss: {epoch_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)


# Function to generate audio from text
def generate_audio(model, text, output_file):
    model.eval()
    text_tensor = torch.zeros(text_dim, dtype=torch.float32).to(device)
    for i, char in enumerate(text):
        if i < text_dim:
            text_tensor[i] = ord(char) / 256.0
    with torch.no_grad():
        mel_spectrogram = model.text2mel(text_tensor.unsqueeze(0))
        high_res_mel = model.super_res(mel_spectrogram)
        waveform = model(text_tensor.unsqueeze(0), high_res_mel)  # Pass high_res_mel to the model
        waveform = waveform.squeeze(0).cpu().numpy()  # remove batch dimension
    # Assuming WaveNet used mu-law
    waveform = librosa.mu_expand(waveform)
    waveform = librosa.db_to_power(waveform)
    waveform = librosa.griffinlim(waveform, hop_length=hop_length, win_length=100)
    waveform /= np.abs(waveform).max()

    sf.write(output_file, waveform, samplerate=sampling_rate)

# Main
if __name__ == "__main__":
    # Paths and hyperparameters
    root_dir = 'LJSpeech-1.1/wavs'
    metadata_file = 'LJSpeech-1.1/metadata.csv'
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3

    # Model path
    model_path = "dctts_model.pth"

    # Initialize model, optimizer, and criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCTTS(text_dim, mel_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Load model if it exists, otherwise train and save
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model încărcat din", model_path)
    else:
        # Initialize dataset and data loader
        dataset = LJDataset(root_dir, metadata_file, num_samples=1000)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        train(model, train_loader, criterion, optimizer, num_epochs, model_path)

    # Generate audio from text
    input_text = "Heck yeah this is a generated dctts text"
    output_file = "output.wav"
    generate_audio(model, input_text, output_file)
