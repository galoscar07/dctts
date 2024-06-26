# DCTTS Text-to-Speech Model

This project implements a Deep Convolutional Text-to-Speech (DCTTS) model using PyTorch. The model takes text as input and generates high-resolution Mel spectrograms, which can then be converted to audio waveforms.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dctts-tacotron.git
cd dctts-tacotron
```

### 2. Create a virtual environment
```bash
python -m venv venv
```
### 3. Activate the virtual environment
On Windows:
```bash
venv\Scripts\activate
```
On MacOS/Linux:
```bash
source venv/bin/activate
```
### 4. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
#### 1. Prepare the LJSpeech Dataset
Download the LJSpeech dataset from LJSpeech Dataset. Extract the contents and place the wavs folder and metadata.csv file in a directory of your choice.

#### 2. Update the paths in the script
Edit the root_dir and metadata_file variables in the main script to point to the location of the LJSpeech dataset files.

#### 3. Run the training script
```bash
python main.py
```

## Model Architecture
The model consists of two main components:

- Text2Mel Network: Converts input text to Mel spectrogram.

- Encoder Block: Processes the input text.
- Decoder Block: Generates Mel spectrogram from encoded text.
- Super-resolution Network: Enhances the resolution of the generated Mel spectrogram.

## Dataset
The LJSpeech dataset is used for training. It consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

## Training
To train the model, run the training script with the following command:

```bash
python main.py
```
The script will load the dataset, initialize the model, and start the training process.

## Dependencies
Python 3.8+
PyTorch
torchaudio
librosa
numpy
tqdm
Install the dependencies using the provided requirements.txt file.

## License
This project is licensed under the MIT License.

## Acknowledgments
The implementation is inspired by the original DCTTS paper: "Deep Convolutional Text-to-Speech".
The LJSpeech dataset is used for training and evaluation.