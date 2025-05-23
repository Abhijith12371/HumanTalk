# 🎹 Emotion Detection Bot

A Python-based Emotion Detection Bot that analyzes audio input and predicts the speaker's emotion using state-of-the-art transformer models and signal processing techniques.

## 🚀 Features

* 🔊 Real-time audio recording and preprocessing
* 🧠 Transformer-based emotion classification using `transformers`, `peft`, and `sentence-transformers`
* 🔉 Noise reduction and audio feature extraction using `librosa` and `noisereduce`
* 🧾 Integration with `datasets` and `huggingface-hub` for dataset/model loading
* 📂 Redis caching for fast inference
* 📈 Progress tracking with `tqdm`
* ✅ Unit-tested components using `unittest`

---

## 🛠️ Installation

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📦 Dependencies

Make sure your `requirements.txt` includes:

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.0.0
sounddevice>=0.4.0
soundfile>=0.10.0
numpy>=1.20.0
redis>=4.0.0
sentence-transformers>=2.0.0
huggingface-hub>=0.0.0
pyyaml>=5.0.0
librosa>=0.9.0
noisereduce>=1.0.0
tqdm>=4.0.0
unittest>=1.0.0
```

---

## 🎯 Usage

### 1. Record and Process Audio

```python
from audio_utils import record_audio, preprocess_audio

waveform = record_audio(duration=5)
clean_waveform = preprocess_audio(waveform)
```

### 2. Predict Emotion

```python
from emotion_detector import EmotionDetector

detector = EmotionDetector()
emotion = detector.predict(clean_waveform)
print(f"Detected Emotion: {emotion}")
```

---

## 🤪 Testing

Run unit tests:

```bash
python -m unittest discover tests/
```

---

## 🧠 Model Details

The bot uses a fine-tuned transformer model (like `facebook/wav2vec2-base-960h`) or sentence embeddings with `sentence-transformers` to detect emotions such as:

* 😃 Happy
* 😢 Sad
* 😡 Angry
* 😨 Fear
* 😐 Neutral

---

## 🔧 Project Structure

```
emotion-bot/
│
├── audio_utils.py            # Audio recording & preprocessing
├── emotion_detector.py       # Inference logic using Hugging Face
├── config.yaml               # Config for model and parameters
├── requirements.txt
├── README.md
└── tests/
    ├── test_audio_utils.py
    └── test_emotion_detector.py
```

---

## 🧠 Future Improvements

* Integrate facial emotion detection via webcam
* Add multilingual emotion detection
* Deploy as a web app or chatbot

---



## 📄 License

This project is licensed under the MIT License.
