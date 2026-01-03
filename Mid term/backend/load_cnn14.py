import torch
from models import Cnn14  # This works because you're inside the repo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)

# Load pretrained weights (download from: https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth)
checkpoint = torch.load('./Cnn14_mAP=0.431.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

print("✅ Model loaded successfully.")



import librosa
import numpy as np

# Load audio file (replace 'example.wav' with your actual file)
audio_path = 'example.wav'
waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
waveform = waveform[None, :]  # shape: (1, time_steps)
waveform = torch.Tensor(waveform).to(device)

# Predict
with torch.no_grad():
    output = model(waveform)

# Print top predictions
clipwise_output = output['clipwise_output'].data.cpu().numpy()[0]
top_indices = np.argsort(clipwise_output)[::-1][:5]  # top 5

# Load class labels
import csv
with open('resources/classes.csv', newline='') as f:
    reader = csv.reader(f)
    classes = [row[0] for row in reader]

print("Top predictions:")
for i in top_indices:
    print(f"{classes[i]}: {clipwise_output[i]:.4f}")
