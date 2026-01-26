# test_sound.py - SIMPLE ABNORMAL SOUND DETECTOR
import numpy as np
import librosa
import sys
import os

print("=" * 50)
print("ABNORMAL SOUND DETECTOR")
print("=" * 50)

# Get file path from command line or input
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = input("\n📁 Enter FULL path to audio file: ").strip()

# Remove quotes if user added them
file_path = file_path.strip('"').strip("'")

print(f"\nChecking: {file_path}")

if not os.path.exists(file_path):
    print(f"❌ ERROR: File not found!")
    print(f"Make sure the path is correct.")
    print(f"Try: C:/Users/Bimlendra/Downloads/car-crash-382137.mp3")
    exit()

print(f"✅ File found!")

try:
    # Load the audio
    audio, sr = librosa.load(file_path, sr=16000)
    print(f"Loaded: {len(audio)} samples, {len(audio)/sr:.2f} seconds")
    
    # Simple features
    rms = np.max(librosa.feature.rms(y=audio)[0])
    zcr = np.max(librosa.feature.zero_crossing_rate(y=audio)[0])
    
    print(f"\n📊 Analysis:")
    print(f"  Max Loudness: {rms:.3f}")
    print(f"  Abruptness: {zcr:.3f}")
    
    # Detection
    if rms > 0.3:
        print(f"\n🚨 CAR CRASH DETECTED!")
        print(f"   (Very loud sound: {rms:.3f})")
    elif zcr > 0.4:
        print(f"\n🚨 ABNRUPT SOUND DETECTED!")
        print(f"   (Sharp/abrupt sound)")
    else:
        print(f"\n✅ Normal sound")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Try converting MP3 to WAV first!")