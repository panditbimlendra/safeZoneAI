# ============================================================================
# COMPLETE WORKING ABNORMAL SOUND DETECTION SYSTEM
# With simplified visualization options
# ============================================================================

import numpy as np
import pandas as pd
import os  # Import os at the top level
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavf
import librosa
import librosa.display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. AUDIO PROCESSING FUNCTIONS
# ============================================================================

def segment_signal(rawsignal, sample_rate, segment_duration):
    """Segment audio into chunks"""
    pts_segment = int(segment_duration * sample_rate)
    num_segments = len(rawsignal) // pts_segment
    segmented = np.zeros((num_segments, pts_segment))
    
    for i in range(num_segments):
        start = i * pts_segment
        end = start + pts_segment
        segmented[i, :] = rawsignal[start:end]
    
    return segmented

# ============================================================================
# 2. SIMPLIFIED VISUALIZATION FUNCTIONS
# ============================================================================

def plot_basic_waveform(audio, sr, title="Audio Waveform"):
    """Plot simple waveform"""
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(time, audio, color='blue', alpha=0.7, linewidth=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title="Spectrogram"):
    """Plot simple spectrogram"""
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

def plot_mfcc(audio, sr, title="MFCC Features"):
    """Plot MFCC features"""
    plt.figure(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='coolwarm')
    plt.colorbar()
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

def plot_detection_result(audio, sr, is_abnormal, sound_type, features):
    """Plot detection result summary"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    color = 'red' if is_abnormal else 'green'
    axes[0, 0].plot(time, audio, color=color, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title(f"Waveform: {sound_type}")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=axes[0, 1], cmap='hot')
    axes[0, 1].set_title("Spectrogram")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    
    # Key features
    key_features = ['Max RMS', 'Crest Factor', 'Zero-Crossing', 'Spectral Centroid']
    key_values = [
        min(features['max_rms'] * 10, 10),
        min(features['crest_factor'], 15),
        features['max_zcr'] * 20,
        features['centroid_mean'] / 500
    ]
    
    colors = ['red' if is_abnormal else 'green' for _ in key_features]
    axes[1, 0].bar(key_features, key_values, color=colors, alpha=0.7)
    axes[1, 0].set_title("Key Features")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Result
    axes[1, 1].axis('off')
    if is_abnormal:
        axes[1, 1].text(0.5, 0.7, '🚨 ABNORMAL', fontsize=24, fontweight='bold', 
                       color='red', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, f'Type: {sound_type}', fontsize=16, 
                       ha='center', va='center')
    else:
        axes[1, 1].text(0.5, 0.7, '✅ NORMAL', fontsize=24, fontweight='bold', 
                       color='green', ha='center', va='center')
    
    plt.suptitle(f"Sound Detection Result: {sound_type}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 3. QUICK DETECTION FUNCTION
# ============================================================================

def quick_detect_abnormal_sound(audio_file, plot_level=1):
    """
    Detect abnormal sounds with simple visualization options
    plot_level: 0=none, 1=basic, 2=detailed, 3=all
    """
    print(f"\n🔊 Analyzing: {audio_file}")
    
    try:
        # Load audio
        if isinstance(audio_file, str):
            audio, sr = librosa.load(audio_file, sr=16000)
        else:
            audio = audio_file
            sr = 16000
        
        # Extract key features
        features = {}
        
        # Energy features
        rms = librosa.feature.rms(y=audio)[0]
        features['max_rms'] = np.max(rms)
        features['mean_rms'] = np.mean(rms)
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['max_zcr'] = np.max(zcr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['centroid_mean'] = np.mean(spectral_centroid)
        
        # Temporal features
        features['crest_factor'] = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        
        # Rule-based detection
        is_abnormal = False
        reasons = []
        
        if features['max_rms'] > 0.25:
            is_abnormal = True
            reasons.append(f"Loud (RMS={features['max_rms']:.3f})")
        
        if features['crest_factor'] > 7:
            is_abnormal = True
            reasons.append(f"Sharp (crest={features['crest_factor']:.1f})")
        
        if features['max_zcr'] > 0.3:
            is_abnormal = True
            reasons.append(f"Abrupt (ZCR={features['max_zcr']:.3f})")
        
        if features['centroid_mean'] > 2500:
            is_abnormal = True
            reasons.append(f"High freq ({features['centroid_mean']:.0f} Hz)")
        
        # Classify type
        sound_type = "normal"
        if is_abnormal:
            if features['max_rms'] > 0.35 and features['crest_factor'] > 9:
                sound_type = "explosion/crash"
            elif features['centroid_mean'] > 3500 and features['max_zcr'] > 0.4:
                sound_type = "gunshot/glass"
            elif features['max_rms'] > 0.3:
                sound_type = "scream/alarm"
            elif features['crest_factor'] > 8:
                sound_type = "impact/bump"
            else:
                sound_type = "abnormal"
        
        # Display results
        print(f"\n📊 Features:")
        for key, value in features.items():
            print(f"  {key:20}: {value:8.4f}")
        
        print(f"\n🔍 Result:")
        if is_abnormal:
            print(f"  🚨 ABNORMAL: {sound_type}")
            if reasons:
                print(f"  Reasons: {', '.join(reasons)}")
        else:
            print(f"  ✅ Normal sound")
        
        # Visualization based on plot_level
        if plot_level >= 1 and plot_level <= 3:
            if plot_level == 1:
                plot_detection_result(audio, sr, is_abnormal, sound_type, features)
            elif plot_level == 2:
                plot_basic_waveform(audio, sr, f"Audio: {sound_type}")
                plot_spectrogram(audio, sr, f"Spectrogram: {sound_type}")
            elif plot_level == 3:
                plot_basic_waveform(audio, sr, f"Audio: {sound_type}")
                plot_spectrogram(audio, sr, f"Spectrogram: {sound_type}")
                plot_mfcc(audio, sr, f"MFCC: {sound_type}")
        
        return {
            'is_abnormal': is_abnormal,
            'sound_type': sound_type,
            'features': features,
            'reasons': reasons,
            'audio': audio,
            'sample_rate': sr
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 4. ABNORMAL SOUND DETECTOR CLASS
# ============================================================================

class AbnormalSoundDetector:
    """Main detector class"""
    
    def __init__(self):
        self.model = None
        self.feature_history = []
        
    def load_and_process_audio(self, audio_file, segment_duration=0.4):
        """Load and segment audio"""
        sample_rate, rawsignal = wavf.read(audio_file)
        
        # Convert to mono if stereo
        if len(rawsignal.shape) > 1:
            rawsignal = rawsignal.mean(axis=1)
        
        # Segment the signal
        segmented = segment_signal(rawsignal, sample_rate, segment_duration)
        
        return segmented, sample_rate, segmented.shape[0]
    
    def extract_features(self, segmented_audio, sample_rate):
        """Extract features from segmented audio"""
        n_segments = segmented_audio.shape[0]
        features_list = []
        
        print(f"Extracting features from {n_segments} segments...")
        
        for i in range(n_segments):
            if i % 100 == 0 and n_segments > 100:
                print(f"  Segment {i}/{n_segments}")
            
            audio_segment = segmented_audio[i, :]
            
            # Extract features
            features = []
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
            features.extend(np.mean(mfccs, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)[0]
            
            features.append(np.mean(spectral_centroid))
            features.append(np.mean(spectral_bandwidth))
            
            # Energy features
            rms = librosa.feature.rms(y=audio_segment)[0]
            features.append(np.max(rms))
            features.append(np.mean(rms))
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0]
            features.append(np.mean(zcr))
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def train_model(self, audio_file, labels, plot_results=False):
        """Train the detector"""
        print("Training detector...")
        
        # Load and process audio
        segmented_audio, sample_rate, _ = self.load_and_process_audio(audio_file)
        
        # Extract features
        X = self.extract_features(segmented_audio, sample_rate)
        
        # Adjust labels to match features
        if len(labels) > X.shape[0]:
            labels = labels[:X.shape[0]]
        elif len(labels) < X.shape[0]:
            X = X[:len(labels), :]
        
        y = labels.flatten()
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=10
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training complete!")
        print(f"   Accuracy: {accuracy:.2%}")
        
        # Plot results if requested
        if plot_results:
            self.plot_training_results(y_test, y_pred)
        
        return accuracy
    
    def plot_training_results(self, y_test, y_pred):
        """Plot simple training results"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Accuracy bar
        accuracy = accuracy_score(y_test, y_pred)
        axes[1].bar(['Accuracy'], [accuracy], color='green')
        axes[1].set_ylim([0, 1])
        axes[1].set_title(f'Model Accuracy: {accuracy:.2%}')
        axes[1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.show()
    
    def predict_file(self, audio_file, plot_level=1):
        """Predict abnormal sounds in a file"""
        print(f"\n🔍 Analyzing {audio_file}...")
        
        # Quick detection
        result = quick_detect_abnormal_sound(audio_file, plot_level=plot_level)
        
        # If model exists, do detailed analysis
        if self.model is not None and result and result['is_abnormal']:
            try:
                # Load and extract features
                segmented_audio, sample_rate, _ = self.load_and_process_audio(audio_file)
                features = self.extract_features(segmented_audio, sample_rate)
                
                # Predict segments
                predictions = self.model.predict(features)
                abnormal_segments = np.sum(predictions == 1)
                total_segments = len(predictions)
                
                print(f"  Model analysis: {abnormal_segments}/{total_segments} segments abnormal")
            except Exception as e:
                print(f"  Model analysis failed: {str(e)}")
        
        return result

# ============================================================================
# 5. TESTING AND DEMONSTRATION
# ============================================================================

def create_test_sounds():
    """Create test sounds for demonstration"""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Normal sound
    normal = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    
    # Abnormal sound
    abnormal = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
    abnormal[int(sr*0.5):int(sr*0.5)+100] = 0.8  # Impulse
    abnormal[int(sr*1.2):int(sr*1.2)+200] = 0.6  # Another impulse
    
    return normal, abnormal, sr

def simple_demo():
    """Simple demonstration of the system"""
    print("\n" + "="*60)
    print("ABNORMAL SOUND DETECTION DEMO")
    print("="*60)
    
    # Create test sounds
    print("\nCreating test sounds...")
    normal_sound, abnormal_sound, sr = create_test_sounds()
    
    # Save test sounds
    import soundfile as sf
    sf.write('demo_normal.wav', normal_sound, sr)
    sf.write('demo_abnormal.wav', abnormal_sound, sr)
    
    # Create detector
    detector = AbnormalSoundDetector()
    
    # Ask for visualization level
    print("\n📊 Visualization Options:")
    print("  0: No plots")
    print("  1: Result summary (recommended)")
    print("  2: Waveform + spectrogram")
    print("  3: All plots")
    
    try:
        plot_level = int(input("\nSelect visualization level (0-3): "))
    except:
        plot_level = 1
    
    # Test normal sound
    print(f"\n{'='*40}")
    print("TEST 1: NORMAL SOUND")
    print('='*40)
    result1 = detector.predict_file('demo_normal.wav', plot_level=plot_level)
    
    # Test abnormal sound
    print(f"\n{'='*40}")
    print("TEST 2: ABNORMAL SOUND")
    print('='*40)
    result2 = detector.predict_file('demo_abnormal.wav', plot_level=plot_level)
    
    # Summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print('='*40)
    if result1:
        print(f"Normal sound:   {'ABNORMAL' if result1['is_abnormal'] else 'NORMAL'}")
    else:
        print("Normal sound:   ERROR")
    
    if result2:
        print(f"Abnormal sound: {'ABNORMAL' if result2['is_abnormal'] else 'NORMAL'}")
    else:
        print("Abnormal sound: ERROR")
    
    # Cleanup
    for file in ['demo_normal.wav', 'demo_abnormal.wav']:
        if os.path.exists(file):
            os.remove(file)
    
    return detector

def batch_test_directory(directory_path, plot_level=0):
    """Test all audio files in a directory"""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    import glob
    audio_files = glob.glob(os.path.join(directory_path, "*.wav")) + \
                  glob.glob(os.path.join(directory_path, "*.mp3")) + \
                  glob.glob(os.path.join(directory_path, "*.flac"))
    
    if not audio_files:
        print("No audio files found in directory")
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    
    detector = AbnormalSoundDetector()
    results = []
    
    for i, audio_file in enumerate(audio_files[:10]):  # Limit to 10 files
        print(f"\n[{i+1}/{min(10, len(audio_files))}] {os.path.basename(audio_file)}")
        
        try:
            result = detector.predict_file(audio_file, plot_level=plot_level)
            if result:
                results.append({
                    'file': os.path.basename(audio_file),
                    'abnormal': result['is_abnormal'],
                    'type': result['sound_type']
                })
            else:
                results.append({
                    'file': os.path.basename(audio_file),
                    'abnormal': False,
                    'type': 'error'
                })
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'file': os.path.basename(audio_file),
                'abnormal': False,
                'type': 'error'
            })
    
    # Summary
    abnormal_count = sum(1 for r in results if r['abnormal'])
    print(f"\n{'='*40}")
    print("BATCH TEST SUMMARY")
    print('='*40)
    print(f"Total files tested: {len(results)}")
    print(f"Abnormal sounds: {abnormal_count}")
    print(f"Normal sounds: {len(results) - abnormal_count}")
    
    return results

# ============================================================================
# 6. MAIN INTERFACE
# ============================================================================

def main():
    """Main interface"""
    print("\n" + "="*60)
    print("ABNORMAL SOUND DETECTION SYSTEM")
    print("="*60)
    
    print("\nOptions:")
    print("1. Quick test a sound file")
    print("2. Simple demo with test sounds")
    print("3. Test all files in a directory")
    print("4. Train on your data")
    print("5. Exit")
    
    try:
        choice = int(input("\nEnter choice (1-5): "))
    except:
        choice = 1
    
    if choice == 1:
        # Quick test a file
        file_path = input("Enter path to audio file: ").strip()
        
        # Use the os module imported at the top
        if file_path and os.path.exists(file_path):
            print("\nVisualization Options:")
            print("  0: No plots")
            print("  1: Result summary")
            print("  2: Waveform + spectrogram")
            print("  3: All plots")
            
            try:
                plot_level = int(input("\nSelect visualization level (0-3): "))
            except:
                plot_level = 1
            
            detector = AbnormalSoundDetector()
            result = detector.predict_file(file_path, plot_level=plot_level)
            
            if result:
                print(f"\n{'='*40}")
                print("FINAL RESULT:")
                if result['is_abnormal']:
                    print(f"🚨 ABNORMAL: {result['sound_type']}")
                    if result['reasons']:
                        print(f"Reasons: {', '.join(result['reasons'])}")
                else:
                    print(f"✅ NORMAL SOUND")
        else:
            print(f"File not found: {file_path}")
            print("Running demo instead...")
            simple_demo()
    
    elif choice == 2:
        # Simple demo
        simple_demo()
    
    elif choice == 3:
        # Batch test directory
        dir_path = input("Enter directory path: ").strip()
        if dir_path:
            batch_test_directory(dir_path, plot_level=0)
        else:
            print("No directory specified")
    
    elif choice == 4:
        # Train on data
        print("\nTraining requires labeled data.")
        print("This option uses your original code structure.")
        
        # You would integrate your original training code here
        detector = AbnormalSoundDetector()
        
        # Example with dummy data
        print("\nCreating example training data...")
        
        # Create example audio file
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(len(t))
        
        # Add some abnormalities
        audio[int(sr*1.0):int(sr*1.0)+50] = 0.6
        audio[int(sr*3.0):int(sr*3.0)+100] = 0.7
        
        # Save audio
        import soundfile as sf
        sf.write('training_example.wav', audio, sr)
        
        # Create labels (0=normal, 1=abnormal)
        segment_duration = 0.4
        pts_segment = int(segment_duration * sr)
        num_segments = len(audio) // pts_segment
        labels = np.zeros(num_segments)
        
        # Mark abnormal segments
        labels[int(1.0/segment_duration):int(1.2/segment_duration)] = 1  # First abnormality
        labels[int(3.0/segment_duration):int(3.2/segment_duration)] = 1  # Second abnormality
        
        # Train
        accuracy = detector.train_model('training_example.wav', labels, plot_results=True)
        print(f"\nModel trained with accuracy: {accuracy:.2%}")
        
        # Cleanup
        if os.path.exists('training_example.wav'):
            os.remove('training_example.wav')
    
    elif choice == 5:
        print("\nExiting...")
        return
    
    else:
        # Default to demo
        simple_demo()

# ============================================================================
# 7. COMMAND LINE SUPPORT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        audio_file = sys.argv[1]
        
        if os.path.exists(audio_file):
            plot_level = 1  # Default to summary plots
            if len(sys.argv) > 2:
                try:
                    plot_level = int(sys.argv[2])
                except:
                    pass
            
            detector = AbnormalSoundDetector()
            result = detector.predict_file(audio_file, plot_level=plot_level)
            
            if result:
                print(f"\n{'='*40}")
                print("RESULT:")
                if result['is_abnormal']:
                    print(f"🚨 ABNORMAL: {result['sound_type']}")
                    if result['reasons']:
                        print(f"Reasons: {', '.join(result['reasons'])}")
                else:
                    print(f"✅ NORMAL")
        else:
            print(f"File not found: {audio_file}")
    else:
        # Interactive mode
        main()