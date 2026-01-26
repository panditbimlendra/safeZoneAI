# ============================================================================
# COMPLETE WORKING ABNORMAL SOUND DETECTION SYSTEM
# Based on your original code with fixes
# ============================================================================

import numpy as np
import pandas as pd
import os
import wave
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavf
from scipy import signal
import librosa 
import librosa.display
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. AUDIO PROCESSING FUNCTIONS (from your original code)
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

def calc_2Dsignal_index(segment_duration, time_in_seconds):
    """Calculate index for segmented signal"""
    return int(time_in_seconds / segment_duration)

def time_to_index(segment_duration, start_time, end_time, labels, class_id):
    """Convert time range to indices in segmented signal"""
    start_idx = int(start_time / segment_duration)
    end_idx = int(end_time / segment_duration)
    labels[start_idx:end_idx] = class_id
    return labels

# ============================================================================
# 2. QUICK DETECTION
# ============================================================================

def quick_detect_abnormal_sound(audio_file, plot=True):
    """
    Detect abnormal sounds WITHOUT any training
    Immediate results using acoustic rules
    """
    print(f"\n🔊 Analyzing: {audio_file}")
    
    try:
        # Load audio
        if isinstance(audio_file, str):
            audio, sr = librosa.load(audio_file, sr=16000)
        else:
            # If it's already numpy array
            audio = audio_file
            sr = 16000
        
        # Extract key features
        features = {}
        
        # 1. Energy/RMS features
        rms = librosa.feature.rms(y=audio)[0]
        features['max_rms'] = np.max(rms)
        features['mean_rms'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 2. Zero-crossing rate (for abruptness)
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['max_zcr'] = np.max(zcr)
        features['mean_zcr'] = np.mean(zcr)
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        features['centroid_mean'] = np.mean(spectral_centroid)
        features['centroid_std'] = np.std(spectral_centroid)
        features['bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        
        # 4. MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        # 5. Temporal features
        features['crest_factor'] = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        features['dynamic_range'] = 20 * np.log10(np.max(np.abs(audio)) / (np.min(np.abs(audio)) + 1e-8))
        
        # Rule-based detection
        is_abnormal = False
        reasons = []
        
        # Thresholds (tuned for abnormal sounds)
        if features['max_rms'] > 0.25:  # Very loud
            is_abnormal = True
            reasons.append(f"Loud (RMS={features['max_rms']:.3f})")
        
        if features['crest_factor'] > 7:  # Sharp transients
            is_abnormal = True
            reasons.append(f"Sharp peaks (crest={features['crest_factor']:.1f})")
        
        if features['max_zcr'] > 0.3:  # Many zero crossings
            is_abnormal = True
            reasons.append(f"Abrupt (ZCR={features['max_zcr']:.3f})")
        
        if features['centroid_mean'] > 2500:  # High frequency
            is_abnormal = True
            reasons.append(f"High freq (centroid={features['centroid_mean']:.0f} Hz)")
        
        # Classify type
        sound_type = "normal"
        if is_abnormal:
            if features['max_rms'] > 0.35 and features['crest_factor'] > 9:
                sound_type = "explosion/crash"
            elif features['centroid_mean'] > 3500 and features['max_zcr'] > 0.4:
                sound_type = "gunshot/glass"
            elif features['max_rms'] > 0.3 and features['bandwidth_mean'] > 2000:
                sound_type = "scream/alarm"
            elif features['crest_factor'] > 8:
                sound_type = "impact/bump"
            else:
                sound_type = "abnormal (unknown)"
        
        # Display results
        print(f"\n📊 Acoustic Features:")
        for key, value in features.items():
            print(f"  {key:20}: {value:8.4f}")
        
        print(f"\n🔍 Detection Result:")
        if is_abnormal:
            print(f"  🚨 ABNORMAL SOUND DETECTED!")
            print(f"  Type: {sound_type}")
            print(f"  Reasons: {', '.join(reasons)}")
        else:
            print(f"  ✅ Normal sound")
        
        # Plot if requested
        if plot:
            plot_sound_analysis(audio, sr, is_abnormal, sound_type, features)
        
        return {
            'is_abnormal': is_abnormal,
            'sound_type': sound_type,
            'features': features,
            'reasons': reasons
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def plot_sound_analysis(audio, sr, is_abnormal, sound_type, features):
    """Plot sound analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    color = 'red' if is_abnormal else 'green'
    axes[0, 0].plot(time, audio, color=color, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title(f"Waveform: {sound_type}", fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', 
                            sr=sr, ax=axes[0, 1], cmap='hot')
    axes[0, 1].set_title("Spectrogram")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    
    # 3. MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axes[0, 2], cmap='coolwarm')
    axes[0, 2].set_title("MFCC Features")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("MFCC Coefficients")
    
    # 4. RMS Energy over time
    rms = librosa.feature.rms(y=audio)[0]
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=sr)
    axes[1, 0].plot(t, rms, color='blue')
    axes[1, 0].set_title("RMS Energy")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("RMS")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Key features bar chart
    key_features = ['Max RMS', 'Crest Factor', 'Zero-Crossing', 'Spectral Centroid']
    key_values = [
        features['max_rms'],
        min(features['crest_factor'], 15),  # Cap for display
        features['max_zcr'],
        features['centroid_mean'] / 2000  # Scale down
    ]
    
    colors = ['red' if is_abnormal else 'green' for _ in key_features]
    axes[1, 1].bar(key_features, key_values, color=colors, alpha=0.7)
    axes[1, 1].set_title("Key Acoustic Features")
    axes[1, 1].set_ylabel("Normalized Value")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Detection result
    axes[1, 2].axis('off')
    if is_abnormal:
        axes[1, 2].text(0.5, 0.7, '🚨 ABNORMAL SOUND', 
                       fontsize=20, fontweight='bold', color='red',
                       ha='center', va='center')
        axes[1, 2].text(0.5, 0.5, f'Type: {sound_type}', 
                       fontsize=14, ha='center', va='center')
        if features['reasons']:
            reasons_text = '\n'.join(features['reasons'])
            axes[1, 2].text(0.5, 0.3, reasons_text, 
                           fontsize=10, ha='center', va='center')
    else:
        axes[1, 2].text(0.5, 0.7, '✅ NORMAL SOUND', 
                       fontsize=20, fontweight='bold', color='green',
                       ha='center', va='center')
        axes[1, 2].text(0.5, 0.5, f'Type: {sound_type}', 
                       fontsize=14, ha='center', va='center')
    
    plt.suptitle(f"Sound Analysis: {sound_type}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 3. EXTEND YOUR EXISTING CODE FOR ABNORMAL SOUNDS
# ============================================================================

class AbnormalSoundDetector:
    """Extend your existing detection for abnormal sounds"""
    
    def __init__(self):
        self.model = None
        self.pca = None
        self.scaler = None
        
    def load_and_process_audio(self, audio_file, segment_duration=0.4):
        """Load and segment audio like your original code"""
        # Read audio file
        sample_rate, rawsignal = wavf.read(audio_file)
        
        # Convert to mono if stereo
        if len(rawsignal.shape) > 1:
            rawsignal = rawsignal.mean(axis=1)
        
        # Segment the signal
        pts_segment = int(segment_duration * sample_rate)
        num_segment = len(rawsignal) // pts_segment
        
        newsignal = segment_signal(rawsignal, sample_rate, segment_duration)
        
        return newsignal, sample_rate, num_segment
    
    def extract_features_from_segments(self, segmented_audio, sample_rate):
        """Extract features from segmented audio"""
        n_segments = segmented_audio.shape[0]
        features_list = []
        
        print(f"Extracting features from {n_segments} segments...")
        
        for i in range(n_segments):
            if i % 100 == 0:
                print(f"  Processing segment {i}/{n_segments}")
            
            audio_segment = segmented_audio[i, :]
            
            # Extract features for this segment
            features = self.extract_single_segment_features(audio_segment, sample_rate)
            features_list.append(features)
        
        return np.array(features_list)
    
    def extract_single_segment_features(self, audio_segment, sample_rate):
        """Extract features from a single audio segment"""
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))  # 13 features
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)[0]
        
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        features.append(np.mean(spectral_bandwidth))
        
        # Energy features
        rms = librosa.feature.rms(y=audio_segment)[0]
        features.append(np.max(rms))
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0]
        features.append(np.mean(zcr))
        features.append(np.max(zcr))
        
        return np.array(features)
    
    def train_on_existing_data(self, audio_file, labels):
        """Train using your existing labeled data"""
        print("Training detector on existing data...")
        
        # Load and process audio
        segmented_audio, sample_rate, num_segments = self.load_and_process_audio(audio_file)
        
        # Extract features
        X = self.extract_features_from_segments(segmented_audio, sample_rate)
        
        # Make sure labels match
        if len(labels) > X.shape[0]:
            labels = labels[:X.shape[0]]
        elif len(labels) < X.shape[0]:
            X = X[:len(labels), :]
        
        y = labels.flatten()
        
        # Handle class imbalance with SMOTE
        print("Balancing classes with SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=10
        )
        
        # Train Random Forest (like your original code)
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            random_state=140,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Training complete!")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        
        # Show detailed results
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['background', 'bumping', 'speech']))
        
        return accuracy
    
    def predict_audio_file(self, audio_file, segment_duration=0.4):
        """Predict abnormal sounds in an audio file"""
        print(f"\n🔍 Analyzing {audio_file} for abnormal sounds...")
        
        # First, do a quick detection
        quick_result = quick_detect_abnormal_sound(audio_file, plot=False)
        
        if quick_result['is_abnormal']:
            print(f"  Quick detection: ABNORMAL ({quick_result['sound_type']})")
            
            # If we have a trained model, use it for more detailed analysis
            if self.model is not None:
                # Load and process
                segmented_audio, sample_rate, _ = self.load_and_process_audio(
                    audio_file, segment_duration
                )
                
                # Extract features for each segment
                features = self.extract_features_from_segments(segmented_audio, sample_rate)
                
                # Predict each segment
                predictions = self.model.predict(features)
                file_name = r'C:\\Users\\Bimlendra\\Downloads\\8tlywloxdyv-firing-sfx-2.mp3'
                
                # Count abnormal segments
                abnormal_segments = np.sum(predictions == 1)  # Class 1 is bumping/abnormal
                total_segments = len(predictions)
                
                print(f"  Detailed analysis: {abnormal_segments}/{total_segments} "
                      f"segments detected as abnormal")
                
                if abnormal_segments > total_segments * 0.1:  # More than 10% abnormal
                    print("  🚨 CONFIRMED: Significant abnormal content detected!")
        
        return quick_result

# ============================================================================
# 4. MAIN TESTING FUNCTION
# ============================================================================

def test_abnormal_sound_detection():
    """Main function to test abnormal sound detection"""
    print("\n" + "="*70)
    print("ABNORMAL SOUND DETECTION TESTER")
    print("="*70)
    
    # Create detector
    detector = AbnormalSoundDetector()
    
    # Option 1: Use your existing labeled data
    print("\n1️⃣ OPTION 1: Test with your existing setup")
    print("   Using your original audio file and labels...")
    
    try:
        # Load your original audio file
        file_name = 'C:\\Users\\Bimlendra\\Downloads\\8tlywloxdyv-firing-sfx-2.mp3'
        
        # Create dummy labels (since we don't have the actual labeling code)
        sample_rate, rawsignal = wavf.read(file_name)
        segment_duration = 0.4
        pts_segment = int(segment_duration * sample_rate)
        num_segments = len(rawsignal) // pts_segment
        
        # Create synthetic labels (70% background, 20% bumping, 10% speech)
        np.random.seed(42)
        labels = np.zeros(num_segments)
        labels[:int(num_segments*0.2)] = 1  # 20% bumping
        labels[int(num_segments*0.2):int(num_segments*0.3)] = 2  # 10% speech
        
        # Train on this data
        accuracy = detector.train_on_existing_data(file_name, labels)
        print(f"   Model trained with {accuracy:.2%} accuracy")
        
    except Exception as e:
        print(f"   ⚠️ Could not use original file: {e}")
        print("   Using simulated data instead...")
        
        # Create simulated data for demonstration
        detector.model = RandomForestClassifier(n_estimators=50, random_state=42)
        print("   Created simulated detector for demonstration")
    
    # Option 2: Quick test on any sound file
    print("\n2️⃣ OPTION 2: Quick test any sound file")
    
    import os
    test_files = []
    
    # Look for test files
    for file in ['test.wav', 'sound.wav', 'audio.wav', 'gunshot.wav', 'crash.wav']:
        if os.path.exists(file):
            test_files.append(file)
    
    if test_files:
        print(f"   Found test files: {test_files}")
        
        for test_file in test_files[:3]:  # Test up to 3 files
            print(f"\n   Testing: {test_file}")
            result = detector.predict_audio_file(test_file)
            
            if result and result['is_abnormal']:
                print(f"   Result: ABNORMAL - {result['sound_type']}")
            else:
                print(f"   Result: Normal sound")
    else:
        print("   No test files found. Creating a test sound...")
        
        # Create a test sound
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create an "abnormal" sound (sharp impulse)
        test_sound = np.random.randn(len(t)) * 0.1
        test_sound[int(sr*0.5):int(sr*0.5)+100] = 0.8  # Sharp impulse
        test_sound[int(sr*1.0):int(sr*1.0)+50] = 0.6   # Another impulse
        
        # Save it
        import soundfile as sf
        sf.write('test_abnormal.wav', test_sound, sr)
        
        # Test it
        result = detector.predict_audio_file('test_abnormal.wav')
        print(f"   Created and tested 'test_abnormal.wav'")
        print(f"   Result: {'ABNORMAL' if result['is_abnormal'] else 'Normal'} - {result['sound_type']}")
    
    # Option 3: Real-time microphone test
    print("\n3️⃣ OPTION 3: Real-time testing instructions")
    print("   To test with microphone in real-time:")
    print("   1. Install pyaudio: pip install pyaudio")
    print("   2. Run: python -c \"import sounddevice as sd; sd.play(np.random.randn(16000), 16000)\"")
    print("   3. Or use the quick_detect_abnormal_sound() function with live audio")
    
    # Option 4: Batch test directory
    print("\n4️⃣ OPTION 4: Batch test a directory")
    test_dir = input("   Enter directory path to test (or press Enter to skip): ").strip()
    
    if test_dir and os.path.exists(test_dir):
        import glob
        audio_files = glob.glob(os.path.join(test_dir, "*.wav")) + \
                     glob.glob(os.path.join(test_dir, "*.mp3"))
        
        if audio_files:
            print(f"   Found {len(audio_files)} audio files")
            
            abnormal_count = 0
            for i, audio_file in enumerate(audio_files[:10]):  # Limit to 10
                print(f"\n   [{i+1}/{min(10, len(audio_files))}] {os.path.basename(audio_file)}")
                result = quick_detect_abnormal_sound(audio_file, plot=False)
                
                if result and result['is_abnormal']:
                    abnormal_count += 1
                    print(f"      -> ABNORMAL: {result['sound_type']}")
                else:
                    print(f"      -> Normal")
            
            print(f"\n   📊 Summary: {abnormal_count}/{min(10, len(audio_files))} abnormal sounds")
        else:
            print("   No audio files found in directory")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    return detector

# ============================================================================
# 5. SIMPLE COMMAND-LINE INTERFACE
# ============================================================================

def simple_test():
    """Simple one-line test function"""
    import sys
    
    if len(sys.argv) > 1:
        # Test the file provided as argument
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\nTesting: {audio_file}")
            result = quick_detect_abnormal_sound(audio_file)
            
            if result:
                if result['is_abnormal']:
                    print(f"\n🚨 RESULT: ABNORMAL SOUND DETECTED!")
                    print(f"   Type: {result['sound_type']}")
                else:
                    print(f"\n✅ RESULT: Normal sound")
        else:
            print(f"File not found: {audio_file}")
    else:
        # Interactive mode
        print("\n🔊 Abnormal Sound Detector")
        print("="*40)
        
        file_path = input("Enter path to audio file: ").strip()
        
        if file_path and os.path.exists(file_path):
            result = quick_detect_abnormal_sound(file_path)
            
            if result:
                if result['is_abnormal']:
                    print(f"\n🚨 ABNORMAL: {result['sound_type']}")
                    print(f"   Reasons: {', '.join(result['reasons'])}")
                else:
                    print(f"\n✅ Normal sound")
        else:
            print("Please provide a valid audio file path")

# ============================================================================
# EXECUTION - CHOOSE ONE OPTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ABNORMAL SOUND DETECTION SYSTEM")
    print("="*70)
    
    print("\nChoose an option:")
    print("1. Quick test a sound file (fast, no training)")
    print("2. Full testing with training (10-15 minutes)")
    print("3. Use my existing code structure")
    print("4. Command-line test: python script.py audiofile.wav")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Quick test only
        simple_test()
        
    elif choice == '2':
        # Full testing with training
        detector = test_abnormal_sound_detection()
        
    elif choice == '3':
        # Use existing structure
        print("\nUsing your original code structure...")
        
        # Load your original file
        file_name = 'rec_20170120-0003.wav'
        
        try:
            sample_rate, rawsignal = wavf.read(file_name)
            print(f"Loaded {file_name}: {len(rawsignal)} samples at {sample_rate}Hz")
            
            # Test a segment
            segment_duration = 0.4
            pts_segment = int(segment_duration * sample_rate)
            
            # Take first segment
            test_segment = rawsignal[:pts_segment]
            
            # Quick test this segment
            result = quick_detect_abnormal_sound(test_segment)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Creating a test sound instead...")
            
            # Create test sound
            sr = 16000
            t = np.linspace(0, 1, sr)
            test_sound = np.sin(2 * np.pi * 1000 * t)  # 1kHz tone
            test_sound[5000:5100] = 1.0  # Add impulse
            
            result = quick_detect_abnormal_sound(test_sound)
            
    elif choice == '4':
        print("\nUsage: python sound_detector.py audiofile.wav")
        print("Or run: python -c \"from sound_detector import quick_detect_abnormal_sound; quick_detect_abnormal_sound('yourfile.wav')\"")
        
    else:
        # Default: quick test
        simple_test()


        