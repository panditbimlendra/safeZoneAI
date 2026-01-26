# ============================================================================
# ENHANCED ACCURATE ABNORMAL SOUND DETECTION SYSTEM
# ============================================================================

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wavf
from scipy import signal, stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ENHANCED FEATURE EXTRACTION
# ============================================================================

class AdvancedAudioFeatures:
    """Extract advanced acoustic features for better accuracy"""
    
    @staticmethod
    def extract_all_features(audio, sr=16000):
        """Extract comprehensive feature set"""
        features = {}
        
        # ========== TIME DOMAIN FEATURES ==========
        # 1. Energy features
        rms = librosa.feature.rms(y=audio)[0]
        features['max_rms'] = np.max(rms)
        features['mean_rms'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_skew'] = stats.skew(rms)
        features['rms_kurtosis'] = stats.kurtosis(rms)
        
        # 2. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['max_zcr'] = np.max(zcr)
        features['mean_zcr'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 3. Amplitude statistics
        features['amplitude_max'] = np.max(np.abs(audio))
        features['amplitude_mean'] = np.mean(np.abs(audio))
        features['amplitude_std'] = np.std(audio)
        features['amplitude_skew'] = stats.skew(audio)
        features['amplitude_kurtosis'] = stats.kurtosis(audio)
        
        # 4. Temporal features
        envelope = np.abs(signal.hilbert(audio))
        features['attack_time'] = np.argmax(envelope) / sr
        features['decay_time'] = len(audio) / sr - features['attack_time']
        features['crest_factor'] = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        features['dynamic_range'] = 20 * np.log10(np.max(np.abs(audio)) / (np.min(np.abs(audio[np.nonzero(audio)])) + 1e-8))
        
        # ========== FREQUENCY DOMAIN FEATURES ==========
        # 5. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        features['centroid_mean'] = np.mean(spectral_centroid)
        features['centroid_std'] = np.std(spectral_centroid)
        features['bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        
        # 6. Spectral flux (change in spectrum)
        spec = np.abs(librosa.stft(audio))
        spectral_flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features['spectral_flux_max'] = np.max(spectral_flux)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        
        # 7. Spectral flatness (noise vs tone)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['flatness_mean'] = np.mean(spectral_flatness)
        features['flatness_std'] = np.std(spectral_flatness)
        
        # ========== MFCC & CEPSTRAL FEATURES ==========
        # 8. Enhanced MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        for i in range(13):  # First 13 MFCCs are most important
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 9. MFCC deltas (temporal changes)
        mfcc_delta = librosa.feature.delta(mfccs)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        
        # 10. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # ========== SPECIALIZED ABNORMAL SOUND FEATURES ==========
        # 11. Impulse detection
        impulse_ratio = np.sum(np.abs(audio) > 3 * np.std(audio)) / len(audio)
        features['impulse_ratio'] = impulse_ratio
        
        # 12. Silence ratio
        silence_threshold = 0.01 * np.max(np.abs(audio))
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        features['silence_ratio'] = silence_ratio
        
        # 13. Peak frequency detection
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        peak_freq = freqs[np.argmax(fft)]
        features['peak_frequency'] = peak_freq
        
        # 14. Harmonic-to-noise ratio
        harmonic = librosa.effects.harmonic(audio)
        percussive = librosa.effects.percussive(audio)
        hnr = np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-8)
        features['harmonic_noise_ratio'] = hnr
        
        return features
    
    @staticmethod
    def extract_abnormal_specific_features(audio, sr):
        """Features specifically tuned for abnormal sounds"""
        features = {}
        
        # Gunshot/Explosion detection
        # 1. Short-term energy rise
        frame_length = 256
        hop_length = 128
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy_gradient = np.gradient(energy)
        features['energy_rise_max'] = np.max(energy_gradient[energy_gradient > 0])
        
        # 2. Transient detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_max'] = np.max(onset_env)
        features['onset_strength_mean'] = np.mean(onset_env)
        
        # 3. Frequency spread
        spectrogram = np.abs(librosa.stft(audio))
        freq_variance = np.var(spectrogram, axis=0)
        features['freq_variance_max'] = np.max(freq_variance)
        
        # 4. Temporal centroid
        temporal_centroid = np.sum(np.arange(len(audio)) * np.abs(audio)) / np.sum(np.abs(audio))
        features['temporal_centroid'] = temporal_centroid / sr
        
        return features

# ============================================================================
# 2. MACHINE LEARNING MODEL WITH ENSEMBLE
# ============================================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class EnhancedAbnormalSoundDetector:
    """Advanced detector with ensemble learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_extractor = AdvancedAudioFeatures()
        self.is_trained = False
        
    def extract_features_for_training(self, audio_paths, labels):
        """Extract features for multiple audio files"""
        features_list = []
        labels_list = []
        
        print("Extracting advanced features...")
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(audio_paths)} files")
            
            try:
                audio, sr = librosa.load(path, sr=16000, duration=2.0)
                
                # Extract standard features
                features = self.feature_extractor.extract_all_features(audio, sr)
                
                # Extract abnormal-specific features
                abnormal_features = self.feature_extractor.extract_abnormal_specific_features(audio, sr)
                features.update(abnormal_features)
                
                features_list.append(list(features.values()))
                labels_list.append(label)
                
            except Exception as e:
                print(f"  Skipping {path}: {e}")
                continue
        
        return np.array(features_list), np.array(labels_list)
    
    def train_ensemble(self, X, y):
        """Train ensemble of models for better accuracy"""
        print("\nTraining ensemble of models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest
        print("  1. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        self.models['rf'] = rf
        rf_score = rf.score(X_test_scaled, y_test)
        print(f"    Accuracy: {rf_score:.4f}")
        
        # 2. Gradient Boosting
        print("  2. Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)
        self.models['gb'] = gb
        gb_score = gb.score(X_test_scaled, y_test)
        print(f"    Accuracy: {gb_score:.4f}")
        
        # 3. SVM with RBF kernel
        print("  3. Training SVM...")
        svm = SVC(
            C=10,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        svm.fit(X_train_scaled, y_train)
        self.models['svm'] = svm
        svm_score = svm.score(X_test_scaled, y_test)
        print(f"    Accuracy: {svm_score:.4f}")
        
        # 4. Neural Network
        print("  4. Training Neural Network...")
        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        nn.fit(X_train_scaled, y_train)
        self.models['nn'] = nn
        nn_score = nn.score(X_test_scaled, y_test)
        print(f"    Accuracy: {nn_score:.4f}")
        
        # 5. Ensemble voting
        print("  5. Creating ensemble voting classifier...")
        estimators = [
            ('rf', rf),
            ('gb', gb),
            ('svm', svm),
            ('nn', nn)
        ]
        self.models['ensemble'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1.2, 1.0, 1.1, 0.9]  # Weight RF more heavily
        )
        self.models['ensemble'].fit(X_train_scaled, y_train)
        ensemble_score = self.models['ensemble'].score(X_test_scaled, y_test)
        print(f"    Ensemble Accuracy: {ensemble_score:.4f}")
        
        # Evaluate
        y_pred = self.models['ensemble'].predict(X_test_scaled)
        print(f"\n📊 ENSEMBLE PERFORMANCE:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return ensemble_score
    
    def predict(self, audio_path):
        """Predict using ensemble model"""
        if not self.is_trained:
            print("⚠️ Model not trained. Loading pre-trained model...")
            if not self.load_model():
                print("❌ No model available. Please train first.")
                return None
        
        try:
            # Load and extract features
            audio, sr = librosa.load(audio_path, sr=16000, duration=2.0)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(audio, sr)
            abnormal_features = self.feature_extractor.extract_abnormal_specific_features(audio, sr)
            features.update(abnormal_features)
            
            # Prepare for prediction
            X = np.array(list(features.values())).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name != 'ensemble':
                    prob = model.predict_proba(X_scaled)[0]
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = pred
                    probabilities[name] = prob
            
            # Ensemble prediction
            ensemble_pred = self.models['ensemble'].predict(X_scaled)[0]
            ensemble_prob = self.models['ensemble'].predict_proba(X_scaled)[0]
            
            # Get confidence scores
            confidence = np.max(ensemble_prob)
            
            # Determine if abnormal
            is_abnormal = ensemble_pred == 1  # Assuming 1 is abnormal class
            
            # Get detailed probability breakdown
            class_names = ['Normal', 'Abnormal', 'Speech', 'Background']
            if len(ensemble_prob) > len(class_names):
                class_names.extend([f'Class_{i}' for i in range(len(ensemble_prob) - len(class_names))])
            
            return {
                'is_abnormal': bool(is_abnormal),
                'predicted_class': int(ensemble_pred),
                'confidence': float(confidence),
                'class_probabilities': dict(zip(class_names[:len(ensemble_prob)], ensemble_prob)),
                'model_predictions': predictions,
                'features': features
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_model(self, filename='enhanced_sound_detector.pkl'):
        """Save trained model"""
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(save_data, filename)
        print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename='enhanced_sound_detector.pkl'):
        """Load trained model"""
        try:
            save_data = joblib.load(filename)
            self.models = save_data['models']
            self.scaler = save_data['scaler']
            self.is_trained = save_data['is_trained']
            print(f"✅ Model loaded from {filename}")
            return True
        except:
            print(f"❌ Could not load model from {filename}")
            return False

# ============================================================================
# 3. REAL-TIME DETECTION WITH THRESHOLD OPTIMIZATION
# ============================================================================

class RealTimeAbnormalDetector:
    """Real-time detection with adaptive thresholds"""
    
    def __init__(self, sr=16000, window_size=1.0, hop_size=0.5):
        self.sr = sr
        self.window_size = int(window_size * sr)
        self.hop_size = int(hop_size * sr)
        self.buffer = np.array([])
        
        # Adaptive thresholds (initialize with defaults)
        self.thresholds = {
            'rms': 0.25,
            'zcr': 0.35,
            'centroid': 3000,
            'crest': 8.0,
            'spectral_flux': 5.0
        }
        
        # History for adaptive thresholds
        self.history = {
            'rms': [],
            'zcr': [],
            'centroid': [],
            'crest': []
        }
        
        # Load pre-trained detector
        self.detector = EnhancedAbnormalSoundDetector()
        if not self.detector.load_model():
            print("⚠️ No pre-trained model found. Using rule-based detection.")
    
    def update_thresholds(self, features):
        """Adapt thresholds based on recent history"""
        for key in self.history.keys():
            if key in features:
                self.history[key].append(features[key])
                if len(self.history[key]) > 100:  # Keep last 100 values
                    self.history[key].pop(0)
                
                # Update threshold as mean + 2*std of recent history
                if len(self.history[key]) > 10:
                    mean_val = np.mean(self.history[key])
                    std_val = np.std(self.history[key])
                    self.thresholds[key] = mean_val + 2 * std_val
    
    def detect_in_buffer(self, audio_chunk):
        """Detect abnormal sounds in audio chunk"""
        # Extract features
        features = AdvancedAudioFeatures().extract_all_features(audio_chunk, self.sr)
        
        # Update adaptive thresholds
        self.update_thresholds(features)
        
        # Rule-based detection with adaptive thresholds
        alerts = []
        
        if features['max_rms'] > self.thresholds['rms']:
            alerts.append(f"Loud (RMS: {features['max_rms']:.3f} > {self.thresholds['rms']:.3f})")
        
        if features['crest_factor'] > self.thresholds['crest']:
            alerts.append(f"Sharp (Crest: {features['crest_factor']:.1f} > {self.thresholds['crest']:.1f})")
        
        if features['max_zcr'] > self.thresholds['zcr']:
            alerts.append(f"Abrupt (ZCR: {features['max_zcr']:.3f} > {self.thresholds['zcr']:.3f})")
        
        if features['centroid_mean'] > self.thresholds['centroid']:
            alerts.append(f"High freq ({features['centroid_mean']:.0f}Hz > {self.thresholds['centroid']:.0f}Hz)")
        
        # Use ML model if available
        ml_result = None
        if self.detector.is_trained:
            # Temporary save chunk and predict
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio_chunk, self.sr)
                ml_result = self.detector.predict(tmp.name)
                os.unlink(tmp.name)
        
        return {
            'is_abnormal': len(alerts) > 0,
            'alerts': alerts,
            'features': features,
            'ml_result': ml_result,
            'thresholds': self.thresholds.copy()
        }
    
    def process_stream(self, audio_stream, duration=10):
        """Process audio stream in real-time"""
        import time
        
        print(f"🎤 Listening for {duration} seconds...")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        abnormal_count = 0
        total_chunks = 0
        
        try:
            while time.time() - start_time < duration:
                # Get audio chunk (simulated - replace with actual stream)
                if len(self.buffer) < self.window_size:
                    # Simulate getting more audio
                    chunk = np.random.randn(self.hop_size) * 0.1
                    self.buffer = np.concatenate([self.buffer, chunk])
                
                if len(self.buffer) >= self.window_size:
                    # Process window
                    window = self.buffer[:self.window_size]
                    result = self.detect_in_buffer(window)
                    
                    total_chunks += 1
                    
                    if result['is_abnormal']:
                        abnormal_count += 1
                        current_time = time.time() - start_time
                        print(f"\n[{current_time:.1f}s] 🚨 ABNORMAL SOUND DETECTED!")
                        for alert in result['alerts'][:2]:  # Show first 2 alerts
                            print(f"   • {alert}")
                        
                        if result['ml_result']:
                            print(f"   ML Confidence: {result['ml_result']['confidence']:.1%}")
                    
                    # Slide window
                    self.buffer = self.buffer[self.hop_size:]
                
                time.sleep(0.1)  # Simulate real-time
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        print(f"\n📊 Summary: {abnormal_count}/{total_chunks} abnormal chunks detected")
        return abnormal_count

# ============================================================================
# 4. ADVANCED VISUALIZATION AND ANALYSIS
# ============================================================================

def plot_detailed_analysis(audio_path, result=None):
    """Create comprehensive visualization"""
    audio, sr = librosa.load(audio_path, sr=16000, duration=4.0)
    
    # Extract features
    features = AdvancedAudioFeatures().extract_all_features(audio, sr)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Waveform with abnormalities highlighted
    ax1 = plt.subplot(3, 3, 1)
    time = np.linspace(0, len(audio)/sr, len(audio))
    ax1.plot(time, audio, 'b-', alpha=0.7, linewidth=0.5)
    
    # Highlight potential abnormalities
    rms = librosa.feature.rms(y=audio)[0]
    rms_threshold = 0.25
    abnormal_regions = rms > rms_threshold
    if np.any(abnormal_regions):
        frames = np.where(abnormal_regions)[0]
        t_frames = librosa.frames_to_time(frames, sr=sr)
        for t in t_frames:
            ax1.axvspan(t-0.1, t+0.1, alpha=0.3, color='red')
    
    ax1.set_title("Waveform with Abnormal Regions", fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # 2. Spectrogram with annotations
    ax2 = plt.subplot(3, 3, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2, cmap='hot')
    ax2.set_title("Spectrogram", fontweight='bold')
    
    # 3. Feature radar chart
    ax3 = plt.subplot(3, 3, 3, projection='polar')
    key_features = ['Loudness', 'Sharpness', 'Abruptness', 'High Freq', 'Impulsiveness']
    feature_values = [
        min(features['max_rms'] * 3, 1.0),
        min(features['crest_factor'] / 15, 1.0),
        min(features['max_zcr'] * 2, 1.0),
        min(features['centroid_mean'] / 5000, 1.0),
        min(features['impulse_ratio'] * 10, 1.0)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(key_features), endpoint=False).tolist()
    feature_values += feature_values[:1]
    angles += angles[:1]
    
    ax3.plot(angles, feature_values, 'o-', linewidth=2)
    ax3.fill(angles, feature_values, alpha=0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(key_features)
    ax3.set_title("Acoustic Feature Profile", fontweight='bold')
    ax3.grid(True)
    
    # 4. Time-domain features
    ax4 = plt.subplot(3, 3, 4)
    rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    ax4.plot(rms_time, rms, 'g-', label='RMS Energy')
    ax4.axhline(y=0.25, color='r', linestyle='--', label='Abnormal Threshold')
    ax4.set_title("Energy Over Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("RMS")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Frequency analysis
    ax5 = plt.subplot(3, 3, 5)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    ax5.semilogy(freqs, fft, 'b-')
    ax5.set_title("Frequency Spectrum")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Magnitude")
    ax5.grid(True, alpha=0.3)
    
    # 6. Detection result
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('off')
    
    is_abnormal = False
    if result and 'is_abnormal' in result:
        is_abnormal = result['is_abnormal']
    
    if is_abnormal:
        ax6.text(0.5, 0.7, '🚨 ABNORMAL SOUND', 
                fontsize=20, fontweight='bold', color='red',
                ha='center', va='center')
        if result and 'confidence' in result:
            ax6.text(0.5, 0.5, f'Confidence: {result["confidence"]:.1%}',
                    fontsize=14, ha='center', va='center')
        if result and 'class_probabilities' in result:
            probs = result['class_probabilities']
            top_class = max(probs.items(), key=lambda x: x[1])
            ax6.text(0.5, 0.3, f'Type: {top_class[0]}',
                    fontsize=14, ha='center', va='center')
    else:
        ax6.text(0.5, 0.7, '✅ NORMAL SOUND', 
                fontsize=20, fontweight='bold', color='green',
                ha='center', va='center')
    
    # 7. Feature importance bars
    ax7 = plt.subplot(3, 3, 7)
    top_features = {
        'Max RMS': features['max_rms'],
        'Crest Factor': min(features['crest_factor'], 20),
        'Zero-Crossing': features['max_zcr'],
        'Spectral Centroid': features['centroid_mean'] / 1000,
        'Impulse Ratio': features['impulse_ratio'] * 10
    }
    
    colors = ['red' if is_abnormal else 'blue' for _ in top_features]
    ax7.barh(list(top_features.keys()), list(top_features.values()), color=colors)
    ax7.set_title("Key Feature Values")
    ax7.set_xlabel("Normalized Value")
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Onset detection
    ax8 = plt.subplot(3, 3, 8)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    ax8.plot(onset_times, onset_env, 'r-', label='Onset Strength')
    ax8.set_title("Transient Detection")
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Onset Strength")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistical distribution
    ax9 = plt.subplot(3, 3, 9)
    ax9.hist(audio, bins=50, alpha=0.7, density=True)
    ax9.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax9.set_title("Amplitude Distribution")
    ax9.set_xlabel("Amplitude")
    ax9.set_ylabel("Density")
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f"Advanced Sound Analysis: {os.path.basename(audio_path)}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. MAIN APPLICATION WITH GUI (OPTIONAL)
# ============================================================================

def main_menu():
    """Main menu for the enhanced detector"""
    print("=" * 70)
    print("ENHANCED ABNORMAL SOUND DETECTION SYSTEM")
    print("=" * 70)
    print("Features:")
    print("  • Advanced feature extraction (50+ features)")
    print("  • Ensemble machine learning (4 models)")
    print("  • Adaptive thresholding")
    print("  • Real-time detection capability")
    print("  • Detailed visualization")
    print("=" * 70)
    
    detector = EnhancedAbnormalSoundDetector()
    
    while True:
        print("\n📋 MAIN MENU:")
        print("  1. Quick test a sound file")
        print("  2. Train on custom dataset")
        print("  3. Real-time detection (microphone)")
        print("  4. Batch test directory")
        print("  5. Advanced visualization")
        print("  6. Save/Load model")
        print("  7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            file_path = input("Enter audio file path: ").strip().strip('"\'')
            if os.path.exists(file_path):
                result = detector.predict(file_path)
                if result:
                    print(f"\n🔍 RESULT:")
                    if result['is_abnormal']:
                        print(f"  🚨 ABNORMAL SOUND DETECTED!")
                        print(f"  Confidence: {result['confidence']:.1%}")
                        for cls, prob in sorted(result['class_probabilities'].items(), 
                                              key=lambda x: x[1], reverse=True)[:3]:
                            print(f"  {cls}: {prob:.1%}")
                    else:
                        print(f"  ✅ NORMAL SOUND")
                    
                    # Show detailed visualization
                    plot_choice = input("\nShow detailed analysis? (y/n): ").lower()
                    if plot_choice == 'y':
                        plot_detailed_analysis(file_path, result)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '2':
            print("\n⚠️  Training requires labeled dataset.")
            print("Create folders: abnormal/, normal/, speech/, background/")
            print("Place audio files in respective folders.")
            
            data_dir = input("\nEnter dataset directory: ").strip()
            if os.path.exists(data_dir):
                # This would require actual dataset preparation
                print("Dataset training would be implemented here...")
            else:
                print("Directory not found.")
        
        elif choice == '3':
            print("\n🎤 Real-time detection starting...")
            rt_detector = RealTimeAbnormalDetector()
            rt_detector.process_stream(None, duration=30)
        
        elif choice == '4':
            test_dir = input("\nEnter directory to test: ").strip()
            if os.path.exists(test_dir):
                import glob
                audio_files = glob.glob(os.path.join(test_dir, "*.wav")) + \
                             glob.glob(os.path.join(test_dir, "*.mp3"))
                
                if audio_files:
                    print(f"\nFound {len(audio_files)} files. Testing first 10...")
                    for i, file in enumerate(audio_files[:10]):
                        print(f"\n[{i+1}/10] {os.path.basename(file)}")
                        result = detector.predict(file)
                        if result:
                            status = "🚨 ABNORMAL" if result['is_abnormal'] else "✅ NORMAL"
                            print(f"  {status} (Confidence: {result['confidence']:.1%})")
        
        elif choice == '5':
            file_path = input("\nEnter audio file for visualization: ").strip().strip('"\'')
            if os.path.exists(file_path):
                plot_detailed_analysis(file_path)
            else:
                print("File not found.")
        
        elif choice == '6':
            print("\n💾 Model Management:")
            print("  1. Save current model")
            print("  2. Load saved model")
            print("  3. Back to main menu")
            
            sub_choice = input("Select (1-3): ").strip()
            if sub_choice == '1':
                detector.save_model()
            elif sub_choice == '2':
                detector.load_model()
        
        elif choice == '7':
            print("\n👋 Exiting...")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

# ============================================================================
# SIMPLE COMMAND-LINE INTERFACE
# ============================================================================

def simple_cli():
    """Simple command-line interface"""
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter audio file path: ").strip().strip('"\'').strip()
    
    # Fix Windows path if needed
    file_path = file_path.replace('\\', '/')
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n🔍 Analyzing: {os.path.basename(file_path)}")
    
    # Use enhanced detector
    detector = EnhancedAbnormalSoundDetector()
    
    # Try to load pre-trained model
    if not detector.load_model():
        print("⚠️ Using rule-based detection (no trained model found)")
        # Fall back to quick detection
        audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
        features = AdvancedAudioFeatures().extract_all_features(audio, sr)
        
        # Simple rule-based detection
        is_abnormal = (
            features['max_rms'] > 0.3 or
            features['crest_factor'] > 8 or
            features['max_zcr'] > 0.4 or
            features['centroid_mean'] > 3500
        )
        
        if is_abnormal:
            if features['max_rms'] > 0.4 and features['crest_factor'] > 10:
                sound_type = "CAR CRASH / EXPLOSION"
            elif features['centroid_mean'] > 4000:
                sound_type = "GUNSHOT / GLASS"
            else:
                sound_type = "ABNORMAL SOUND"
            
            print(f"\n🚨 {sound_type} DETECTED!")
            print(f"  Max Loudness: {features['max_rms']:.3f}")
            print(f"  Crest Factor: {features['crest_factor']:.1f}")
            print(f"  Frequency: {features['centroid_mean']:.0f} Hz")
        else:
            print(f"\n✅ NORMAL SOUND")
    else:
        # Use ML model
        result = detector.predict(file_path)
        if result:
            if result['is_abnormal']:
                print(f"\n🚨 ABNORMAL SOUND DETECTED!")
                print(f"  Confidence: {result['confidence']:.1%}")
                
                # Show top predictions
                print(f"\n📊 Probability Breakdown:")
                for cls, prob in sorted(result['class_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {cls}: {prob:.1%}")
            else:
                print(f"\n✅ NORMAL SOUND")
                print(f"  Confidence: {result['confidence']:.1%}")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # For simple command-line usage
    if len(sys.argv) > 1:
        simple_cli()
    else:
        # For interactive mode
        main_menu()