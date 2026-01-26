# not accurate much


# ============================================================================
# NO-TRAINING ABNORMAL SOUND DETECTOR - WORKS IMMEDIATELY
# ============================================================================

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal, stats

print("=" * 70)
print("🎯 ACCURATE ABNORMAL SOUND DETECTOR")
print("No training required - Works immediately!")
print("=" * 70)

class IntelligentSoundAnalyzer:
    """Advanced sound analysis without ML training"""
    
    def __init__(self):
        # Sound type profiles (pre-defined based on acoustic research)
        self.sound_profiles = {
            'car_crash': {
                'rms_min': 0.35,
                'crest_min': 9.0,
                'zcr_range': (0.3, 0.6),
                'centroid_range': (2000, 5000),
                'impulse_min': 0.02
            },
            'gunshot': {
                'rms_min': 0.25,
                'crest_min': 10.0,
                'zcr_range': (0.4, 0.7),
                'centroid_range': (3500, 8000),
                'impulse_min': 0.03
            },
            'explosion': {
                'rms_min': 0.4,
                'crest_min': 8.0,
                'zcr_range': (0.25, 0.5),
                'centroid_range': (1000, 4000),
                'impulse_min': 0.04
            },
            'scream': {
                'rms_min': 0.3,
                'crest_min': 6.0,
                'zcr_range': (0.35, 0.6),
                'centroid_range': (2500, 5000),
                'impulse_min': 0.01
            },
            'glass_break': {
                'rms_min': 0.2,
                'crest_min': 11.0,
                'zcr_range': (0.45, 0.8),
                'centroid_range': (4000, 10000),
                'impulse_min': 0.025
            }
        }
    
    def extract_advanced_features(self, audio, sr=16000):
        """Extract comprehensive acoustic features"""
        features = {}
        
        # ===== BASIC FEATURES =====
        # 1. Energy features
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        features['max_rms'] = float(np.max(rms))
        features['mean_rms'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 2. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)[0]
        features['max_zcr'] = float(np.max(zcr))
        features['mean_zcr'] = float(np.mean(zcr))
        
        # 3. Spectral features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=2048, hop_length=512)[0]
        
        features['centroid_mean'] = float(np.mean(centroid))
        features['centroid_std'] = float(np.std(centroid))
        features['bandwidth_mean'] = float(np.mean(bandwidth))
        features['rolloff_mean'] = float(np.mean(rolloff))
        
        # 4. Crest factor
        crest = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        features['crest_factor'] = float(crest)
        
        # ===== ADVANCED FEATURES =====
        # 5. Impulse detection
        impulse_threshold = 3 * np.std(audio)
        impulse_count = np.sum(np.abs(audio) > impulse_threshold)
        features['impulse_ratio'] = float(impulse_count / len(audio))
        
        # 6. Spectral flux (change detection)
        spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        spectral_flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        features['flux_max'] = float(np.max(spectral_flux))
        features['flux_mean'] = float(np.mean(spectral_flux))
        
        # 7. Harmonic-Percussive separation
        harmonic = librosa.effects.harmonic(audio)
        percussive = librosa.effects.percussive(audio)
        features['harmonic_ratio'] = float(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-8))
        
        # 8. Onset strength (transient detection)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
        features['onset_max'] = float(np.max(onset_env))
        features['onset_mean'] = float(np.mean(onset_env))
        
        # 9. MFCC statistics
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        
        # 10. Silence ratio
        silence_threshold = 0.02 * np.max(np.abs(audio))
        silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
        features['silence_ratio'] = float(silence_ratio)
        
        # 11. Kurtosis and Skewness
        features['kurtosis'] = float(stats.kurtosis(audio))
        features['skewness'] = float(stats.skew(audio))
        
        return features
    
    def match_sound_profile(self, features):
        """Match extracted features to known sound profiles"""
        matches = {}
        
        for sound_type, profile in self.sound_profiles.items():
            score = 0
            total_weight = 0
            
            # Check RMS (loudness)
            if features['max_rms'] >= profile['rms_min']:
                score += 25
            total_weight += 25
            
            # Check Crest factor (sharpness)
            if features['crest_factor'] >= profile['crest_min']:
                score += 20
            total_weight += 20
            
            # Check ZCR (abruptness)
            zcr_min, zcr_max = profile['zcr_range']
            if zcr_min <= features['max_zcr'] <= zcr_max:
                score += 15
            total_weight += 15
            
            # Check Spectral centroid (frequency)
            centroid_min, centroid_max = profile['centroid_range']
            if centroid_min <= features['centroid_mean'] <= centroid_max:
                score += 20
            total_weight += 20
            
            # Check Impulse ratio
            if features['impulse_ratio'] >= profile['impulse_min']:
                score += 10
            total_weight += 10
            
            # Calculate match percentage
            match_percent = (score / total_weight) * 100
            matches[sound_type] = match_percent
        
        return matches
    
    def analyze_sound(self, audio_path):
        """Main analysis function"""
        print(f"\n🔊 Analyzing: {os.path.basename(audio_path)}")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            duration = len(audio) / sr
            print(f"   Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
            
            # Extract features
            features = self.extract_advanced_features(audio, sr)
            
            # Match to profiles
            profile_matches = self.match_sound_profile(features)
            
            # Calculate abnormality score
            abnormality_score = self.calculate_abnormality_score(features)
            
            # Determine result
            best_match = max(profile_matches.items(), key=lambda x: x[1])
            best_type, best_score = best_match
            
            is_abnormal = False
            sound_type = "Normal"
            confidence = 0
            
            if abnormality_score > 60 or best_score > 65:
                is_abnormal = True
                sound_type = best_type.replace('_', ' ').title()
                confidence = max(abnormality_score, best_score) / 100
            else:
                sound_type = "Normal"
                confidence = 1.0 - (abnormality_score / 100)
            
            # Display results
            self.display_results(features, profile_matches, abnormality_score, 
                               is_abnormal, sound_type, confidence)
            
            # Optional visualization
            if len(sys.argv) > 2 and sys.argv[2] == '--plot':
                self.plot_analysis(audio, sr, features, sound_type, confidence)
            
            return {
                'is_abnormal': is_abnormal,
                'sound_type': sound_type,
                'confidence': confidence,
                'abnormality_score': abnormality_score,
                'best_match': best_type,
                'match_score': best_score,
                'features': features
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def calculate_abnormality_score(self, features):
        """Calculate overall abnormality score (0-100)"""
        score = 0
        
        # Loudness contributes 30%
        if features['max_rms'] > 0.3:
            score += 30 * min(1.0, features['max_rms'] / 0.5)
        
        # Sharpness contributes 25%
        if features['crest_factor'] > 7:
            score += 25 * min(1.0, features['crest_factor'] / 15)
        
        # Abruptness contributes 20%
        if features['max_zcr'] > 0.3:
            score += 20 * min(1.0, features['max_zcr'] / 0.8)
        
        # High frequency contributes 15%
        if features['centroid_mean'] > 3000:
            score += 15 * min(1.0, features['centroid_mean'] / 8000)
        
        # Impulsiveness contributes 10%
        if features['impulse_ratio'] > 0.01:
            score += 10 * min(1.0, features['impulse_ratio'] / 0.1)
        
        return min(100, score)
    
    def display_results(self, features, profile_matches, abnormality_score, 
                       is_abnormal, sound_type, confidence):
        """Display analysis results"""
        
        print(f"\n📊 FEATURE ANALYSIS:")
        print("-" * 50)
        print(f"  Max Loudness:       {features['max_rms']:.4f}")
        print(f"  Crest Factor:       {features['crest_factor']:.2f}")
        print(f"  Zero-Crossing Rate: {features['max_zcr']:.4f}")
        print(f"  Spectral Centroid:  {features['centroid_mean']:.0f} Hz")
        print(f"  Impulse Ratio:      {features['impulse_ratio']:.4f}")
        print(f"  Harmonic Ratio:     {features['harmonic_ratio']:.2f}")
        print(f"  Onset Strength:     {features['onset_max']:.2f}")
        
        print(f"\n🎯 PROFILE MATCHES:")
        print("-" * 50)
        for sound_type, match_score in sorted(profile_matches.items(), 
                                             key=lambda x: x[1], reverse=True):
            if match_score > 40:
                print(f"  {sound_type.replace('_', ' ').title():<20} {match_score:5.1f}%")
        
        print(f"\n🔍 DETECTION SUMMARY:")
        print("-" * 50)
        print(f"  Abnormality Score:  {abnormality_score:.1f}/100")
        
        if is_abnormal:
            print(f"\n🚨 {sound_type.upper()} DETECTED!")
            print(f"   Confidence: {confidence:.1%}")
            
            if abnormality_score > 80:
                print(f"   ⚠️  HIGH RISK - Immediate attention required!")
            elif abnormality_score > 60:
                print(f"   ⚠️  Medium risk - Investigate further")
        else:
            print(f"\n✅ NORMAL SOUND")
            print(f"   Confidence: {confidence:.1%}")
    
    def plot_analysis(self, audio, sr, features, sound_type, confidence):
        """Create visualization plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Waveform
        time = np.linspace(0, len(audio)/sr, len(audio))
        color = 'red' if 'abnormal' in sound_type.lower() else 'green'
        axes[0,0].plot(time, audio, color=color, alpha=0.7, linewidth=0.5)
        axes[0,0].set_title("Waveform")
        axes[0,0].set_xlabel("Time (s)")
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', 
                                sr=sr, ax=axes[0,1], cmap='hot')
        axes[0,1].set_title("Spectrogram")
        
        # 3. Feature radar
        ax_radar = plt.subplot(2, 3, 3, projection='polar')
        feature_names = ['Loudness', 'Sharpness', 'Abruptness', 'High Freq', 'Impulse']
        feature_values = [
            min(features['max_rms'] * 3, 1.0),
            min(features['crest_factor'] / 15, 1.0),
            min(features['max_zcr'] * 1.5, 1.0),
            min(features['centroid_mean'] / 8000, 1.0),
            min(features['impulse_ratio'] * 20, 1.0)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        feature_values += feature_values[:1]
        angles += angles[:1]
        
        ax_radar.plot(angles, feature_values, 'o-', linewidth=2)
        ax_radar.fill(angles, feature_values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(feature_names)
        ax_radar.set_title("Feature Profile")
        
        # 4. Energy over time
        rms = librosa.feature.rms(y=audio)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        axes[1,0].plot(rms_time, rms, 'g-')
        axes[1,0].axhline(y=0.25, color='r', linestyle='--', alpha=0.7)
        axes[1,0].set_title("Energy (RMS)")
        axes[1,0].set_xlabel("Time (s)")
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_time = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        axes[1,1].plot(onset_time, onset_env, 'r-')
        axes[1,1].set_title("Transient Detection")
        axes[1,1].set_xlabel("Time (s)")
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Result display
        axes[1,2].axis('off')
        if 'abnormal' in sound_type.lower():
            axes[1,2].text(0.5, 0.7, '🚨 ABNORMAL', 
                          fontsize=16, fontweight='bold', color='red',
                          ha='center', va='center')
            axes[1,2].text(0.5, 0.5, sound_type, 
                          fontsize=14, ha='center', va='center')
            axes[1,2].text(0.5, 0.3, f'Confidence: {confidence:.1%}', 
                          fontsize=12, ha='center', va='center')
        else:
            axes[1,2].text(0.5, 0.7, '✅ NORMAL', 
                          fontsize=16, fontweight='bold', color='green',
                          ha='center', va='center')
            axes[1,2].text(0.5, 0.5, 'Sound', 
                          fontsize=14, ha='center', va='center')
            axes[1,2].text(0.5, 0.3, f'Confidence: {confidence:.1%}', 
                          fontsize=12, ha='center', va='center')
        
        plt.suptitle(f"Sound Analysis: {sound_type}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    analyzer = IntelligentSoundAnalyzer()
    
    while True:
        print("\n" + "=" * 50)
        print("MENU:")
        print("  1. Test a sound file")
        print("  2. Test with visualization")
        print("  3. Batch test directory")
        print("  4. Exit")
        print("=" * 50)
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            file_path = input("\n📁 Enter audio file path: ").strip().strip('"\'')
            file_path = file_path.replace('\\', '/')
            
            if os.path.exists(file_path):
                analyzer.analyze_sound(file_path)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '2':
            file_path = input("\n📁 Enter audio file path: ").strip().strip('"\'')
            file_path = file_path.replace('\\', '/')
            
            if os.path.exists(file_path):
                # Run with plot flag
                import subprocess
                subprocess.run([sys.executable, __file__, file_path, '--plot'])
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '3':
            test_dir = input("\n📂 Enter directory path: ").strip().strip('"\'')
            test_dir = test_dir.replace('\\', '/')
            
            if os.path.exists(test_dir):
                import glob
                audio_files = glob.glob(os.path.join(test_dir, "*.wav")) + \
                             glob.glob(os.path.join(test_dir, "*.mp3")) + \
                             glob.glob(os.path.join(test_dir, "*.flac"))
                
                if audio_files:
                    print(f"\nFound {len(audio_files)} audio files")
                    print("Testing first 5 files...")
                    
                    abnormal_count = 0
                    for i, file in enumerate(audio_files[:5]):
                        print(f"\n[{i+1}/5] {os.path.basename(file)}")
                        result = analyzer.analyze_sound(file)
                        if result and result['is_abnormal']:
                            abnormal_count += 1
                    
                    print(f"\n📊 BATCH SUMMARY:")
                    print(f"  Abnormal: {abnormal_count}/5")
                    print(f"  Normal:   {5 - abnormal_count}/5")
                else:
                    print("❌ No audio files found in directory")
            else:
                print("❌ Directory not found")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        file_path = sys.argv[1]
        file_path = file_path.strip('"\'')
        
        analyzer = IntelligentSoundAnalyzer()
        
        if len(sys.argv) > 2 and sys.argv[2] == '--plot':
            result = analyzer.analyze_sound(file_path)
            if result:
                audio, sr = librosa.load(file_path, sr=16000, duration=3.0)
                analyzer.plot_analysis(audio, sr, result['features'], 
                                     result['sound_type'], result['confidence'])
        else:
            analyzer.analyze_sound(file_path)
    else:
        # Interactive mode
        main()