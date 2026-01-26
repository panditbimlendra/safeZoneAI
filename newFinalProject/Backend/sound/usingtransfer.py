# accurate_detector.py - USING PRE-TRAINED TRANSFORMER MODELS
import numpy as np
import torch
import torchaudio
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🤖 STATE-OF-THE-ART ABNORMAL SOUND DETECTION")
print("Using pre-trained transformer models")
print("=" * 70)

class AdvancedAbnormalDetector:
    def __init__(self):
        print("Loading pre-trained models...")
        
        # Load multiple specialized models
        self.models = {}
        
        try:
            # Model 1: Facebook's Wav2Vec2 for audio classification
            print("  1. Loading Facebook Wav2Vec2...")
            self.models['wav2vec2'] = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-base",
                device=-1  # Use CPU (-1) or GPU (0)
            )
            
            # Model 2: MIT's Audio Spectrogram Transformer
            print("  2. Loading MIT AST model...")
            self.models['ast'] = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            
            # Model 3: Google's YAMNet (if transformers fails, use direct)
            print("  3. Loading sound event detection model...")
            self.models['sound_events'] = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-ks"
            )
            
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Some models failed to load: {e}")
            print("Using fallback models...")
            self.load_fallback_models()
    
    def load_fallback_models(self):
        """Load alternative pre-trained models"""
        try:
            # Fallback: Simple audio classifier
            self.models['fallback'] = pipeline(
                "audio-classification",
                model="m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
            )
        except:
            print("❌ Could not load any pre-trained models")
            self.models = {}
    
    def detect_with_wav2vec2(self, audio_path):
        """Use Facebook's Wav2Vec2 for classification"""
        try:
            result = self.models['wav2vec2'](audio_path)
            return result[:5]  # Top 5 predictions
        except:
            return None
    
    def detect_with_ast(self, audio_path):
        """Use MIT's Audio Spectrogram Transformer"""
        try:
            result = self.models['ast'](audio_path)
            
            # Filter for abnormal sounds
            abnormal_keywords = [
                'gunshot', 'explosion', 'crash', 'scream', 'siren',
                'alarm', 'breaking', 'fire', 'emergency', 'accident',
                'shot', 'bang', 'boom', 'cry', 'yell', 'shout'
            ]
            
            abnormal_results = []
            for pred in result:
                label = pred['label'].lower()
                score = pred['score']
                
                # Check if label contains abnormal keywords
                if any(keyword in label for keyword in abnormal_keywords):
                    abnormal_results.append({
                        'label': label,
                        'score': score,
                        'type': self.classify_abnormality(label)
                    })
            
            return abnormal_results[:3]  # Top 3 abnormal
            
        except:
            return None
    
    def classify_abnormality(self, label):
        """Classify the type of abnormality"""
        label = label.lower()
        
        if any(word in label for word in ['gun', 'shot', 'firearm', 'bullet']):
            return 'GUNSHOT'
        elif any(word in label for word in ['crash', 'collision', 'accident', 'car']):
            return 'CAR_CRASH'
        elif any(word in label for word in ['explosion', 'blast', 'boom', 'bang']):
            return 'EXPLOSION'
        elif any(word in label for word in ['scream', 'yell', 'shout', 'cry']):
            return 'SCREAM'
        elif any(word in label for word in ['siren', 'alarm', 'emergency']):
            return 'EMERGENCY_SIREN'
        elif any(word in label for word in ['glass', 'break', 'shatter']):
            return 'GLASS_BREAKING'
        elif any(word in label for word in ['fire', 'flame', 'burn']):
            return 'FIRE'
        else:
            return 'ABNORMAL_SOUND'
    
    def analyze_audio_file(self, audio_path):
        """Main analysis function"""
        print(f"\n🔊 Analyzing: {os.path.basename(audio_path)}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                transform = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = transform(waveform)
                sample_rate = 16000
            
            print(f"   Duration: {waveform.shape[1]/sample_rate:.2f}s")
            print(f"   Sample Rate: {sample_rate}Hz")
            
            # Get predictions from all models
            all_predictions = []
            
            # Wav2Vec2 predictions
            wav2vec_preds = self.detect_with_wav2vec2(audio_path)
            if wav2vec_preds:
                print(f"\n📊 Wav2Vec2 Predictions:")
                for pred in wav2vec_preds[:3]:
                    print(f"   • {pred['label']}: {pred['score']:.1%}")
                    all_predictions.append(pred)
            
            # AST predictions
            ast_preds = self.detect_with_ast(audio_path)
            if ast_preds:
                print(f"\n📊 Audio Spectrogram Transformer Predictions:")
                for pred in ast_preds:
                    print(f"   🚨 {pred['type']}: {pred['score']:.1%} ({pred['label']})")
                    all_predictions.append({
                        'label': pred['type'],
                        'score': pred['score']
                    })
            
            # Determine if abnormal
            is_abnormal = False
            confidence = 0
            sound_type = "Normal"
            
            if ast_preds:
                is_abnormal = True
                top_pred = max(ast_preds, key=lambda x: x['score'])
                sound_type = top_pred['type']
                confidence = top_pred['score']
            elif wav2vec_preds:
                # Check if any wav2vec prediction looks abnormal
                for pred in wav2vec_preds:
                    label = pred['label'].lower()
                    if any(word in label for word in ['scream', 'cry', 'yell', 'gun']):
                        is_abnormal = True
                        sound_type = "SUSPICIOUS_SOUND"
                        confidence = pred['score']
                        break
            
            # Display final result
            print(f"\n" + "=" * 50)
            if is_abnormal:
                print(f"🚨 ABNORMAL SOUND DETECTED!")
                print(f"   Type: {sound_type}")
                print(f"   Confidence: {confidence:.1%}")
                
                if confidence > 0.7:
                    print(f"⚠️  HIGH CONFIDENCE - Immediate attention required!")
                elif confidence > 0.4:
                    print(f"⚠️  Medium confidence - Investigate further")
            else:
                print(f"✅ NORMAL SOUND")
                if wav2vec_preds:
                    top_normal = wav2vec_preds[0]
                    print(f"   Most likely: {top_normal['label']}")
            
            # Acoustic analysis for additional verification
            self.acoustic_verification(audio_path, is_abnormal)
            
            return {
                'is_abnormal': is_abnormal,
                'sound_type': sound_type,
                'confidence': confidence,
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            print(f"❌ Error analyzing file: {str(e)}")
            return None
    
    def acoustic_verification(self, audio_path, ml_result):
        """Additional acoustic verification"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            
            # Basic acoustic features
            rms = librosa.feature.rms(y=audio)[0]
            max_rms = np.max(rms)
            
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            max_zcr = np.max(zcr)
            
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_centroid = np.mean(centroid)
            
            print(f"\n🔬 Acoustic Verification:")
            print(f"   Max Loudness: {max_rms:.3f}")
            print(f"   Abruptness: {max_zcr:.3f}")
            print(f"   Frequency: {mean_centroid:.0f} Hz")
            
            # Cross-verify with acoustic rules
            acoustic_abnormal = (
                max_rms > 0.3 or  # Very loud
                max_zcr > 0.4 or  # Very abrupt
                mean_centroid > 4000  # High frequency
            )
            
            if ml_result and acoustic_abnormal:
                print("   ✅ Acoustic features confirm ML prediction")
            elif ml_result and not acoustic_abnormal:
                print("   ⚠️  Acoustic features don't strongly support ML prediction")
            elif not ml_result and acoustic_abnormal:
                print("   ⚠️  Acoustic features suggest abnormality (false negative?)")
            
        except:
            pass

# ============================================================================
# SIMPLER VERSION WITH ESSENTIAL LIBRARIES ONLY
# ============================================================================

def simple_accurate_detector(audio_path):
    """Simple but accurate detection using essential libraries"""
    
    print("=" * 70)
    print("🎯 SIMPLE BUT ACCURATE ABNORMAL SOUND DETECTOR")
    print("=" * 70)
    
    try:
        import librosa
        import numpy as np
        from scipy import signal, stats
        
        # Load audio
        print(f"\n🔊 Loading: {os.path.basename(audio_path)}")
        audio, sr = librosa.load(audio_path, sr=16000, duration=4.0)
        duration = len(audio) / sr
        print(f"   Duration: {duration:.2f}s")
        
        # ADVANCED FEATURE EXTRACTION
        features = {}
        
        # 1. Short-time Fourier Transform analysis
        stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        
        # 2. Mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_mean'] = np.mean(mel_spec_db)
        features['mel_std'] = np.std(mel_spec_db)
        
        # 3. MFCC with derivatives
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2)
        
        # 4. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['contrast_mean'] = np.mean(spectral_contrast)
        
        # 5. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        
        # 6. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features['zcr_max'] = np.max(zcr)
        features['zcr_mean'] = np.mean(zcr)
        
        # 7. RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_max'] = np.max(rms)
        features['rms_mean'] = np.mean(rms)
        
        # 8. Spectral centroid and bandwidth
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        features['centroid_mean'] = np.mean(centroid)
        features['bandwidth_mean'] = np.mean(bandwidth)
        
        # 9. Crest factor
        crest = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        features['crest_factor'] = crest
        
        # 10. Impulse detection
        impulse_threshold = 4 * np.std(audio)
        impulse_count = np.sum(np.abs(audio) > impulse_threshold)
        features['impulse_ratio'] = impulse_count / len(audio)
        
        # 11. Kurtosis (peakedness)
        features['kurtosis'] = stats.kurtosis(audio)
        
        # 12. Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        
        # 13. Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_max'] = np.max(onset_env)
        
        # DECISION MAKING WITH RULES FROM RESEARCH PAPERS
        abnormality_score = 0
        
        # Rule 1: Very loud sounds (car crashes, explosions)
        if features['rms_max'] > 0.35:
            abnormality_score += 30
            print(f"   🔊 VERY LOUD DETECTED (RMS: {features['rms_max']:.3f})")
        
        # Rule 2: Sharp transients (gunshots, glass breaking)
        if features['crest_factor'] > 9:
            abnormality_score += 25
            print(f"   💥 SHARP TRANSIENT (Crest: {features['crest_factor']:.1f})")
        
        # Rule 3: High frequency content (glass breaking, alarms)
        if features['centroid_mean'] > 4000:
            abnormality_score += 20
            print(f"   📡 HIGH FREQUENCY ({features['centroid_mean']:.0f} Hz)")
        
        # Rule 4: Many impulses (explosions, crashes)
        if features['impulse_ratio'] > 0.02:
            abnormality_score += 15
            print(f"   ⚡ MULTIPLE IMPULSES ({features['impulse_ratio']*100:.1f}%)")
        
        # Rule 5: Abrupt changes (gunshots, door slams)
        if features['zcr_max'] > 0.4:
            abnormality_score += 10
            print(f"   🎯 ABRUPT SOUND (ZCR: {features['zcr_max']:.3f})")
        
        # Determine result
        is_abnormal = abnormality_score > 40
        confidence = min(abnormality_score / 100, 1.0)
        
        # Classify type
        sound_type = "Normal"
        if is_abnormal:
            if features['rms_max'] > 0.4 and features['crest_factor'] > 10:
                sound_type = "🚗 CAR CRASH / EXPLOSION"
            elif features['centroid_mean'] > 4500 and features['crest_factor'] > 11:
                sound_type = "🔫 GUNSHOT / GLASS BREAKING"
            elif features['rms_max'] > 0.35:
                sound_type = "💥 LOUD IMPACT"
            elif features['centroid_mean'] > 4000:
                sound_type = "🚨 HIGH-PITCHED ABNORMALITY"
            else:
                sound_type = "⚠️ ABNORMAL SOUND"
        
        # Display results
        print(f"\n" + "=" * 50)
        print(f"📊 FEATURE ANALYSIS:")
        print(f"   Abnormality Score: {abnormality_score}/100")
        print(f"   Key Features: RMS={features['rms_max']:.3f}, "
              f"Crest={features['crest_factor']:.1f}, "
              f"Freq={features['centroid_mean']:.0f}Hz")
        
        print(f"\n🔍 DETECTION RESULT:")
        if is_abnormal:
            print(f"🚨 {sound_type}")
            print(f"   Confidence: {confidence:.1%}")
            
            if abnormality_score > 70:
                print(f"⚠️  HIGH RISK - Immediate attention required!")
            elif abnormality_score > 50:
                print(f"⚠️  Medium risk - Investigate further")
        else:
            print(f"✅ NORMAL SOUND")
            print(f"   Confidence: {confidence:.1%}")
        
        return {
            'is_abnormal': is_abnormal,
            'sound_type': sound_type,
            'confidence': confidence,
            'abnormality_score': abnormality_score,
            'features': features
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    
    # Get audio file path
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("\n📁 Enter audio file path: ").strip()
    
    # Clean path
    audio_path = audio_path.strip('"').strip("'").replace('\\', '/')
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    # Ask which method to use
    print("\n🎯 Choose detection method:")
    print("  1. Advanced transformers (requires internet)")
    print("  2. Simple but accurate (local, no internet)")
    
    choice = input("\nSelect (1 or 2): ").strip()
    
    if choice == '1':
        try:
            detector = AdvancedAbnormalDetector()
            result = detector.analyze_audio_file(audio_path)
        except:
            print("❌ Could not load transformer models")
            print("Falling back to simple method...")
            result = simple_accurate_detector(audio_path)
    else:
        result = simple_accurate_detector(audio_path)
    
    if result and result['is_abnormal']:
        print(f"\n⚠️  RECOMMENDED ACTION:")
        print("   - Save recording for evidence")
        print("   - Alert authorities if high risk")
        print("   - Review security footage")
        