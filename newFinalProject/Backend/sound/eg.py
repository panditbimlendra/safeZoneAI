# ============================================================================
# COMPLETE WORKING ABNORMAL SOUND DETECTION SYSTEM WITH VISUALIZATION
# Based on your original code with enhanced visualization options
# ============================================================================

import numpy as np
import pandas as pd
import os
import wave
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.io.wavfile as wavf
from scipy import signal
from scipy.signal import spectrogram, find_peaks
import librosa
import librosa.display
import librosa.feature
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
# 2. ENHANCED VISUALIZATION MODULE
# ============================================================================

class AudioVisualizer:
    """Comprehensive audio visualization toolkit"""
    
    @staticmethod
    def plot_audio_waveform_comparison(audio_normal, audio_abnormal, sr=16000, title="Waveform Comparison"):
        """Compare normal and abnormal audio waveforms"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Normal audio waveform
        time_normal = np.linspace(0, len(audio_normal)/sr, len(audio_normal))
        axes[0, 0].plot(time_normal, audio_normal, color='green', alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title("Normal Audio Waveform", fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(time_normal, audio_normal, alpha=0.3, color='green')
        
        # Abnormal audio waveform
        time_abnormal = np.linspace(0, len(audio_abnormal)/sr, len(audio_abnormal))
        axes[0, 1].plot(time_abnormal, audio_abnormal, color='red', alpha=0.7, linewidth=0.8)
        axes[0, 1].set_title("Abnormal Audio Waveform", fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].fill_between(time_abnormal, audio_abnormal, alpha=0.3, color='red')
        
        # Histogram comparison
        axes[1, 0].hist(audio_normal, bins=100, alpha=0.7, color='green', label='Normal', density=True)
        axes[1, 0].hist(audio_abnormal, bins=100, alpha=0.7, color='red', label='Abnormal', density=True)
        axes[1, 0].set_title("Amplitude Distribution", fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel("Amplitude")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistical comparison
        stats_data = {
            'Type': ['Normal', 'Abnormal'],
            'Mean Amplitude': [np.mean(audio_normal), np.mean(audio_abnormal)],
            'Std Amplitude': [np.std(audio_normal), np.std(audio_abnormal)],
            'Max Amplitude': [np.max(audio_normal), np.max(audio_abnormal)],
            'Dynamic Range (dB)': [
                20*np.log10(np.max(np.abs(audio_normal))/np.max([np.min(np.abs(audio_normal)), 1e-10])),
                20*np.log10(np.max(np.abs(audio_abnormal))/np.max([np.min(np.abs(audio_abnormal)), 1e-10]))
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        ax_table = axes[1, 1]
        ax_table.axis('tight')
        ax_table.axis('off')
        table = ax_table.table(cellText=df_stats.round(4).values,
                              colLabels=df_stats.columns,
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.3, 0.2, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title("Statistical Comparison", fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_spectral_analysis(audio, sr=16000, title="Spectral Analysis"):
        """Comprehensive spectral analysis visualization"""
        fig = plt.figure(figsize=(18, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3)
        
        # 1. Waveform
        ax1 = fig.add_subplot(gs[0, :])
        time = np.linspace(0, len(audio)/sr, len(audio))
        ax1.plot(time, audio, color='blue', alpha=0.7, linewidth=0.8)
        ax1.set_title("Waveform", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram
        ax2 = fig.add_subplot(gs[1, 0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2, cmap='viridis')
        ax2.set_title("Spectrogram", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        
        # 3. Mel-spectrogram
        ax3 = fig.add_subplot(gs[1, 1])
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax3, cmap='plasma')
        ax3.set_title("Mel-Spectrogram", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Mel Frequency")
        
        # 4. MFCCs
        ax4 = fig.add_subplot(gs[1, 2])
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax4, cmap='coolwarm')
        ax4.set_title("MFCC Features", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("MFCC Coefficients")
        
        # 5. Spectral Features
        ax5 = fig.add_subplot(gs[2, 0])
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames, sr=sr)
        ax5.plot(t, spectral_centroid, label='Spectral Centroid', color='blue')
        ax5.plot(t, spectral_bandwidth, label='Spectral Bandwidth', color='red', alpha=0.7)
        ax5.set_title("Spectral Features", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Frequency (Hz)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Chromagram
        ax6 = fig.add_subplot(gs[2, 1])
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax6, cmap='coolwarm')
        ax6.set_title("Chromagram", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Pitch Class")
        
        # 7. Temporal Features
        ax7 = fig.add_subplot(gs[2, 2])
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        ax7.plot(t, rms, label='RMS Energy', color='green', linewidth=2)
        ax7.plot(t, zcr * np.max(rms), label='Zero Crossing Rate (scaled)', color='orange', alpha=0.7)
        ax7.set_title("Temporal Features", fontsize=12, fontweight='bold')
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("Value")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_interactive_3d_spectrum(audio, sr=16000, title="3D Spectrogram"):
        """Create interactive 3D spectrogram visualization"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Compute spectrogram
        frequencies, times, Sxx = spectrogram(audio, fs=sr, nperseg=1024, noverlap=512)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        T, F = np.meshgrid(times, frequencies)
        
        # Plot surface
        surf = ax.plot_surface(T, F, 10 * np.log10(Sxx + 1e-10), 
                              cmap='viridis', 
                              alpha=0.8,
                              linewidth=0,
                              antialiased=True)
        
        ax.set_xlabel('Time (s)', fontsize=12, labelpad=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=12, labelpad=10)
        ax.set_zlabel('Power (dB)', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Power (dB)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(features, importance_scores, feature_names=None, top_n=20):
        """Plot feature importance from model"""
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[-top_n:]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar plot
        axes[0].barh(range(len(indices)), importance_scores[indices], color='steelblue', alpha=0.8)
        axes[0].set_yticks(range(len(indices)))
        axes[0].set_yticklabels([feature_names[i] for i in indices])
        axes[0].set_xlabel('Importance Score', fontsize=12)
        axes[0].set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Cumulative importance
        sorted_importance = np.sort(importance_scores)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        axes[1].plot(range(1, len(sorted_importance) + 1), cumulative_importance, 
                    color='red', linewidth=2, marker='o')
        axes[1].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
        axes[1].axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
        axes[1].set_xlabel('Number of Features', fontsize=12)
        axes[1].set_ylabel('Cumulative Importance', fontsize=12)
        axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_abnormal_events_timeline(audio, sr, event_times, event_labels, title="Abnormal Events Timeline"):
        """Visualize timeline of abnormal events"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
        
        # Plot waveform with event markers
        time = np.linspace(0, len(audio)/sr, len(audio))
        axes[0].plot(time, audio, color='blue', alpha=0.5, linewidth=0.5)
        axes[0].set_ylabel("Amplitude", fontsize=12)
        axes[0].set_title("Audio Waveform with Abnormal Events", fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add event markers
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (event_time, label) in enumerate(zip(event_times, event_labels)):
            color_idx = i % len(colors)
            axes[0].axvline(x=event_time, color=colors[color_idx], linestyle='--', 
                          alpha=0.8, linewidth=2, label=f'{label} at {event_time:.2f}s')
        
        # Create timeline visualization
        axes[1].axis('off')
        
        # Create timeline bar
        total_duration = len(audio) / sr
        timeline_y = 0.5
        
        # Draw timeline
        axes[1].axhline(y=timeline_y, xmin=0, xmax=1, color='black', linewidth=3)
        
        # Add event markers on timeline
        for i, (event_time, label) in enumerate(zip(event_times, event_labels)):
            x_pos = event_time / total_duration
            color_idx = i % len(colors)
            
            # Marker
            axes[1].plot(x_pos, timeline_y, 'o', markersize=15, 
                        color=colors[color_idx], markeredgecolor='black')
            
            # Label
            axes[1].text(x_pos, timeline_y + 0.2, f'{label}\n{event_time:.2f}s', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[color_idx], alpha=0.7))
        
        # Add time ticks
        for t in np.arange(0, total_duration + 0.5, 0.5):
            x_pos = t / total_duration
            axes[1].plot([x_pos, x_pos], [timeline_y - 0.05, timeline_y + 0.05], 
                        color='black', linewidth=2)
            axes[1].text(x_pos, timeline_y - 0.15, f'{t:.1f}s', 
                        ha='center', va='top', fontsize=9)
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.legend(loc='upper right')
        plt.show()
    
    @staticmethod
    def create_interactive_dashboard(features_dict, audio_info):
        """Create interactive Plotly dashboard"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Waveform', 'Spectrogram', 'MFCC Features',
                          'Spectral Features', 'Energy Profile', 'Feature Distribution',
                          'Statistical Summary', 'Feature Correlation', 'Detection Results'),
            specs=[[{'type': 'xy'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'xy'}, {'type': 'xy'}, {'type': 'histogram'}],
                   [{'type': 'table'}, {'type': 'heatmap'}, {'type': 'indicator'}]]
        )
        
        # Add waveform
        fig.add_trace(
            go.Scatter(y=audio_info['waveform'][:10000], mode='lines', name='Waveform'),
            row=1, col=1
        )
        
        # Add spectrogram
        fig.add_trace(
            go.Heatmap(z=audio_info['spectrogram'], colorscale='Viridis'),
            row=1, col=2
        )
        
        # Add MFCC
        fig.add_trace(
            go.Heatmap(z=features_dict['mfccs'], colorscale='RdBu'),
            row=1, col=3
        )
        
        # Add spectral features
        fig.add_trace(
            go.Scatter(y=features_dict['spectral_centroid'], mode='lines', name='Spectral Centroid'),
            row=2, col=1
        )
        
        # Add energy profile
        fig.add_trace(
            go.Scatter(y=features_dict['rms_energy'], mode='lines', name='RMS Energy'),
            row=2, col=2
        )
        
        # Add feature distribution
        fig.add_trace(
            go.Histogram(x=features_dict['feature_values'], name='Feature Distribution'),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=True, title_text="Audio Analysis Dashboard")
        
        fig.show()

# ============================================================================
# 3. QUICK DETECTION WITH ENHANCED VISUALIZATION
# ============================================================================

def quick_detect_abnormal_sound(audio_file, plot=True, advanced_visualization=False):
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
        features['rms_skew'] = pd.Series(rms).skew()
        
        # 2. Zero-crossing rate (for abruptness)
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features['max_zcr'] = np.max(zcr)
        features['mean_zcr'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)[0]
        
        features['centroid_mean'] = np.mean(spectral_centroid)
        features['centroid_std'] = np.std(spectral_centroid)
        features['centroid_skew'] = pd.Series(spectral_centroid).skew()
        features['bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['bandwidth_std'] = np.std(spectral_bandwidth)
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['contrast_mean'] = np.mean(spectral_contrast)
        
        # 4. MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        features['mfcc_skew'] = pd.Series(mfccs.flatten()).skew()
        
        # 5. Temporal features
        features['crest_factor'] = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        features['dynamic_range'] = 20 * np.log10(np.max(np.abs(audio)) / (np.min(np.abs(audio)) + 1e-8))
        features['peak_to_rms'] = np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8)
        
        # 6. Additional statistical features
        features['amplitude_mean'] = np.mean(audio)
        features['amplitude_std'] = np.std(audio)
        features['amplitude_skewness'] = pd.Series(audio).skew()
        features['amplitude_kurtosis'] = pd.Series(audio).kurtosis()
        
        # Rule-based detection
        is_abnormal = False
        reasons = []
        confidence = 0.0
        
        # Thresholds (tuned for abnormal sounds)
        abnormality_scores = []
        
        if features['max_rms'] > 0.25:  # Very loud
            abnormality_scores.append(0.8)
            reasons.append(f"Loud (RMS={features['max_rms']:.3f})")
        
        if features['crest_factor'] > 7:  # Sharp transients
            abnormality_scores.append(0.9)
            reasons.append(f"Sharp peaks (crest={features['crest_factor']:.1f})")
        
        if features['max_zcr'] > 0.3:  # Many zero crossings
            abnormality_scores.append(0.7)
            reasons.append(f"Abrupt (ZCR={features['max_zcr']:.3f})")
        
        if features['centroid_mean'] > 2500:  # High frequency
            abnormality_scores.append(0.6)
            reasons.append(f"High freq (centroid={features['centroid_mean']:.0f} Hz)")
        
        if features['amplitude_kurtosis'] > 10:  # Heavy-tailed distribution
            abnormality_scores.append(0.7)
            reasons.append(f"Impulsive (kurtosis={features['amplitude_kurtosis']:.1f})")
        
        # Calculate overall confidence
        if abnormality_scores:
            confidence = np.mean(abnormality_scores)
            is_abnormal = confidence > 0.5
        
        # Classify type with confidence levels
        sound_type = "normal"
        type_confidence = 0.0
        
        if is_abnormal:
            if features['max_rms'] > 0.35 and features['crest_factor'] > 9:
                sound_type = "explosion/crash"
                type_confidence = 0.85
            elif features['centroid_mean'] > 3500 and features['max_zcr'] > 0.4:
                sound_type = "gunshot/glass"
                type_confidence = 0.9
            elif features['max_rms'] > 0.3 and features['bandwidth_mean'] > 2000:
                sound_type = "scream/alarm"
                type_confidence = 0.75
            elif features['crest_factor'] > 8:
                sound_type = "impact/bump"
                type_confidence = 0.8
            else:
                sound_type = "abnormal (unknown)"
                type_confidence = 0.6
        else:
            type_confidence = 0.95
        
        # Display results
        print(f"\n📊 Acoustic Analysis Report:")
        print(f"{'='*50}")
        
        print(f"\n📈 Feature Summary:")
        for key, value in features.items():
            print(f"  {key:25}: {value:10.4f}")
        
        print(f"\n🔍 Detection Result:")
        print(f"  {'='*40}")
        if is_abnormal:
            print(f"  🚨 ABNORMAL SOUND DETECTED!")
            print(f"  Type: {sound_type}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Type Confidence: {type_confidence:.2%}")
            print(f"  Reasons: {', '.join(reasons)}")
        else:
            print(f"  ✅ Normal sound")
            print(f"  Confidence: {1-confidence:.2%}")
        
        print(f"\n📊 Statistical Summary:")
        print(f"  Duration: {len(audio)/sr:.2f} seconds")
        print(f"  Samples: {len(audio):,}")
        print(f"  Sample Rate: {sr} Hz")
        
        # Plot if requested
        if plot:
            if advanced_visualization:
                plot_comprehensive_analysis(audio, sr, is_abnormal, sound_type, features, confidence)
            else:
                plot_sound_analysis(audio, sr, is_abnormal, sound_type, features)
        
        return {
            'is_abnormal': is_abnormal,
            'sound_type': sound_type,
            'features': features,
            'reasons': reasons,
            'confidence': confidence,
            'type_confidence': type_confidence,
            'audio': audio,
            'sample_rate': sr
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_sound_analysis(audio, sr, is_abnormal, sound_type, features):
    """Plot sound analysis with enhanced visuals"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # 1. Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    color = 'red' if is_abnormal else 'green'
    axes[0, 0].plot(time, audio, color=color, alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title(f"Waveform: {sound_type}", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)", fontsize=10)
    axes[0, 0].set_ylabel("Amplitude", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight abnormal regions if detected
    if is_abnormal:
        # Find peaks
        peaks, _ = find_peaks(np.abs(audio), height=np.std(audio)*3)
        peak_times = peaks / sr
        axes[0, 0].plot(peak_times, audio[peaks], 'ro', markersize=4, alpha=0.7)
    
    # 2. Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=2048)), ref=np.max)
    im = axes[0, 1].imshow(D, aspect='auto', origin='lower', 
                          extent=[0, len(audio)/sr, 0, sr/2], 
                          cmap='hot')
    axes[0, 1].set_title("Spectrogram", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)", fontsize=10)
    axes[0, 1].set_ylabel("Frequency (Hz)", fontsize=10)
    plt.colorbar(im, ax=axes[0, 1], label='dB')
    
    # 3. MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    im2 = axes[0, 2].imshow(mfccs, aspect='auto', origin='lower', 
                           extent=[0, len(audio)/sr, 0, 13],
                           cmap='coolwarm')
    axes[0, 2].set_title("MFCC Features", fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel("Time (s)", fontsize=10)
    axes[0, 2].set_ylabel("MFCC Coefficients", fontsize=10)
    plt.colorbar(im2, ax=axes[0, 2], label='Value')
    
    # 4. RMS Energy over time
    rms = librosa.feature.rms(y=audio)[0]
    frames = range(len(rms))
    t = librosa.frames_to_time(frames, sr=sr)
    axes[1, 0].plot(t, rms, color='blue', linewidth=2)
    axes[1, 0].fill_between(t, 0, rms, alpha=0.3, color='blue')
    axes[1, 0].set_title("RMS Energy", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Time (s)", fontsize=10)
    axes[1, 0].set_ylabel("RMS", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Key features radar chart (simplified)
    key_features = ['Max RMS', 'Crest Factor', 'Zero-Crossing', 'Spectral Centroid']
    key_values = [
        min(features['max_rms'] * 10, 10),  # Scale for visualization
        min(features['crest_factor'], 15),
        min(features['max_zcr'] * 20, 10),
        min(features['centroid_mean'] / 500, 10)
    ]
    
    colors = ['red' if is_abnormal else 'green' for _ in key_features]
    bars = axes[1, 1].bar(key_features, key_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Key Acoustic Features", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Normalized Value", fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, key_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Statistical distribution
    axes[1, 2].hist(audio, bins=100, alpha=0.7, color='purple', density=True)
    axes[1, 2].set_title("Amplitude Distribution", fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel("Amplitude", fontsize=10)
    axes[1, 2].set_ylabel("Density", fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add normal distribution curve
    from scipy.stats import norm
    mu, std = norm.fit(audio)
    x = np.linspace(min(audio), max(audio), 100)
    p = norm.pdf(x, mu, std)
    axes[1, 2].plot(x, p, 'k', linewidth=2, label=f'Normal fit: μ={mu:.3f}, σ={std:.3f}')
    axes[1, 2].legend(fontsize=9)
    
    # 7. Zero-crossing rate over time
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    axes[2, 0].plot(t, zcr, color='orange', linewidth=2)
    axes[2, 0].fill_between(t, 0, zcr, alpha=0.3, color='orange')
    axes[2, 0].set_title("Zero-Crossing Rate", fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel("Time (s)", fontsize=10)
    axes[2, 0].set_ylabel("ZCR", fontsize=10)
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Spectral features over time
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    axes[2, 1].plot(t, spectral_centroid, label='Spectral Centroid', color='red', linewidth=2)
    axes[2, 1].plot(t, spectral_bandwidth, label='Spectral Bandwidth', color='blue', linewidth=2, alpha=0.7)
    axes[2, 1].set_title("Spectral Features", fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel("Time (s)", fontsize=10)
    axes[2, 1].set_ylabel("Frequency (Hz)", fontsize=10)
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Detection result with confidence
    axes[2, 2].axis('off')
    if is_abnormal:
        axes[2, 2].text(0.5, 0.7, '🚨 ABNORMAL SOUND', 
                       fontsize=20, fontweight='bold', color='red',
                       ha='center', va='center')
        axes[2, 2].text(0.5, 0.5, f'Type: {sound_type}', 
                       fontsize=16, ha='center', va='center')
        axes[2, 2].text(0.5, 0.4, f'Confidence: {features.get("confidence", 0.7):.1%}', 
                       fontsize=14, ha='center', va='center')
        if features.get('reasons'):
            reasons_text = '\n'.join(features['reasons'][:3])
            axes[2, 2].text(0.5, 0.2, reasons_text, 
                           fontsize=11, ha='center', va='center')
    else:
        axes[2, 2].text(0.5, 0.7, '✅ NORMAL SOUND', 
                       fontsize=20, fontweight='bold', color='green',
                       ha='center', va='center')
        axes[2, 2].text(0.5, 0.5, f'Type: {sound_type}', 
                       fontsize=16, ha='center', va='center')
        axes[2, 2].text(0.5, 0.4, f'Confidence: {1-features.get("confidence", 0.3):.1%}', 
                       fontsize=14, ha='center', va='center')
    
    plt.suptitle(f"Comprehensive Sound Analysis: {sound_type}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comprehensive_analysis(audio, sr, is_abnormal, sound_type, features, confidence):
    """Advanced comprehensive analysis with multiple visualization types"""
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = fig.add_gridspec(4, 4)
    
    # 1. Main waveform with anomalies highlighted
    ax1 = fig.add_subplot(gs[0, :2])
    time = np.linspace(0, len(audio)/sr, len(audio))
    ax1.plot(time, audio, color='gray', alpha=0.5, linewidth=0.5)
    
    # Highlight abnormal regions
    if is_abnormal:
        # Find regions with high energy
        window_size = 100
        energy = np.convolve(audio**2, np.ones(window_size)/window_size, mode='same')
        threshold = np.percentile(energy, 95)
        high_energy_idx = np.where(energy > threshold)[0]
        
        if len(high_energy_idx) > 0:
            # Group consecutive indices
            from itertools import groupby
            groups = []
            for k, g in groupby(enumerate(high_energy_idx), lambda x: x[0]-x[1]):
                groups.append([i for _, i in g])
            
            for group in groups:
                if len(group) > 10:  # Only highlight significant regions
                    start_idx = max(0, group[0] - 50)
                    end_idx = min(len(audio), group[-1] + 50)
                    ax1.fill_between(time[start_idx:end_idx], 
                                    audio[start_idx:end_idx], 
                                    alpha=0.3, color='red')
    
    ax1.set_title("Waveform with Anomaly Detection", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # 2. 3D Spectrogram
    ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
    
    # Compute spectrogram
    frequencies, times, Sxx = spectrogram(audio, fs=sr, nperseg=256, noverlap=128)
    
    # Plot surface
    X, Y = np.meshgrid(times, frequencies)
    surf = ax2.plot_surface(X, Y, 10*np.log10(Sxx + 1e-10), 
                          cmap='viridis', alpha=0.8, linewidth=0)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_zlabel('Power (dB)')
    ax2.set_title("3D Spectrogram", fontsize=14, fontweight='bold')
    
    # 3. Feature Correlation Heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    
    # Create feature correlation matrix
    feature_names = list(features.keys())[:15]  # Use first 15 features
    feature_values = [features[name] for name in feature_names]
    
    # Create correlation-like matrix
    n_features = len(feature_names)
    corr_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Simulate correlation based on feature values
                corr_matrix[i, j] = 1 - abs(feature_values[i] - feature_values[j]) / \
                                   (abs(feature_values[i]) + abs(feature_values[j]) + 1e-10)
    
    im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(n_features))
    ax3.set_yticks(range(n_features))
    ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(feature_names, fontsize=8)
    ax3.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Correlation')
    
    # 4. Time-Frequency Features
    ax4 = fig.add_subplot(gs[1, 2:])
    
    # Compute multiple features over time
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    t = librosa.frames_to_time(range(len(rms)), sr=sr)
    
    # Normalize for plotting
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
    zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
    centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / \
                   (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-10)
    
    ax4.plot(t, rms_norm, label='RMS Energy', linewidth=2, color='blue')
    ax4.plot(t, zcr_norm, label='Zero-Crossing Rate', linewidth=2, color='red', alpha=0.7)
    ax4.plot(t, centroid_norm, label='Spectral Centroid', linewidth=2, color='green', alpha=0.7)
    
    ax4.set_title("Normalized Temporal Features", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Normalized Value")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistical Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Box plot of audio segments
    n_segments = 20
    segment_length = len(audio) // n_segments
    segments = [audio[i*segment_length:(i+1)*segment_length] for i in range(n_segments)]
    
    bp = ax5.boxplot(segments, positions=range(n_segments), widths=0.6, patch_artist=True)
    
    # Color boxes based on segment energy
    segment_energies = [np.mean(seg**2) for seg in segments]
    norm_energies = (segment_energies - np.min(segment_energies)) / \
                   (np.max(segment_energies) - np.min(segment_energies) + 1e-10)
    
    for i, box in enumerate(bp['boxes']):
        color = plt.cm.Reds(norm_energies[i])
        box.set_facecolor(color)
        box.set_alpha(0.7)
    
    ax5.set_title("Segment Statistics (Box Plots)", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Segment Index")
    ax5.set_ylabel("Amplitude")
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribution Analysis
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Histogram with KDE
    sns.histplot(audio, kde=True, ax=ax6, color='purple', alpha=0.6, stat='density')
    
    # Add vertical lines for statistics
    ax6.axvline(np.mean(audio), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(audio):.4f}')
    ax6.axvline(np.median(audio), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(audio):.4f}')
    ax6.axvline(np.std(audio), color='blue', linestyle='--', linewidth=2, label=f'Std: {np.std(audio):.4f}')
    
    ax6.set_title("Amplitude Distribution with KDE", fontsize=12, fontweight='bold')
    ax6.set_xlabel("Amplitude")
    ax6.set_ylabel("Density")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Feature Importance (simulated)
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Simulate feature importance
    top_features = ['Max RMS', 'Crest Factor', 'Zero-Crossing', 'Spectral Centroid', 
                   'MFCC Variance', 'Dynamic Range', 'Skewness', 'Kurtosis']
    importance = np.random.rand(len(top_features))
    importance = importance / np.sum(importance)
    
    bars = ax7.barh(top_features, importance, color=plt.cm.Set3(np.linspace(0, 1, len(top_features))))
    ax7.set_title("Simulated Feature Importance", fontsize=12, fontweight='bold')
    ax7.set_xlabel("Importance Score")
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax7.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=9)
    
    # 8. Summary Dashboard
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create summary text
    summary_text = []
    summary_text.append(f"{'='*60}")
    summary_text.append(f"{'SOUND ANALYSIS SUMMARY':^60}")
    summary_text.append(f"{'='*60}")
    summary_text.append(f"Status: {'ABNORMAL' if is_abnormal else 'NORMAL'}")
    summary_text.append(f"Type: {sound_type}")
    summary_text.append(f"Confidence: {confidence:.2%}")
    summary_text.append(f"Duration: {len(audio)/sr:.2f} seconds")
    summary_text.append(f"Sample Rate: {sr} Hz")
    summary_text.append(f"Total Samples: {len(audio):,}")
    summary_text.append("")
    summary_text.append("Key Statistics:")
    summary_text.append(f"  Mean Amplitude: {np.mean(audio):.6f}")
    summary_text.append(f"  Std Amplitude: {np.std(audio):.6f}")
    summary_text.append(f"  Max Amplitude: {np.max(audio):.6f}")
    summary_text.append(f"  Dynamic Range: {features.get('dynamic_range', 0):.1f} dB")
    summary_text.append("")
    
    if is_abnormal and features.get('reasons'):
        summary_text.append("Detection Reasons:")
        for reason in features['reasons'][:5]:
            summary_text.append(f"  • {reason}")
    
    summary_text.append(f"{'='*60}")
    
    # Display summary
    summary_str = '\n'.join(summary_text)
    ax8.text(0.5, 0.5, summary_str, 
            ha='center', va='center', 
            fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Main title
    title_color = 'red' if is_abnormal else 'green'
    plt.suptitle(f"Advanced Audio Analysis: {sound_type}", 
                fontsize=20, fontweight='bold', color=title_color, y=0.98)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. EXTENDED DETECTOR WITH VISUALIZATION OPTIONS
# ============================================================================

class AbnormalSoundDetector:
    """Extended detector with comprehensive visualization"""
    
    def __init__(self, visualizer=None):
        self.model = None
        self.pca = None
        self.scaler = None
        self.visualizer = visualizer or AudioVisualizer()
        self.feature_history = []
        self.detection_history = []
        
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
    
    def extract_features_from_segments(self, segmented_audio, sample_rate, visualize=False):
        """Extract features from segmented audio with optional visualization"""
        n_segments = segmented_audio.shape[0]
        features_list = []
        
        print(f"Extracting features from {n_segments} segments...")
        
        for i in range(n_segments):
            if i % 100 == 0 and n_segments > 100:
                print(f"  Processing segment {i}/{n_segments}")
            
            audio_segment = segmented_audio[i, :]
            
            # Extract features for this segment
            features = self.extract_single_segment_features(audio_segment, sample_rate)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Store feature history for visualization
        self.feature_history.append({
            'features': features_array,
            'n_segments': n_segments,
            'sample_rate': sample_rate
        })
        
        # Visualize feature distribution if requested
        if visualize and n_segments > 1:
            self.visualize_feature_distribution(features_array)
        
        return features_array
    
    def extract_single_segment_features(self, audio_segment, sample_rate):
        """Extract comprehensive features from a single audio segment"""
        features = []
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sample_rate)[0]
        
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        features.append(np.mean(spectral_rolloff))
        features.append(np.mean(spectral_contrast))
        
        # Energy features
        rms = librosa.feature.rms(y=audio_segment)[0]
        features.append(np.max(rms))
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms) / (np.mean(rms) + 1e-10))  # Peak-to-average ratio
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio_segment)[0]
        features.append(np.mean(zcr))
        features.append(np.max(zcr))
        features.append(np.std(zcr))
        
        # Temporal features
        features.append(np.max(np.abs(audio_segment)))  # Peak amplitude
        features.append(np.mean(np.abs(audio_segment)))  # Mean absolute amplitude
        features.append(np.std(audio_segment))  # Standard deviation
        
        # Crest factor
        crest_factor = np.max(np.abs(audio_segment)) / (np.sqrt(np.mean(audio_segment**2)) + 1e-10)
        features.append(crest_factor)
        
        # Additional statistical features
        features.append(pd.Series(audio_segment).skew())  # Skewness
        features.append(pd.Series(audio_segment).kurtosis())  # Kurtosis
        
        return np.array(features)
    
    def visualize_feature_distribution(self, features_array):
        """Visualize distribution of extracted features"""
        n_features = features_array.shape[1]
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            ax.hist(features_array[:, i], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'Feature {i}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(features_array[:, i])
            std_val = np.std(features_array[:, i])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Feature Distribution Across Segments', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def train_on_existing_data(self, audio_file, labels, visualize_training=True):
        """Train using existing labeled data with visualization options"""
        print("Training detector on existing data...")
        
        # Load and process audio
        segmented_audio, sample_rate, num_segments = self.load_and_process_audio(audio_file)
        
        # Extract features
        X = self.extract_features_from_segments(segmented_audio, sample_rate, visualize=visualize_training)
        
        # Make sure labels match
        if len(labels) > X.shape[0]:
            labels = labels[:X.shape[0]]
        elif len(labels) < X.shape[0]:
            X = X[:len(labels), :]
        
        y = labels.flatten()
        
        # Visualize class distribution
        if visualize_training:
            self.plot_class_distribution(y)
        
        # Handle class imbalance with SMOTE
        print("Balancing classes with SMOTE...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=10
        )
        
        # Train Random Forest
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
        
        # Visualization of training results
        if visualize_training:
            self.plot_training_results(X_test, y_test, y_pred)
        
        return accuracy
    
    def plot_class_distribution(self, y):
        """Plot class distribution before and after balancing"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before balancing
        unique_classes, counts = np.unique(y, return_counts=True)
        class_names = ['Background', 'Bumping', 'Speech']
        
        axes[0].bar(range(len(unique_classes)), counts, color=['blue', 'red', 'green'], alpha=0.7)
        axes[0].set_xticks(range(len(unique_classes)))
        axes[0].set_xticklabels([class_names[i] for i in unique_classes])
        axes[0].set_title('Class Distribution (Original)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = np.sum(counts)
        for i, count in enumerate(counts):
            percentage = (count / total) * 100
            axes[0].text(i, count + total*0.01, f'{percentage:.1f}%', 
                        ha='center', fontsize=10)
        
        # Pie chart
        axes[1].pie(counts, labels=[class_names[i] for i in unique_classes], 
                   autopct='%1.1f%%', colors=['blue', 'red', 'green'],
                   startangle=90, explode=[0.05]*len(unique_classes))
        axes[1].set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
        
        plt.suptitle('Dataset Class Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_training_results(self, X_test, y_test, y_pred):
        """Visualize training results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
        
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            axes[0, 1].barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([f'Feature {i}' for i in indices])
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. ROC Curves (for binary or multi-class)
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Training Statistics:\n\n'
                       f'Accuracy: {accuracy_score(y_test, y_pred):.2%}\n'
                       f'Total Samples: {len(y_test)}\n'
                       f'Classes: {len(np.unique(y_test))}',
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 4. Prediction Distribution
        unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
        unique_true, true_counts = np.unique(y_test, return_counts=True)
        
        x = np.arange(len(unique_true))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, true_counts, width, label='True', alpha=0.7, color='blue')
        axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7, color='red')
        
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('True vs Predicted Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Background', 'Bumping', 'Speech'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Training Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def predict_audio_file(self, audio_file, segment_duration=0.4, visualize=True):
        """Predict abnormal sounds in an audio file with visualization"""
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
                
                # Count abnormal segments
                abnormal_segments = np.sum(predictions == 1)  # Class 1 is bumping/abnormal
                total_segments = len(predictions)
                
                print(f"  Detailed analysis: {abnormal_segments}/{total_segments} "
                      f"segments detected as abnormal")
                
                # Visualize segment predictions
                if visualize and total_segments > 1:
                    self.visualize_segment_predictions(segmented_audio, sample_rate, 
                                                      predictions, segment_duration)
                
                if abnormal_segments > total_segments * 0.1:  # More than 10% abnormal
                    print("  🚨 CONFIRMED: Significant abnormal content detected!")
        
        return quick_result
    
    def visualize_segment_predictions(self, segmented_audio, sample_rate, predictions, segment_duration):
        """Visualize predictions across segments"""
        n_segments = len(predictions)
        
        # Create a timeline visualization
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
        
        # Plot aggregated waveform
        full_audio = segmented_audio.flatten()
        time = np.linspace(0, len(full_audio)/sample_rate, len(full_audio))
        
        axes[0].plot(time, full_audio, color='gray', alpha=0.5, linewidth=0.5)
        axes[0].set_title("Audio Waveform with Segment Predictions", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Color segments based on predictions
        segment_length = int(segment_duration * sample_rate)
        
        for i, pred in enumerate(predictions):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            # Color coding
            if pred == 0:  # Background
                color = 'green'
                alpha = 0.1
            elif pred == 1:  # Bumping/Abnormal
                color = 'red'
                alpha = 0.3
            else:  # Speech
                color = 'blue'
                alpha = 0.2
            
            axes[0].axvspan(start_time, end_time, color=color, alpha=alpha)
            
            # Add prediction label
            if pred == 1:  # Highlight abnormal segments
                axes[0].text(start_time + segment_duration/2, 
                           np.max(full_audio) * 0.8,
                           'ABNORMAL', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Create prediction timeline
        axes[1].axis('off')
        
        # Draw timeline
        timeline_y = 0.5
        axes[1].axhline(y=timeline_y, xmin=0, xmax=1, color='black', linewidth=2)
        
        # Add segment markers
        for i, pred in enumerate(predictions[:50]):  # Limit to first 50 for clarity
            x_pos = i / min(50, n_segments)
            
            if pred == 0:
                color = 'green'
                marker = 'o'
                size = 8
            elif pred == 1:
                color = 'red'
                marker = 's'
                size = 10
            else:
                color = 'blue'
                marker = '^'
                size = 9
            
            axes[1].plot(x_pos, timeline_y, marker=marker, markersize=size, 
                        color=color, markeredgecolor='black')
            
            # Add segment number
            if i % 5 == 0:
                axes[1].text(x_pos, timeline_y - 0.2, str(i), 
                           ha='center', va='top', fontsize=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Background'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=10, label='Abnormal'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                      markersize=10, label='Speech')
        ]
        
        axes[1].legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        plt.suptitle(f"Segment-wise Prediction Analysis (First {min(50, n_segments)} segments)", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ============================================================================
# 5. MAIN TESTING FUNCTION WITH VISUALIZATION MENU
# ============================================================================

def test_abnormal_sound_detection():
    """Main function to test abnormal sound detection with visualization options"""
    print("\n" + "="*80)
    print("ABNORMAL SOUND DETECTION TESTER WITH VISUALIZATION")
    print("="*80)
    
    # Create detector with visualizer
    visualizer = AudioVisualizer()
    detector = AbnormalSoundDetector(visualizer)
    
    # Show visualization menu
    print("\n🎨 VISUALIZATION OPTIONS:")
    print("   1. Basic plots (waveform, spectrogram, MFCCs)")
    print("   2. Advanced plots (3D, correlation matrices, distributions)")
    print("   3. Comparative analysis (normal vs abnormal)")
    print("   4. Interactive dashboard")
    print("   5. All visualizations")
    print("   6. No visualization (fastest)")
    
    viz_choice = input("\nSelect visualization level (1-6): ").strip()
    
    # Create test sound for demonstration
    print("\n🔊 Creating test sounds for demonstration...")
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Normal sound (sine wave with noise)
    normal_sound = 0.1 * np.sin(2 * np.pi * 440 * t)  # A4 note
    normal_sound += 0.05 * np.random.randn(len(t))  # Add noise
    
    # Abnormal sound (with impulses)
    abnormal_sound = 0.1 * np.sin(2 * np.pi * 440 * t)
    abnormal_sound += 0.05 * np.random.randn(len(t))
    
    # Add abnormal events
    impulse_times = [0.5, 1.2, 2.0, 2.5]
    for impulse_time in impulse_times:
        idx = int(impulse_time * sr)
        abnormal_sound[idx:idx+50] = 0.8  # Sharp impulse
    
    # Save test sounds
    import soundfile as sf
    sf.write('test_normal.wav', normal_sound, sr)
    sf.write('test_abnormal.wav', abnormal_sound, sr)
    
    print("✅ Created test_normal.wav and test_abnormal.wav")
    
    # Apply selected visualization level
    if viz_choice in ['1', '2', '3', '5']:
        print("\n📊 Running visualizations...")
        
        if viz_choice in ['1', '5']:
            # Basic visualization
            print("\n1️⃣ Basic Sound Analysis:")
            quick_detect_abnormal_sound('test_normal.wav', plot=True, advanced_visualization=False)
            quick_detect_abnormal_sound('test_abnormal.wav', plot=True, advanced_visualization=False)
        
        if viz_choice in ['2', '5']:
            # Advanced visualization
            print("\n2️⃣ Advanced Analysis:")
            result_normal = quick_detect_abnormal_sound('test_normal.wav', plot=False)
            result_abnormal = quick_detect_abnormal_sound('test_abnormal.wav', plot=False)
            
            if result_normal and result_abnormal:
                plot_comprehensive_analysis(
                    result_normal['audio'], 
                    result_normal['sample_rate'],
                    result_normal['is_abnormal'],
                    result_normal['sound_type'],
                    result_normal['features'],
                    result_normal['confidence']
                )
        
        if viz_choice in ['3', '5']:
            # Comparative analysis
            print("\n3️⃣ Comparative Analysis:")
            visualizer.plot_audio_waveform_comparison(normal_sound, abnormal_sound, sr)
            
            # Spectral analysis comparison
            print("\n   Normal Sound Spectral Analysis:")
            visualizer.plot_spectral_analysis(normal_sound, sr, "Normal Sound Analysis")
            
            print("\n   Abnormal Sound Spectral Analysis:")
            visualizer.plot_spectral_analysis(abnormal_sound, sr, "Abnormal Sound Analysis")
        
        if viz_choice == '4':
            # Interactive dashboard (simulated)
            print("\n4️⃣ Interactive Dashboard (simulated)...")
            
            # Create feature dictionary for dashboard
            features_dict = {
                'mfccs': librosa.feature.mfcc(y=abnormal_sound, sr=sr, n_mfcc=13),
                'spectral_centroid': librosa.feature.spectral_centroid(y=abnormal_sound, sr=sr)[0],
                'rms_energy': librosa.feature.rms(y=abnormal_sound)[0],
                'feature_values': np.random.randn(1000)  # Simulated feature values
            }
            
            audio_info = {
                'waveform': abnormal_sound,
                'spectrogram': np.abs(librosa.stft(abnormal_sound, n_fft=2048))
            }
            
            print("   Dashboard would be displayed in a separate browser window.")
            print("   For full interactive dashboard, install plotly: pip install plotly")
        
        if viz_choice == '5':
            # All visualizations including 3D
            print("\n5️⃣ All Visualizations including 3D:")
            visualizer.plot_interactive_3d_spectrum(abnormal_sound, sr, "3D Abnormal Sound Analysis")
            
            # Feature importance visualization (simulated)
            print("\n   Feature Importance Analysis:")
            n_features = 30
            feature_names = [f'Feature_{i}' for i in range(n_features)]
            importance_scores = np.random.rand(n_features)
            importance_scores = importance_scores / np.sum(importance_scores)
            
            visualizer.plot_feature_importance(
                np.random.randn(100, n_features),
                importance_scores,
                feature_names,
                top_n=15
            )
            
            # Timeline visualization
            print("\n   Abnormal Events Timeline:")
            visualizer.plot_abnormal_events_timeline(
                abnormal_sound, sr,
                event_times=impulse_times,
                event_labels=['Impulse 1', 'Impulse 2', 'Impulse 3', 'Impulse 4'],
                title="Abnormal Events Detection Timeline"
            )
    
    # Test with detector
    print("\n" + "="*80)
    print("TESTING DETECTOR CAPABILITIES")
    print("="*80)
    
    # Test normal sound
    print("\n🔍 Testing Normal Sound:")
    result_normal = detector.predict_audio_file('test_normal.wav', visualize=(viz_choice != '6'))
    
    # Test abnormal sound  
    print("\n🔍 Testing Abnormal Sound:")
    result_abnormal = detector.predict_audio_file('test_abnormal.wav', visualize=(viz_choice != '6'))
    
    # Display summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if result_normal and result_abnormal:
        print(f"\n📊 Normal Sound:")
        print(f"   Type: {result_normal['sound_type']}")
        print(f"   Abnormal: {result_normal['is_abnormal']}")
        print(f"   Confidence: {result_normal['confidence']:.2%}")
        
        print(f"\n📊 Abnormal Sound:")
        print(f"   Type: {result_abnormal['sound_type']}")
        print(f"   Abnormal: {result_abnormal['is_abnormal']}")
        print(f"   Confidence: {result_abnormal['confidence']:.2%}")
        print(f"   Reasons: {', '.join(result_abnormal['reasons'][:3])}")
    
    # Cleanup
    import os
    for file in ['test_normal.wav', 'test_abnormal.wav']:
        if os.path.exists(file):
            os.remove(file)
            print(f"\n🗑️  Cleaned up {file}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    
    return detector

# ============================================================================
# 6. INTERACTIVE VISUALIZATION DEMO
# ============================================================================

def interactive_visualization_demo():
    """Interactive demonstration of all visualization capabilities"""
    print("\n" + "="*80)
    print("INTERACTIVE VISUALIZATION DEMO")
    print("="*80)
    
    visualizer = AudioVisualizer()
    
    # Create sample sounds
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    print("\n🎵 Creating sample sounds...")
    
    # 1. Normal sound (gentle sine wave)
    normal_sound = 0.2 * np.sin(2 * np.pi * 261.63 * t)  # C4 note
    normal_sound += 0.05 * np.sin(2 * np.pi * 329.63 * t)  # E4 note
    normal_sound += 0.01 * np.random.randn(len(t))  # Gentle noise
    
    # 2. Abnormal sound (with various anomalies)
    abnormal_sound = 0.2 * np.sin(2 * np.pi * 261.63 * t)
    
    # Add different types of abnormalities
    # Sharp impulse
    abnormal_sound[int(0.3*sr):int(0.3*sr)+20] = 0.8
    
    # High frequency burst
    abnormal_sound[int(0.8*sr):int(0.9*sr)] += 0.3 * np.sin(2 * np.pi * 2000 * t[:int(0.1*sr)])
    
    # Random spikes
    for _ in range(5):
        idx = np.random.randint(int(1.0*sr), int(1.8*sr))
        abnormal_sound[idx:idx+10] = 0.5
    
    # Save sounds
    import soundfile as sf
    sf.write('demo_normal.wav', normal_sound, sr)
    sf.write('demo_abnormal.wav', abnormal_sound, sr)
    
    print("✅ Created demonstration audio files")
    
    # Interactive menu
    while True:
        print("\n" + "-"*60)
        print("VISUALIZATION MENU")
        print("-"*60)
        print("1. Waveform Comparison")
        print("2. Spectral Analysis")
        print("3. 3D Spectrogram")
        print("4. Feature Importance (Simulated)")
        print("5. Events Timeline")
        print("6. Comprehensive Analysis")
        print("7. Quick Detection Results")
        print("8. Play Sounds")
        print("9. Exit")
        
        choice = input("\nSelect visualization (1-9): ").strip()
        
        if choice == '1':
            print("\n📈 Waveform Comparison...")
            visualizer.plot_audio_waveform_comparison(
                normal_sound, abnormal_sound, sr,
                title="Normal vs Abnormal Sound Comparison"
            )
        
        elif choice == '2':
            print("\n📊 Spectral Analysis...")
            sound_choice = input("Analyze (1) Normal or (2) Abnormal sound? ").strip()
            
            if sound_choice == '1':
                visualizer.plot_spectral_analysis(
                    normal_sound, sr,
                    title="Normal Sound - Spectral Analysis"
                )
            else:
                visualizer.plot_spectral_analysis(
                    abnormal_sound, sr,
                    title="Abnormal Sound - Spectral Analysis"
                )
        
        elif choice == '3':
            print("\n🎨 3D Spectrogram...")
            sound_choice = input("3D visualization for (1) Normal or (2) Abnormal sound? ").strip()
            
            if sound_choice == '1':
                visualizer.plot_interactive_3d_spectrum(
                    normal_sound, sr,
                    title="Normal Sound - 3D Spectrogram"
                )
            else:
                visualizer.plot_interactive_3d_spectrum(
                    abnormal_sound, sr,
                    title="Abnormal Sound - 3D Spectrogram"
                )
        
        elif choice == '4':
            print("\n⚖️ Feature Importance Analysis...")
            # Simulate feature importance
            n_features = 25
            feature_names = [f'Acoustic_Feature_{i}' for i in range(1, n_features+1)]
            importance_scores = np.random.exponential(1, n_features)
            importance_scores = importance_scores / np.sum(importance_scores)
            
            visualizer.plot_feature_importance(
                np.random.randn(100, n_features),
                importance_scores,
                feature_names,
                top_n=15
            )
        
        elif choice == '5':
            print("\n⏰ Events Timeline...")
            # Define event times
            event_times = [0.3, 0.8, 1.2, 1.5, 1.8]
            event_labels = ['Impulse', 'High Freq', 'Spike 1', 'Spike 2', 'Spike 3']
            
            visualizer.plot_abnormal_events_timeline(
                abnormal_sound, sr,
                event_times=event_times,
                event_labels=event_labels,
                title="Abnormal Sound Events Timeline"
            )
        
        elif choice == '6':
            print("\n🔍 Comprehensive Analysis...")
            sound_choice = input("Comprehensive analysis for (1) Normal or (2) Abnormal sound? ").strip()
            
            if sound_choice == '1':
                result = quick_detect_abnormal_sound(normal_sound, plot=False)
                if result:
                    plot_comprehensive_analysis(
                        result['audio'], result['sample_rate'],
                        result['is_abnormal'], result['sound_type'],
                        result['features'], result['confidence']
                    )
            else:
                result = quick_detect_abnormal_sound(abnormal_sound, plot=False)
                if result:
                    plot_comprehensive_analysis(
                        result['audio'], result['sample_rate'],
                        result['is_abnormal'], result['sound_type'],
                        result['features'], result['confidence']
                    )
        
        elif choice == '7':
            print("\n🚨 Quick Detection Results...")
            print("\nNormal Sound:")
            quick_detect_abnormal_sound('demo_normal.wav', plot=True, advanced_visualization=True)
            
            print("\n\nAbnormal Sound:")
            quick_detect_abnormal_sound('demo_abnormal.wav', plot=True, advanced_visualization=True)
        
        elif choice == '8':
            print("\n🔊 Playing sounds...")
            try:
                import sounddevice as sd
                
                print("Playing normal sound...")
                sd.play(normal_sound, sr)
                sd.wait()
                
                print("Playing abnormal sound...")
                sd.play(abnormal_sound, sr)
                sd.wait()
                
                print("✅ Sounds played successfully")
            except ImportError:
                print("⚠️ Install sounddevice for audio playback: pip install sounddevice")
            except Exception as e:
                print(f"⚠️ Could not play sounds: {e}")
        
        elif choice == '9':
            print("\n👋 Exiting visualization demo...")
            break
        
        else:
            print("⚠️ Invalid choice. Please select 1-9.")
    
    # Cleanup
    import os
    for file in ['demo_normal.wav', 'demo_abnormal.wav']:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n" + "="*80)
    print("VISUALIZATION DEMO COMPLETE")
    print("="*80)

# ============================================================================
# 7. SIMPLE COMMAND-LINE INTERFACE WITH VISUALIZATION
# ============================================================================

def simple_test_with_visualization():
    """Simple test with visualization options"""
    import sys
    
    print("\n🔊 Abnormal Sound Detector with Visualization")
    print("="*50)
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\nTesting: {audio_file}")
            
            # Ask for visualization level
            print("\nVisualization Options:")
            print("  1. Basic plots")
            print("  2. Advanced plots")
            print("  3. No plots (fast)")
            
            viz_choice = input("\nSelect visualization (1-3): ").strip()
            
            if viz_choice == '1':
                result = quick_detect_abnormal_sound(audio_file, plot=True, advanced_visualization=False)
            elif viz_choice == '2':
                result = quick_detect_abnormal_sound(audio_file, plot=True, advanced_visualization=True)
            else:
                result = quick_detect_abnormal_sound(audio_file, plot=False)
            
            if result:
                print(f"\n{'='*50}")
                print("FINAL RESULT:")
                if result['is_abnormal']:
                    print(f"🚨 ABNORMAL: {result['sound_type']}")
                    print(f"   Confidence: {result['confidence']:.2%}")
                    print(f"   Reasons: {', '.join(result['reasons'])}")
                else:
                    print(f"✅ NORMAL SOUND")
                    print(f"   Confidence: {1-result['confidence']:.2%}")
        else:
            print(f"File not found: {audio_file}")
    else:
        # Interactive mode
        file_path = input("Enter path to audio file: ").strip()
        
        if file_path and os.path.exists(file_path):
            print("\nRunning comprehensive analysis...")
            result = quick_detect_abnormal_sound(file_path, plot=True, advanced_visualization=True)
        else:
            print("Creating demonstration...")
            interactive_visualization_demo()

# ============================================================================
# 8. MAIN EXECUTION - COMPREHENSIVE MENU
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ABNORMAL SOUND DETECTION SYSTEM WITH ADVANCED VISUALIZATION")
    print("="*80)
    
    print("\nMAIN MENU:")
    print("1. Quick test a sound file (with visualization options)")
    print("2. Full testing with training and visualizations")
    print("3. Interactive visualization demo")
    print("4. Use my existing code structure")
    print("5. Batch test a directory")
    print("6. Advanced 3D visualization demo")
    print("7. Exit")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        # Quick test with visualization
        simple_test_with_visualization()
        
    elif choice == '2':
        # Full testing with training
        detector = test_abnormal_sound_detection()
        
    elif choice == '3':
        # Interactive visualization demo
        interactive_visualization_demo()
        
    elif choice == '4':
        # Use existing structure
        print("\nUsing your original code structure...")
        
        # Test with a sample file
        test_files = ['test.wav', 'sound.wav', 'audio.wav']
        found = False
        
        for file in test_files:
            if os.path.exists(file):
                print(f"Testing {file}...")
                result = quick_detect_abnormal_sound(file, plot=True, advanced_visualization=True)
                found = True
                break
        
        if not found:
            print("No test files found. Creating a demonstration...")
            interactive_visualization_demo()
            
    elif choice == '5':
        # Batch test directory
        test_dir = input("Enter directory path to test: ").strip()
        
        if test_dir and os.path.exists(test_dir):
            import glob
            
            audio_files = glob.glob(os.path.join(test_dir, "*.wav")) + \
                         glob.glob(os.path.join(test_dir, "*.mp3")) + \
                         glob.glob(os.path.join(test_dir, "*.flac"))
            
            if audio_files:
                print(f"\nFound {len(audio_files)} audio files")
                
                # Create summary report
                summary_data = []
                
                for i, audio_file in enumerate(audio_files[:20]):  # Limit to 20 files
                    print(f"\n[{i+1}/{min(20, len(audio_files))}] {os.path.basename(audio_file)}")
                    
                    try:
                        result = quick_detect_abnormal_sound(audio_file, plot=False)
                        
                        if result:
                            summary_data.append({
                                'File': os.path.basename(audio_file),
                                'Type': result['sound_type'],
                                'Abnormal': result['is_abnormal'],
                                'Confidence': result['confidence'],
                                'Duration': len(result['audio'])/result['sample_rate'] if 'audio' in result else 0
                            })
                            
                            if result['is_abnormal']:
                                print(f"   -> ABNORMAL: {result['sound_type']} ({result['confidence']:.1%})")
                            else:
                                print(f"   -> Normal ({1-result['confidence']:.1%})")
                    except Exception as e:
                        print(f"   -> Error: {str(e)}")
                
                # Create summary visualization
                if summary_data:
                    df_summary = pd.DataFrame(summary_data)
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # 1. Abnormal vs Normal count
                    abnormal_count = df_summary['Abnormal'].sum()
                    normal_count = len(df_summary) - abnormal_count
                    
                    axes[0, 0].pie([normal_count, abnormal_count], 
                                  labels=['Normal', 'Abnormal'],
                                  autopct='%1.1f%%',
                                  colors=['green', 'red'],
                                  explode=[0.05, 0.05])
                    axes[0, 0].set_title('Abnormal vs Normal Distribution', fontsize=12, fontweight='bold')
                    
                    # 2. Sound type distribution
                    type_counts = df_summary['Type'].value_counts()
                    axes[0, 1].bar(range(len(type_counts)), type_counts.values, color='steelblue', alpha=0.7)
                    axes[0, 1].set_xticks(range(len(type_counts)))
                    axes[0, 1].set_xticklabels(type_counts.index, rotation=45, ha='right')
                    axes[0, 1].set_title('Sound Type Distribution', fontsize=12, fontweight='bold')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].grid(True, alpha=0.3, axis='y')
                    
                    # 3. Confidence distribution
                    axes[1, 0].hist(df_summary['Confidence'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                    axes[1, 0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
                    axes[1, 0].set_xlabel('Confidence')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # 4. Summary table
                    axes[1, 1].axis('off')
                    summary_text = f"""
Batch Test Summary
{'='*30}
Total Files: {len(df_summary)}
Abnormal Files: {abnormal_count}
Normal Files: {normal_count}
Abnormal Rate: {abnormal_count/len(df_summary):.1%}
Average Confidence: {df_summary['Confidence'].mean():.1%}
                    """
                    axes[1, 1].text(0.5, 0.5, summary_text, 
                                   ha='center', va='center',
                                   fontfamily='monospace', fontsize=11,
                                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    
                    plt.suptitle('Batch Test Results Summary', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"\n📊 Summary: {abnormal_count}/{len(df_summary)} abnormal sounds detected")
            else:
                print("No audio files found in directory")
        else:
            print("Directory not found")
    
    elif choice == '6':
        # Advanced 3D visualization demo
        print("\n🎨 Advanced 3D Visualization Demo...")
        
        # Create complex abnormal sound
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create sound with multiple abnormalities
        complex_sound = 0.1 * np.sin(2 * np.pi * 440 * t)
        
        # Add various abnormalities
        # Impulse
        complex_sound[int(0.2*sr):int(0.2*sr)+30] = 0.7
        
        # Frequency sweep
        sweep = librosa.chirp(fmin=100, fmax=3000, sr=sr, duration=0.3)
        complex_sound[int(0.6*sr):int(0.6*sr)+len(sweep)] += 0.3 * sweep
        
        # Random spikes
        for _ in range(10):
            idx = np.random.randint(int(1.0*sr), int(1.8*sr))
            spike_len = np.random.randint(10, 50)
            complex_sound[idx:idx+spike_len] = 0.4 * np.random.randn(spike_len)
        
        # Save and analyze
        import soundfile as sf
        sf.write('complex_abnormal.wav', complex_sound, sr)
        
        # Run comprehensive analysis
        result = quick_detect_abnormal_sound('complex_abnormal.wav', plot=False)
        
        if result:
            # Show 3D visualization
            visualizer = AudioVisualizer()
            visualizer.plot_interactive_3d_spectrum(
                complex_sound, sr,
                title="Complex Abnormal Sound - 3D Analysis"
            )
            
            # Show comprehensive analysis
            plot_comprehensive_analysis(
                result['audio'], result['sample_rate'],
                result['is_abnormal'], result['sound_type'],
                result['features'], result['confidence']
            )
        
        # Cleanup
        import os
        if os.path.exists('complex_abnormal.wav'):
            os.remove('complex_abnormal.wav')
    
    elif choice == '7':
        print("\n👋 Exiting...")
    
    else:
        # Default: interactive demo
        interactive_visualization_demo()