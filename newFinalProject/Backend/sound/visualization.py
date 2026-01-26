# ============================================================================
# SOUND VISUALIZATION MODULE
# Import this module in your main code
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. BASIC VISUALIZATION FUNCTIONS
# ============================================================================

def plot_waveform(audio, sr, title="Audio Waveform", color='blue', alpha=0.7):
    """Plot the waveform of an audio signal"""
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(time, audio, color=color, alpha=alpha, linewidth=0.5)
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_spectrogram(audio, sr, title="Spectrogram", cmap='hot'):
    """Plot spectrogram of audio"""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, cmap=cmap)
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return plt

def plot_mfcc(audio, sr, title="MFCC Features", n_mfcc=13, cmap='coolwarm'):
    """Plot MFCC features"""
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap=cmap)
    plt.title(title, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.tight_layout()
    return plt

# ============================================================================
# 2. SIMPLE VISUALIZATION (One-page summary)
# ============================================================================

def simple_plot_sound(audio, sr, is_abnormal, sound_type, features):
    """Simple one-page visualization for sound analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Waveform with highlights
    time = np.linspace(0, len(audio)/sr, len(audio))
    color = 'red' if is_abnormal else 'green'
    axes[0, 0].plot(time, audio, color=color, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title(f"Waveform: {sound_type}", fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spectrogram (simple)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, 
                            ax=axes[0, 1], cmap='hot')
    axes[0, 1].set_title("Spectrogram", fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    
    # 3. Key features radar/spider chart
    feature_names = ['RMS', 'Crest', 'ZCR', 'Centroid', 'Bandwidth']
    feature_values = [
        min(features.get('max_rms', 0) * 10, 10),
        min(features.get('crest_factor', 0) / 2, 10),
        min(features.get('max_zcr', 0) * 20, 10),
        min(features.get('centroid_mean', 0) / 1000, 10),
        min(features.get('bandwidth_mean', 0) / 500, 10)
    ]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    feature_values += feature_values[:1]
    angles += angles[:1]
    
    ax = axes[1, 0]
    ax = plt.subplot(2, 2, 3, projection='polar')
    ax.plot(angles, feature_values, 'o-', linewidth=2, color='red' if is_abnormal else 'green')
    ax.fill(angles, feature_values, alpha=0.25, color='red' if is_abnormal else 'green')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    ax.set_ylim(0, 10)
    ax.set_title('Acoustic Features (Normalized)', fontweight='bold')
    
    # 4. Detection result
    axes[1, 1].axis('off')
    
    if is_abnormal:
        axes[1, 1].text(0.5, 0.7, '🚨 ABNORMAL', fontsize=24, fontweight='bold', 
                       color='red', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, f'Type: {sound_type}', fontsize=16, 
                       ha='center', va='center')
        
        # Show reasons
        if 'reasons' in features and features['reasons']:
            reasons_text = '\n'.join(features['reasons'][:3])  # Show first 3 reasons
            axes[1, 1].text(0.5, 0.3, 'Reasons:', fontsize=12, fontweight='bold',
                          ha='center', va='center')
            axes[1, 1].text(0.5, 0.2, reasons_text, fontsize=10,
                          ha='center', va='center')
    else:
        axes[1, 1].text(0.5, 0.7, '✅ NORMAL', fontsize=24, fontweight='bold',
                       color='green', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, f'Type: {sound_type}', fontsize=16,
                       ha='center', va='center')
    
    plt.suptitle(f"Sound Analysis Report", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 3. COMPARISON VISUALIZATION
# ============================================================================

def compare_two_sounds(audio1, audio2, sr1, sr2, name1="Sound 1", name2="Sound 2"):
    """Compare two sounds side by side"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Process both sounds
    sounds = [audio1, audio2]
    names = [name1, name2]
    srs = [sr1, sr2]
    
    for i in range(2):
        audio = sounds[i]
        sr = srs[i]
        
        # Waveform
        time = np.linspace(0, len(audio)/sr, len(audio))
        axes[0, i].plot(time, audio, alpha=0.7, linewidth=0.5)
        axes[0, i].set_title(f"{names[i]} - Waveform", fontweight='bold')
        axes[0, i].set_xlabel("Time (s)")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', 
                                sr=sr, ax=axes[1, i], cmap='hot')
        axes[1, i].set_title(f"{names[i]} - Spectrogram", fontweight='bold')
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("Frequency (Hz)")
    
    # Comparison plot (RMS)
    axes[0, 2].axis('off')
    
    # Extract features for comparison
    features_list = []
    for i, audio in enumerate(sounds):
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=srs[i])[0]
        
        features = {
            'max_rms': np.max(rms),
            'mean_rms': np.mean(rms),
            'max_zcr': np.max(zcr),
            'mean_zcr': np.mean(zcr),
            'centroid_mean': np.mean(spectral_centroid)
        }
        features_list.append(features)
    
    # Create comparison bar chart
    feature_names = ['Max RMS', 'Mean RMS', 'Max ZCR', 'Centroid']
    x_pos = np.arange(len(feature_names))
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, [
        features_list[0]['max_rms'],
        features_list[0]['mean_rms'],
        features_list[0]['max_zcr'],
        features_list[0]['centroid_mean'] / 2000
    ], width, label=names[0], alpha=0.7, color='blue')
    
    axes[1, 2].bar(x_pos + width/2, [
        features_list[1]['max_rms'],
        features_list[1]['mean_rms'],
        features_list[1]['max_zcr'],
        features_list[1]['centroid_mean'] / 2000
    ], width, label=names[1], alpha=0.7, color='orange')
    
    axes[1, 2].set_title("Feature Comparison", fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(feature_names)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Sound Comparison Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return features_list

# ============================================================================
# 4. TIME-SERIES VISUALIZATION
# ============================================================================

def plot_time_series_features(audio, sr, title="Time-Series Feature Analysis"):
    """Plot multiple time-series features in one view"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # 1. Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time, audio, alpha=0.7, linewidth=0.5)
    axes[0].set_title("Waveform", fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 2. RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    frames = range(len(rms))
    t_rms = librosa.frames_to_time(frames, sr=sr)
    axes[1].plot(t_rms, rms, color='blue', linewidth=1.5)
    axes[1].set_title("RMS Energy", fontweight='bold')
    axes[1].set_ylabel("RMS")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    t_zcr = librosa.frames_to_time(frames, sr=sr)
    axes[2].plot(t_zcr, zcr, color='green', linewidth=1.5)
    axes[2].set_title("Zero-Crossing Rate", fontweight='bold')
    axes[2].set_ylabel("ZCR")
    axes[2].grid(True, alpha=0.3)
    
    # 4. Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    t_cent = librosa.frames_to_time(frames, sr=sr)
    axes[3].plot(t_cent, spectral_centroid, color='red', linewidth=1.5)
    axes[3].set_title("Spectral Centroid (Hz)", fontweight='bold')
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. DASHBOARD VISUALIZATION
# ============================================================================

def create_audio_dashboard(audio_files, sr=16000, title="Audio Files Dashboard"):
    """Create a dashboard showing multiple audio files"""
    n_files = min(len(audio_files), 4)  # Show up to 4 files
    fig, axes = plt.subplots(n_files, 3, figsize=(15, 4 * n_files))
    
    if n_files == 1:
        axes = axes.reshape(1, -1)
    
    for i, audio_file in enumerate(audio_files[:n_files]):
        try:
            if isinstance(audio_file, str):
                audio, _ = librosa.load(audio_file, sr=sr)
                filename = os.path.basename(audio_file)
            else:
                audio = audio_file
                filename = f"Audio {i+1}"
            
            # 1. Waveform
            time = np.linspace(0, len(audio)/sr, len(audio))
            axes[i, 0].plot(time, audio, alpha=0.6, linewidth=0.5, color='blue')
            axes[i, 0].set_title(f"{filename}\nWaveform", fontweight='bold')
            axes[i, 0].set_xlabel("Time (s)")
            axes[i, 0].set_ylabel("Amplitude")
            axes[i, 0].grid(True, alpha=0.3)
            
            # 2. Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', 
                                    sr=sr, ax=axes[i, 1], cmap='viridis')
            axes[i, 1].set_title(f"Spectrogram", fontweight='bold')
            axes[i, 1].set_xlabel("Time (s)")
            axes[i, 1].set_ylabel("Frequency (Hz)")
            
            # 3. Feature summary
            axes[i, 2].axis('off')
            
            # Calculate features
            rms = librosa.feature.rms(y=audio)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            features_text = (
                f"Duration: {len(audio)/sr:.2f}s\n"
                f"Max RMS: {np.max(rms):.3f}\n"
                f"Mean RMS: {np.mean(rms):.3f}\n"
                f"Max ZCR: {np.max(zcr):.3f}\n"
                f"Centroid: {np.mean(centroid):.0f} Hz\n"
                f"Peak: {np.max(np.abs(audio)):.3f}"
            )
            
            axes[i, 2].text(0.1, 0.9, "Features:", fontweight='bold', fontsize=10)
            axes[i, 2].text(0.1, 0.7, features_text, fontsize=9,
                          verticalalignment='top', linespacing=1.5)
            
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f"Error: {str(e)}", 
                           ha='center', va='center', color='red', fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. FEATURE DISTRIBUTION VISUALIZATION
# ============================================================================

def plot_feature_distribution(features_dict, title="Feature Distribution"):
    """Plot distribution of multiple features"""
    n_features = len(features_dict)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (feature_name, feature_values) in enumerate(features_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        if isinstance(feature_values, (list, np.ndarray)):
            axes[row, col].hist(feature_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].set_title(feature_name, fontweight='bold')
            axes[row, col].set_xlabel("Value")
            axes[row, col].set_ylabel("Frequency")
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            axes[row, col].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            axes[row, col].axvline(mean_val + std_val, color='orange', linestyle=':', label=f'±1 std')
            axes[row, col].axvline(mean_val - std_val, color='orange', linestyle=':')
            axes[row, col].legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(features_dict), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. REAL-TIME VISUALIZATION (For streaming audio)
# ============================================================================

def plot_realtime_audio(audio_chunk, sr, chunk_number, ax_waveform, ax_spectrogram, ax_features):
    """Update plots for real-time audio visualization"""
    # Clear previous plots
    ax_waveform.clear()
    ax_spectrogram.clear()
    ax_features.clear()
    
    # Plot waveform
    time = np.linspace(0, len(audio_chunk)/sr, len(audio_chunk))
    ax_waveform.plot(time, audio_chunk, color='blue', alpha=0.7, linewidth=0.5)
    ax_waveform.set_title(f"Waveform - Chunk {chunk_number}")
    ax_waveform.set_xlabel("Time (s)")
    ax_waveform.set_ylabel("Amplitude")
    ax_waveform.grid(True, alpha=0.3)
    
    # Plot spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_chunk, n_fft=512)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax_spectrogram, cmap='hot')
    ax_spectrogram.set_title("Spectrogram")
    ax_spectrogram.set_xlabel("Time (s)")
    ax_spectrogram.set_ylabel("Frequency (Hz)")
    
    # Calculate and plot features
    rms = np.sqrt(np.mean(audio_chunk**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk)[0])
    
    features = ['RMS', 'Zero-Crossing']
    values = [rms * 10, zcr * 50]  # Scale for visualization
    
    colors = ['green' if rms < 0.1 else 'orange' if rms < 0.3 else 'red',
              'green' if zcr < 0.1 else 'orange' if zcr < 0.3 else 'red']
    
    ax_features.bar(features, values, color=colors, alpha=0.7)
    ax_features.set_title("Audio Features")
    ax_features.set_ylabel("Scaled Value")
    ax_features.set_ylim(0, 10)
    ax_features.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.pause(0.01)

# ============================================================================
# 8. INTERACTIVE VISUALIZATION MENU
# ============================================================================

def visualize_sound_menu(audio, sr, filename="audio"):
    """Interactive menu for visualizing sound"""
    while True:
        print(f"\n📊 VISUALIZATION MENU for {filename}")
        print("="*40)
        print("1. Waveform")
        print("2. Spectrogram")
        print("3. MFCC Features")
        print("4. Time-Series Analysis")
        print("5. All plots (comprehensive)")
        print("6. Feature distribution")
        print("7. Back to main menu")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            plot_waveform(audio, sr, f"Waveform: {filename}").show()
        elif choice == '2':
            plot_spectrogram(audio, sr, f"Spectrogram: {filename}").show()
        elif choice == '3':
            plot_mfcc(audio, sr, f"MFCC Features: {filename}").show()
        elif choice == '4':
            plot_time_series_features(audio, sr, f"Time-Series Analysis: {filename}")
        elif choice == '5':
            # Comprehensive view
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            
            # Waveform
            time = np.linspace(0, len(audio)/sr, len(audio))
            axes[0, 0].plot(time, audio, color='blue', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title("Waveform", fontweight='bold')
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Amplitude")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, 
                                    ax=axes[0, 1], cmap='hot')
            axes[0, 1].set_title("Spectrogram", fontweight='bold')
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Frequency (Hz)")
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, x_axis='time', sr=sr, 
                                    ax=axes[1, 0], cmap='coolwarm')
            axes[1, 0].set_title("MFCC Features", fontweight='bold')
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("MFCC Coefficients")
            
            # Feature summary
            axes[1, 1].axis('off')
            rms = librosa.feature.rms(y=audio)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            
            features_text = (
                f"File: {filename}\n"
                f"Duration: {len(audio)/sr:.2f}s\n"
                f"Sample Rate: {sr} Hz\n"
                f"Max RMS: {np.max(rms):.3f}\n"
                f"Mean RMS: {np.mean(rms):.3f}\n"
                f"Max ZCR: {np.max(zcr):.3f}\n"
                f"Samples: {len(audio)}"
            )
            
            axes[1, 1].text(0.1, 0.9, "Audio Summary:", fontweight='bold', fontsize=12)
            axes[1, 1].text(0.1, 0.7, features_text, fontsize=10,
                          verticalalignment='top', linespacing=1.5)
            
            plt.suptitle(f"Comprehensive Analysis: {filename}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        elif choice == '6':
            # Extract features for distribution
            rms = librosa.feature.rms(y=audio)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            features_dict = {
                'RMS': rms,
                'Zero-Crossing Rate': zcr,
                'Spectral Centroid': centroid
            }
            
            plot_feature_distribution(features_dict, f"Feature Distribution: {filename}")
            
        elif choice == '7':
            print("Returning to main menu...")
            break
        else:
            print("Invalid choice. Please try again.")

# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

if __name__ == "__main__":
    print("Sound Visualization Module")
    print("Import this module in your main code:")
    print("from visualization import *")
    print("\nAvailable functions:")
    print("- simple_plot_sound()")
    print("- compare_two_sounds()")
    print("- plot_time_series_features()")
    print("- create_audio_dashboard()")
    print("- visualize_sound_menu()")
    print("- plot_waveform(), plot_spectrogram(), plot_mfcc()")
    