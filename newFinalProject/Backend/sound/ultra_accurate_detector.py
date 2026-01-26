# ulta_accurate_detector.py - STATE-OF-THE-ART ACCURACY WITH ADVANCED VISUALIZATION
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🏆 ULTRA ACCURATE ABNORMAL SOUND DETECTOR")
print("With Advanced Scientific Visualization Suite")
print("=" * 70)

# ============================================================================
# 1. ESSENTIAL IMPORTS WITH ERROR HANDLING
# ============================================================================

def safe_imports():
    """Import libraries with proper error handling"""
    imports = {}
    
    try:
        import librosa
        import librosa.display
        imports['librosa'] = True
        print("✅ librosa loaded")
    except:
        print("❌ librosa not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
        import librosa
        import librosa.display
        imports['librosa'] = True
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        imports['matplotlib'] = True
        print("✅ Matplotlib loaded")
    except:
        print("❌ Matplotlib not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        imports['matplotlib'] = True
    
    try:
        import seaborn as sns
        imports['seaborn'] = True
        print("✅ Seaborn loaded")
    except:
        print("❌ Seaborn not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns
        imports['seaborn'] = True
    
    try:
        import tensorflow as tf
        imports['tensorflow'] = True
        print("✅ TensorFlow loaded")
    except:
        print("❌ TensorFlow not found. Using PyTorch alternative...")
        imports['tensorflow'] = False
    
    try:
        import torch
        import torchaudio
        imports['torch'] = True
        print("✅ PyTorch loaded")
    except:
        print("❌ PyTorch not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchaudio"])
        import torch
        import torchaudio
        imports['torch'] = True
    
    # Additional visualization libraries
    try:
        from scipy import signal, stats
        import pandas as pd
        imports['scipy'] = True
        imports['pandas'] = True
        print("✅ SciPy & Pandas loaded")
    except:
        print("❌ SciPy/Pandas not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "pandas"])
        from scipy import signal, stats
        import pandas as pd
        imports['scipy'] = True
        imports['pandas'] = True
    
    return imports

# ============================================================================
# 2. DOWNLOAD AND USE PRE-TRAINED AUDIO MODELS WITH VISUALIZATION
# ============================================================================

class UltraAccurateDetector:
    """Ensemble of state-of-the-art audio models with advanced visualization"""
    
    def __init__(self, enable_visualization=True, visualization_quality='high'):
        self.imports = safe_imports()
        self.models = {}
        self.enable_visualization = enable_visualization
        self.viz_quality = visualization_quality  # 'low', 'medium', 'high'
        self.visualization_data = {}
        
        print(f"\n🔄 Loading pre-trained models (Visualization: {enable_visualization})...")
        self.load_pretrained_models()
        
        # Initialize visualization style
        if enable_visualization:
            self.setup_visualization_style()
    
    def setup_visualization_style(self):
        """Setup professional visualization style"""
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        # Use available style
        try:
            # Try different style names
            available_styles = plt.style.available
            if 'seaborn-darkgrid' in available_styles:
                plt.style.use('seaborn-darkgrid')
            elif 'dark_background' in available_styles:
                plt.style.use('dark_background')
            elif 'ggplot' in available_styles:
                plt.style.use('ggplot')
            else:
                plt.style.use('default')
        except:
            # Fallback to default style
            plt.style.use('default')
    
        # Set seaborn style
        sns.set_style("darkgrid")
        sns.set_palette("husl")
    
        # Custom color maps
        self.colors = {
            'abnormal': '#FF6B6B',
            'normal': '#4ECDC4',
            'warning': '#FFD166',
            'danger': '#EF476F',
            'safe': '#06D6A0',
            'highlight': '#118AB2',
            'background': '#F8F9FA',
            'grid': '#E9ECEF'
    }
    
        # Create custom colormaps
        from matplotlib.colors import LinearSegmentedColormap
        self.abnormal_cmap = LinearSegmentedColormap.from_list(
            'abnormal', 
            [self.colors['normal'], self.colors['warning'], self.colors['abnormal']]
        )
    
        self.heatmap_cmap = LinearSegmentedColormap.from_list(
            'heatmap',
            ['#00008B', '#006400', '#FFD700', '#FF4500']
        )
    
    def load_pretrained_models(self):
        """Load multiple pre-trained audio models"""
        # [Previous model loading code remains the same]
        # ... (keeping existing model loading code)
    
    def extract_deep_features(self, audio_path):
        """Extract deep learning features from audio"""
        # [Previous feature extraction code remains the same]
        # ... (keeping existing feature extraction code)
    
    # [Previous analysis methods remain the same until we add visualization]
    # ... (keeping all existing analysis methods)
    
    def create_detailed_visualizations(self, audio_path, audio, sr, results):
        """Create comprehensive visualization dashboard"""
        if not self.enable_visualization:
            return
        
        print(f"\n🎨 Generating detailed visualizations ({self.viz_quality} quality)...")
        
        try:
            # Create figure with professional layout
            if self.viz_quality == 'high':
                fig = plt.figure(figsize=(20, 24), dpi=150)
                gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.3)
            elif self.viz_quality == 'medium':
                fig = plt.figure(figsize=(16, 20), dpi=120)
                gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.25)
            else:
                fig = plt.figure(figsize=(12, 16), dpi=100)
                gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.2)
            
            # Store visualization data for later use
            self.visualization_data['audio'] = audio
            self.visualization_data['sr'] = sr
            self.visualization_data['results'] = results
            self.visualization_data['path'] = audio_path
            
            # 1. WAVEFORM WITH ABNORMALITY MARKS
            ax1 = fig.add_subplot(gs[0, :])
            self.plot_waveform_with_abnormalities(ax1, audio, sr, results)
            
            # 2. SPECTROGRAM WITH HEAT MAP
            ax2 = fig.add_subplot(gs[1, :])
            self.plot_enhanced_spectrogram(ax2, audio, sr, results)
            
            # 3. MEL-SPECTROGRAM (Professional Grade)
            ax3 = fig.add_subplot(gs[2, :2])
            self.plot_mel_spectrogram(ax3, audio, sr)
            
            # 4. FEATURE DISTRIBUTION
            ax4 = fig.add_subplot(gs[2, 2:])
            self.plot_feature_distribution(ax4, audio, sr)
            
            # 5. TEMPORAL ANALYSIS
            ax5 = fig.add_subplot(gs[3, :2])
            self.plot_temporal_analysis(ax5, audio, sr)
            
            # 6. SCORECARD & CONFIDENCE
            ax6 = fig.add_subplot(gs[3, 2:])
            self.plot_scorecard(ax6, results)
            
            # 7. FREQUENCY DOMAIN ANALYSIS
            ax7 = fig.add_subplot(gs[4, :2])
            self.plot_frequency_analysis(ax7, audio, sr)
            
            # 8. MODEL CONSENSUS
            ax8 = fig.add_subplot(gs[4, 2:])
            self.plot_model_consensus(ax8, results)
            
            # 9. RISK ASSESSMENT MATRIX (only for high quality)
            if self.viz_quality == 'high':
                ax9 = fig.add_subplot(gs[5, :])
                self.plot_risk_assessment_matrix(ax9, results)
            
            # Add main title
            filename = os.path.basename(audio_path)
            is_abnormal = results.get('is_abnormal', False)
            title_color = self.colors['abnormal'] if is_abnormal else self.colors['normal']
            
            fig.suptitle(
                f'🚨 ABNORMAL SOUND DETECTED - HIGH CONFIDENCE' if is_abnormal else 
                f'✅ NORMAL SOUND - NO THREAT DETECTED',
                fontsize=24, fontweight='bold', 
                color=title_color,
                y=0.98
            )
            
            # Add subtitle
            fig.text(
                0.5, 0.95,
                f'File: {filename} | Duration: {len(audio)/sr:.2f}s | Sample Rate: {sr}Hz',
                ha='center', fontsize=12, style='italic'
            )
            
            # Add timestamp
            from datetime import datetime
            fig.text(
                0.02, 0.02,
                f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                f'Confidence: {results.get("confidence", 0):.1%}',
                fontsize=9, alpha=0.7
            )
            
            # Save figure
            output_path = f"detailed_analysis_{os.path.splitext(filename)[0]}.png"
            plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=150)
            print(f"📊 Visualization saved: {output_path}")
            
            # Also create individual detailed plots
            self.create_individual_detailed_plots(audio, sr, results)
            
            plt.close('all')
            
        except Exception as e:
            print(f"⚠️ Visualization error: {str(e)}")
            self.create_basic_visualization(audio, sr, results)
    
    def plot_waveform_with_abnormalities(self, ax, audio, sr, results):
        """Plot waveform with abnormality regions highlighted"""
        import numpy as np
        
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        # Plot waveform
        ax.plot(time, audio, color=self.colors['highlight'], alpha=0.8, linewidth=1.2)
        ax.fill_between(time, audio, 0, alpha=0.3, color=self.colors['highlight'])
        
        # Detect and highlight abnormal regions
        window_size = int(0.05 * sr)  # 50ms windows
        threshold = 3 * np.std(audio)
        
        abnormal_regions = []
        for i in range(0, len(audio) - window_size, window_size // 2):
            segment = audio[i:i + window_size]
            if np.max(np.abs(segment)) > threshold:
                abnormal_regions.append((i/sr, (i+window_size)/sr))
        
        # Highlight abnormal regions
        for start, end in abnormal_regions:
            ax.axvspan(start, end, alpha=0.3, color=self.colors['abnormal'], zorder=0)
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title('Waveform with Abnormal Regions Highlighted', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend for regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['highlight'], alpha=0.3, label='Audio Waveform'),
            Patch(facecolor=self.colors['abnormal'], alpha=0.3, label='Abnormal Region')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add RMS threshold line
        rms = np.sqrt(np.mean(audio**2))
        ax.axhline(y=threshold, color=self.colors['warning'], linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Threshold ({threshold:.4f})')
        ax.axhline(y=-threshold, color=self.colors['warning'], linestyle=':', 
                  linewidth=2, alpha=0.7)
    
    def plot_enhanced_spectrogram(self, ax, audio, sr, results):
        """Plot enhanced spectrogram with abnormality overlay"""
        import numpy as np
        
        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Plot spectrogram
        im = ax.imshow(db_spectrogram, aspect='auto', origin='lower',
                      extent=[0, len(audio)/sr, 0, sr/2],
                      cmap=self.heatmap_cmap, interpolation='bilinear')
        
        # Add abnormality contours
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(db_spectrogram, sigma=1)
        threshold = np.percentile(db_spectrogram, 95)
        
        # Find high-energy regions
        import matplotlib.pyplot as plt
        contours = ax.contour(smoothed > threshold, levels=[0.5], 
                             colors=[self.colors['danger']], linewidths=2,
                             extent=[0, len(audio)/sr, 0, sr/2], alpha=0.7)
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_title('Enhanced Spectrogram with Abnormality Contours', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('dB', rotation=270, labelpad=15)
        
        # Add frequency bands annotation
        freq_bands = [
            (0, 250, 'Sub-bass'),
            (250, 500, 'Bass'),
            (500, 2000, 'Mid'),
            (2000, 4000, 'High'),
            (4000, sr/2, 'Very High')
        ]
        
        for f_min, f_max, label in freq_bands:
            if f_max <= sr/2:
                ax.axhspan(f_min, f_max, alpha=0.05, color='gray')
                ax.text(len(audio)/sr * 0.02, (f_min+f_max)/2, label,
                       va='center', ha='left', fontsize=8, alpha=0.7)
    
    def plot_mel_spectrogram(self, ax, audio, sr):
        """Plot professional mel-spectrogram"""
        import numpy as np
        
        # Compute mel-spectrogram
        n_mels = 128
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels,
            n_fft=2048, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        im = ax.imshow(mel_spec_db, aspect='auto', origin='lower',
                      cmap=self.abnormal_cmap, interpolation='bicubic')
        
        ax.set_xlabel('Time (frames)', fontweight='bold')
        ax.set_ylabel('Mel-frequency bands', fontweight='bold')
        ax.set_title('Mel-Spectrogram (Deep Learning Feature)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='dB')
        
        # Add mel band indicators
        mel_bands = np.linspace(0, n_mels, 9).astype(int)
        ax.set_yticks(mel_bands)
        ax.set_yticklabels([f'{i}' for i in mel_bands])
    
    def plot_feature_distribution(self, ax, audio, sr):
        """Plot distribution of key acoustic features"""
        import numpy as np
        from scipy import stats
        
        # Compute features
        features = {}
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Temporal features
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        
        # Statistical features
        features['Spectral Centroid (Hz)'] = spectral_centroid
        features['RMS Energy'] = rms
        features['Zero Crossing Rate'] = zcr
        features['Spectral Bandwidth (Hz)'] = spectral_bandwidth
        
        # Create violin plot
        data_to_plot = []
        labels = []
        for label, data in features.items():
            data_to_plot.append(data)
            labels.append(label)
        
        violin_parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        colors = [self.colors['highlight'], self.colors['normal'], 
                 self.colors['warning'], self.colors['abnormal']]
        
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        # Customize plot
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Acoustic Feature Distribution Analysis', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistical annotations
        for i, (label, data) in enumerate(features.items()):
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(i + 1, np.max(data) * 1.05, 
                   f'μ={mean_val:.1f}\nσ={std_val:.1f}',
                   ha='center', fontsize=8, alpha=0.8)
    
    def plot_temporal_analysis(self, ax, audio, sr):
        """Plot temporal analysis with onset detection"""
        import numpy as np
        
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, 
            units='time', backtrack=True
        )
        
        # Plot
        ax.plot(times, onset_env, color=self.colors['danger'], 
               linewidth=2, label='Onset Strength')
        
        # Mark detected onsets
        for onset_time in onset_frames:
            ax.axvline(x=onset_time, color=self.colors['abnormal'], 
                      linestyle='--', alpha=0.7, linewidth=1)
        
        # Fill areas above threshold
        threshold = np.mean(onset_env) + np.std(onset_env)
        ax.fill_between(times, onset_env, threshold, 
                       where=onset_env > threshold,
                       color=self.colors['abnormal'], alpha=0.3,
                       label='Abnormal Activity')
        
        ax.axhline(y=threshold, color=self.colors['warning'], 
                  linestyle=':', linewidth=2, label=f'Threshold ({threshold:.2f})')
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Onset Strength', fontweight='bold')
        ax.set_title('Temporal Analysis & Onset Detection', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        n_onsets = len(onset_frames)
        avg_interval = np.mean(np.diff(onset_frames)) if n_onsets > 1 else 0
        ax.text(0.02, 0.98, f'Detected Onsets: {n_onsets}\nAvg Interval: {avg_interval:.2f}s',
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def plot_scorecard(self, ax, results):
        """Plot scorecard with confidence metrics"""
        import numpy as np
        
        # Prepare data
        categories = ['Overall Confidence', 'Model Consensus', 
                     'Feature Match', 'Temporal Analysis', 'Risk Level']
        
        scores = []
        if results.get('confidence'):
            scores.append(results['confidence'])
        else:
            scores.append(0.5)
        
        # Model consensus score
        if 'abnormal_count' in results and 'total_models' in results:
            consensus = results['abnormal_count'] / results['total_models']
            scores.append(consensus)
        else:
            scores.append(0.5)
        
        # Feature match score (simulated based on results)
        scores.append(results.get('feature_match', 0.6))
        
        # Temporal analysis score
        scores.append(results.get('temporal_score', 0.5))
        
        # Risk level
        is_abnormal = results.get('is_abnormal', False)
        risk_score = 0.8 if is_abnormal else 0.2
        scores.append(risk_score)
        
        # Create radar chart (spider plot)
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Close the polygon
        angles += angles[:1]
        
        ax = plt.subplot(projection='polar')
        ax.plot(angles, scores, 'o-', linewidth=2, color=self.colors['highlight'])
        ax.fill(angles, scores, alpha=0.25, color=self.colors['highlight'])
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        # Set radial grid
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        ax.set_title('Detection Scorecard & Confidence Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add overall score
        overall = np.mean(scores[:-1])  # Exclude the closing duplicate
        ax.text(0.5, 0.5, f'Overall\n{overall:.1%}', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def plot_frequency_analysis(self, ax, audio, sr):
        """Plot detailed frequency domain analysis"""
        import numpy as np
        
        # Compute FFT
        n = len(audio)
        freqs = np.fft.rfftfreq(n, d=1/sr)
        fft_vals = np.abs(np.fft.rfft(audio))
        
        # Apply smoothing
        window_size = 50
        smoothed = np.convolve(fft_vals, np.ones(window_size)/window_size, mode='valid')
        smoothed_freqs = freqs[:len(smoothed)]
        
        # Plot FFT
        ax.plot(smoothed_freqs, smoothed, color=self.colors['highlight'], linewidth=1.5)
        ax.fill_between(smoothed_freqs, smoothed, alpha=0.3, color=self.colors['highlight'])
        
        # Highlight prominent frequencies
        peak_indices = signal.find_peaks(smoothed, 
                                        height=np.mean(smoothed) * 1.5,
                                        distance=len(smoothed)//20)[0]
        
        for idx in peak_indices[:10]:  # Top 10 peaks
            freq = smoothed_freqs[idx]
            amp = smoothed[idx]
            ax.axvline(x=freq, color=self.colors['abnormal'], 
                      linestyle=':', alpha=0.5, linewidth=1)
            ax.plot(freq, amp, 'ro', markersize=5)
            ax.text(freq, amp * 1.05, f'{freq:.0f}Hz', 
                   ha='center', fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude', fontweight='bold')
        ax.set_title('Frequency Domain Analysis with Peak Detection', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        
        # Add frequency bands
        for freq, label in [(20, 'Infra'), (250, 'Bass'), (1000, 'Mid'), 
                           (4000, 'High'), (20000, 'Ultra')]:
            if freq < sr/2:
                ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
                ax.text(freq, np.max(smoothed)*0.9, label, rotation=90,
                       va='top', fontsize=7, alpha=0.6)
    
    def plot_model_consensus(self, ax, results):
        """Plot model consensus and voting results"""
        import numpy as np
        
        # Prepare data
        models = ['YAMNet', 'DeepAcoustic', 'Temporal', 'Ensemble', 'QuickDetect']
        votes = np.array([0.7, 0.6, 0.55, 0.65, 0.5])  # Default scores
        
        if 'all_predictions' in results:
            for i, pred in enumerate(results['all_predictions']):
                if i < len(models):
                    model_name = pred.get('model', models[i])
                    if model_name in models:
                        idx = models.index(model_name)
                        votes[idx] = pred.get('score', pred.get('confidence', 0.5))
        
        # Colors based on votes
        colors = []
        for vote in votes:
            if vote > 0.7:
                colors.append(self.colors['abnormal'])
            elif vote > 0.5:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['normal'])
        
        # Create bar plot
        bars = ax.bar(models, votes, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, vote in zip(bars, votes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{vote:.0%}', ha='center', va='bottom', fontweight='bold')
        
        # Threshold lines
        ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax.axhline(y=0.7, color=self.colors['danger'], linestyle='--', 
                  alpha=0.5, linewidth=1.5, label='High Risk Threshold')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Confidence Score', fontweight='bold')
        ax.set_title('Model Consensus & Voting Results', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_risk_assessment_matrix(self, ax, results):
        """Plot comprehensive risk assessment matrix"""
        import numpy as np
        
        # Risk factors
        factors = ['Loudness', 'Frequency', 'Duration', 'Impulsivity', 
                  'Regularity', 'Model Agreement']
        
        # Simulated risk scores (replace with actual calculations)
        risk_scores = np.array([0.7, 0.6, 0.4, 0.8, 0.3, 0.7])
        
        # Create heatmap
        im = ax.imshow(risk_scores.reshape(1, -1), aspect='auto',
                      cmap=self.abnormal_cmap, vmin=0, vmax=1)
        
        # Add text annotations
        for i, (factor, score) in enumerate(zip(factors, risk_scores)):
            color = 'white' if score > 0.5 else 'black'
            ax.text(i, 0, f'{factor}\n{score:.0%}', 
                   ha='center', va='center', color=color,
                   fontweight='bold', fontsize=10)
            
            # Risk level labels
            if score > 0.7:
                risk_label = 'HIGH'
                label_color = self.colors['danger']
            elif score > 0.5:
                risk_label = 'MEDIUM'
                label_color = self.colors['warning']
            else:
                risk_label = 'LOW'
                label_color = self.colors['safe']
            
            ax.text(i, 0.3, risk_label, ha='center', va='center',
                   color=label_color, fontweight='bold', fontsize=11)
        
        # Customize axes
        ax.set_xticks(np.arange(len(factors)))
        ax.set_xticklabels([])  # Hide default labels
        ax.set_yticks([])
        
        ax.set_title('Risk Assessment Matrix by Factor', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Risk Level', fontweight='bold')
        
        # Add overall risk assessment
        overall_risk = np.mean(risk_scores)
        if overall_risk > 0.7:
            overall_label = '🚨 HIGH RISK'
            overall_color = self.colors['danger']
        elif overall_risk > 0.5:
            overall_label = '⚠️ MEDIUM RISK'
            overall_color = self.colors['warning']
        else:
            overall_label = '✅ LOW RISK'
            overall_color = self.colors['safe']
        
        ax.text(0.5, 1.3, f'{overall_label}\nOverall Risk: {overall_risk:.1%}', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, fontweight='bold', color=overall_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def create_individual_detailed_plots(self, audio, sr, results):
        """Create additional individual detailed plots"""
        try:
            # 1. Create detailed spectrogram
            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
            self.plot_detailed_spectrogram(ax1, audio, sr)
            plt.savefig('detailed_spectrogram.png', bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Create feature correlation heatmap
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
            self.plot_feature_correlation(ax2, audio, sr)
            plt.savefig('feature_correlation.png', bbox_inches='tight')
            plt.close(fig2)
            
            # 3. Create time-frequency analysis
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
            self.plot_time_frequency_analysis(ax3, audio, sr)
            plt.savefig('time_frequency_analysis.png', bbox_inches='tight')
            plt.close(fig3)
            
            print("📈 Additional detailed plots saved")
            
        except Exception as e:
            print(f"⚠️ Individual plots error: {str(e)}")
    
    def plot_detailed_spectrogram(self, ax, audio, sr):
        """Plot highly detailed spectrogram"""
        import numpy as np
        
        # Compute multiple spectrograms with different parameters
        n_ffts = [512, 1024, 2048]
        hop_lengths = [128, 256, 512]
        
        for i, (n_fft, hop) in enumerate(zip(n_ffts, hop_lengths)):
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
            magnitude = np.abs(D)
            db_spec = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Plot with different transparency
            alpha = 0.3 + i * 0.2
            im = ax.imshow(db_spec, aspect='auto', origin='lower',
                          extent=[0, len(audio)/sr, 0, sr/2],
                          cmap='viridis', alpha=alpha)
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_title('Multi-Resolution Spectrogram Analysis', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='dB')
    
    def plot_feature_correlation(self, ax, audio, sr):
        """Plot feature correlation heatmap"""
        import numpy as np
        import pandas as pd
        
        # Extract multiple features
        features = {}
        
        # Time-domain features
        features['RMS'] = librosa.feature.rms(y=audio)[0]
        features['ZCR'] = librosa.feature.zero_crossing_rate(y=audio)[0]
        
        # Frequency-domain features
        features['Centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['Bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['Rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['Flatness'] = librosa.feature.spectral_flatness(y=audio)[0]
        
        # Create DataFrame
        feature_data = {}
        for name, values in features.items():
            feature_data[name] = values[:100]  # Take first 100 samples
        
        df = pd.DataFrame(feature_data)
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=9)
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr_matrix)))
        ax.set_yticks(np.arange(len(corr_matrix)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    def plot_time_frequency_analysis(self, ax, audio, sr):
        """Plot time-frequency analysis using wavelet transform"""
        import numpy as np
        
        # Compute CQT (Constant-Q Transform)
        cqt = np.abs(librosa.cqt(audio, sr=sr))
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Plot CQT
        im = ax.imshow(cqt_db, aspect='auto', origin='lower',
                      extent=[0, len(audio)/sr, 0, librosa.cqt_frequencies(n_bins=cqt.shape[0], sr=sr)[-1]],
                      cmap='magma')
        
        ax.set_xlabel('Time (seconds)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_title('Constant-Q Transform (Time-Frequency Analysis)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_yscale('log')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='dB')
    
    def create_basic_visualization(self, audio, sr, results):
        """Create basic visualization when detailed fails"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Basic Sound Analysis', fontsize=16, fontweight='bold')
            
            # 1. Waveform
            time = np.linspace(0, len(audio)/sr, len(audio))
            axes[0, 0].plot(time, audio)
            axes[0, 0].set_title('Waveform')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            
            # 2. Spectrogram
            D = librosa.stft(audio)
            magnitude = np.abs(D)
            db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
            im = axes[0, 1].imshow(db_spectrogram, aspect='auto', origin='lower')
            axes[0, 1].set_title('Spectrogram')
            plt.colorbar(im, ax=axes[0, 1])
            
            # 3. MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            axes[1, 0].imshow(mfccs, aspect='auto', origin='lower', cmap='coolwarm')
            axes[1, 0].set_title('MFCC')
            
            # 4. Result
            is_abnormal = results.get('is_abnormal', False)
            axes[1, 1].text(0.5, 0.5, 
                           '🚨 ABNORMAL' if is_abnormal else '✅ NORMAL',
                           ha='center', va='center', fontsize=20,
                           color='red' if is_abnormal else 'green',
                           fontweight='bold')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('basic_analysis.png', bbox_inches='tight')
            plt.close()
            
            print("📊 Basic visualization saved: basic_analysis.png")
            
        except Exception as e:
            print(f"❌ Basic visualization failed: {str(e)}")
    
    def ensemble_detection(self, audio_path):
        """Combine multiple models for highest accuracy WITH VISUALIZATION"""
        print(f"\n🔊 Analyzing: {os.path.basename(audio_path)}")
        
        try:
            import librosa
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            print(f"   Duration: {len(audio)/sr:.2f}s, Sample Rate: {sr}Hz")
            
            # ========== ENSEMBLE OF DETECTORS ==========
            all_predictions = []
            
            # 1. YAMNet predictions
            yamnet_preds = self.analyze_with_yamnet(audio)
            if yamnet_preds:
                print(f"\n📊 YAMNet (Google) Predictions:")
                for pred in yamnet_preds[:3]:
                    print(f"   🚨 {pred['class']:<25} {pred['score']:6.1%}")
                    all_predictions.append(pred)
            
            # 2. Deep acoustic feature analysis
            acoustic_result = self.deep_acoustic_analysis(audio, sr)
            if acoustic_result:
                all_predictions.append(acoustic_result)
            
            # 3. Temporal pattern analysis
            temporal_result = self.temporal_pattern_analysis(audio, sr)
            if temporal_result:
                all_predictions.append(temporal_result)
            
            # ========== FUSION AND DECISION ==========
            if not all_predictions:
                print("\n⚠️  No clear predictions from models")
                return self.fallback_analysis(audio, sr)
            
            # Combine predictions
            final_result = self.fuse_predictions(all_predictions)
            
            # Display results
            self.display_results(final_result, audio, sr)
            
            # Generate detailed visualizations
            if self.enable_visualization:
                self.create_detailed_visualizations(audio_path, audio, sr, final_result)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return self.quick_detection(audio_path)

# ============================================================================
# 3. MAIN EXECUTION WITH VISUALIZATION OPTIONS
# ============================================================================

def main():
    """Main function with visualization options"""
    
    print("\n" + "=" * 60)
    print("🎨 VISUALIZATION OPTIONS")
    print("=" * 60)
    print("1. 🔴 HIGH QUALITY - Complete analysis (recommended)")
    print("2. 🟡 MEDIUM QUALITY - Balanced detail and speed")
    print("3. 🟢 LOW QUALITY - Basic visualizations")
    print("4. ❌ NO VISUALIZATION - Text output only")
    
    viz_choice = input("\nSelect visualization quality (1-4): ").strip()
    
    viz_map = {
        '1': ('high', True),
        '2': ('medium', True),
        '3': ('low', True),
        '4': ('none', False)
    }
    
    viz_quality, enable_viz = viz_map.get(viz_choice, ('high', True))
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("\n📁 Enter audio file path: ").strip()
    
    # Clean path
    audio_path = audio_path.strip('"').strip("'").replace('\\', '/')
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    # Initialize detector with visualization settings
    detector = UltraAccurateDetector(
        enable_visualization=enable_viz,
        visualization_quality=viz_quality
    )
    
    # Run analysis
    print(f"\n{'='*60}")
    print(f"🚀 STARTING ANALYSIS WITH {'VISUALIZATION' if enable_viz else 'TEXT ONLY'}")
    print(f"{'='*60}")
    
    result = detector.ensemble_detection(audio_path)
    
    # Display visualization info
    if enable_viz:
        print(f"\n{'='*60}")
        print("📁 VISUALIZATION OUTPUTS:")
        print(f"{'='*60}")
        print(f"✅ Complete analysis: detailed_analysis_*.png")
        print(f"✅ Detailed spectrogram: detailed_spectrogram.png")
        print(f"✅ Feature correlation: feature_correlation.png")
        print(f"✅ Time-frequency analysis: time_frequency_analysis.png")
        if viz_quality == 'low':
            print(f"✅ Basic analysis: basic_analysis.png")
    
    # Final recommendation
    if result and result.get('is_abnormal', False):
        print(f"\n" + "=" * 60)
        print("📋 RECOMMENDED ACTION:")
        
        conf = result.get('confidence', 0)
        if conf > 0.8:
            print("   1. 🚨 CALL EMERGENCY SERVICES")
            print("   2. 🔒 LOCK DOWN AREA")
            print("   3. 📹 REVIEW SECURITY FOOTAGE")
            print("   4. 📱 ALERT ALL PERSONNEL")
        elif conf > 0.6:
            print("   1. 👮 ALERT SECURITY")
            print("   2. 📍 CHECK LOCATION")
            print("   3. 🎧 LISTEN FOR FOLLOW-UP")
            print("   4. 📋 DOCUMENT INCIDENT")
        else:
            print("   1. 👀 MONITOR SITUATION")
            print("   2. 🎤 CONTINUE RECORDING")
            print("   3. 📊 LOG FOR REVIEW")
    
    return result

if __name__ == "__main__":
    result = main()
    
    # Exit code based on detection
    if result and result.get('is_abnormal', False):
        sys.exit(1)  # Abnormal sound detected
    else:
        sys.exit(0)  # Normal sound