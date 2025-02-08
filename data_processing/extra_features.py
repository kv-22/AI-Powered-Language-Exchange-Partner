import librosa 
import numpy as np

# Load audio file
audio_path = "path/to/audio.wav"
y, sr = librosa.load(audio_path, sr=16000)  # y: audio signal, sr: sampling rate

pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
pitch_mean = np.mean(pitches[pitches > 0])  # Ignore zero pitches
pitch_std = np.std(pitches[pitches > 0])  # Pitch variability
print("Mean pitch:", pitch_mean)
print("Pitch variability:", pitch_std)

# Extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCCs (Mel-frequency cepstral coefficients)
# Example: Aggregate MFCCs
mfcc_mean = np.mean(mfccs, axis=1)
mfcc_std = np.std(mfccs, axis=1)
aggregated_features = np.hstack([mfcc_mean, mfcc_std])

print("Aggregated features shape:", aggregated_features.shape)
