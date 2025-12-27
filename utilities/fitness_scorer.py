# from scipy.spatial.distance import cosine
from typing import Any

import librosa

import torch
import torchaudio

import numpy as np
import scipy.stats
import soundfile as sf
from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder


class FitnessScorer:
    def __init__(self,target_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 24000

        self.encoder = VoiceEncoder()
        self.target_audio, _ = sf.read(target_path,dtype="float32")
        self.target_wav = preprocess_wav(target_path,source_sr=self.sr)
        self.target_embed = self.encoder.embed_utterance(self.target_wav)

        self._target_tensor = torch.from_numpy(self.target_audio).unsqueeze(0).to(self.device)
        self._init_transforms()
        self.target_features = self.extract_features(self.target_audio)

    def _init_transforms(self):
        n_fft = 2048
        hop_length = 512
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=13,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': 128}
        ).to(self.device)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128
        ).to(self.device)
        
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2
        ).to(self.device)

    def hybrid_similarity(self, audio: NDArray[np.float32], audio2: NDArray[np.float32],target_similarity: float):
        features = self.extract_features(audio)
        self_similarity = self.self_similarity(audio,audio2)
        target_features_pentalty = self.target_feature_penalty(features)

        #Normalize and make higher = better
        feature_similarity = (100.0 - target_features_pentalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01

        values = [target_similarity, self_similarity, feature_similarity]
        # Playing around with the weights can greatly affect scoring and random walk behavior
        weights = [0.48,0.5,0.02]
        score = (np.sum(weights) / np.sum(np.array(weights) / np.array(values))) * 100.0

        return {
            "score": score,
            "target_similarity": target_similarity,
            "self_similarity": self_similarity,
            "feature_similarity": feature_similarity
        }

    def target_similarity(self,audio: NDArray[np.float32]) -> float:
        audio_wav = preprocess_wav(audio,source_sr=24000)
        audio_embed = self.encoder.embed_utterance(audio_wav)
        similarity = np.inner(audio_embed, self.target_embed)
        return similarity

    def target_feature_penalty(self,features: dict[str, Any]) -> float:
        """Penalizes for differences in audio features"""
        # Normalized feature difference compared to target features
        penalty = 0.0
        for key, value in features.items():
            #diff = abs((value - self.target_features[key])/self.target_features[key])
            if self.target_features[key] != 0:
                diff = abs((value - self.target_features[key])/self.target_features[key])
            else:
                diff = abs(value)
                
            penalty += diff
        return penalty

    def self_similarity(self,audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        """Self similarity indicates model stability. Poor self similarity means different input makes different sounding voices"""
        audio_wav1 = preprocess_wav(audio1,source_sr=24000)
        audio_embed1 = self.encoder.embed_utterance(audio_wav1)

        audio_wav2 = preprocess_wav(audio2,source_sr=24000)
        audio_embed2 = self.encoder.embed_utterance(audio_wav2)
        return np.inner(audio_embed1, audio_embed2)

    def extract_features(self, audio: NDArray[np.float32] | NDArray[np.float64], sr: int = 24000) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.

        Args:
            audio: Audio signal as numpy array (np.float32)
            sr: Sample rate (fixed at 24000 Hz)

        Returns:
            Dictionary containing extracted features
        """
        # Ensure audio is the right shape (flatten stereo to mono if needed)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        # Initialize features dictionary
        features = {}

        # Basic features
        # features["duration"] = len(audio) / sr
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))

        signs = torch.sign(audio_tensor)
        zero_crossings = torch.sum(torch.abs(signs[:, 1:] - signs[:, :-1])) / (2 * audio_tensor.shape[1])

        #features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        features["zero_crossing_rate"] = float(zero_crossings.cpu())

        # Spectral features
        # Compute STFT
        #n_fft = 2048  # window size
        #hop_length = 512  # hop length

        # Spectral centroid and bandwidth (where the "center" of the sound is)
        #spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        #features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        #features["spectral_centroid_std"] = float(np.std(spectral_centroids))
        spec = self.spectrogram_transform(audio_tensor)
        freqs = torch.linspace(0, sr / 2, spec.shape[1], device=self.device)
        spec_sum = spec.sum(dim=1, keepdim=True) + 1e-10
        
        n_fft = 2048  # window size
        hop_length = 512  # hop length

        spectral_centroids = (freqs.view(1, -1, 1) * spec).sum(dim=1) / spec_sum.squeeze(1)
        features["spectral_centroid_mean"] = float(spectral_centroids.mean().cpu())
        features["spectral_centroid_std"] = float(spectral_centroids.std().cpu()) 

        #spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        #features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        #features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        spectral_bandwidth = torch.sqrt(((freqs.view(1, -1, 1) - spectral_centroids.unsqueeze(1))**2 * spec).sum(dim=1) / spec_sum.squeeze(1))
        features["spectral_bandwidth_mean"] = float(spectral_bandwidth.mean().cpu())
        features["spectral_bandwidth_std"] = float(spectral_bandwidth.std().cpu())


        # Spectral rolloff
        #rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        #features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        #features["spectral_rolloff_std"] = float(np.std(rolloff))

        cumsum = torch.cumsum(spec, dim=1)
        threshold = 0.85 * spec.sum(dim=1, keepdim=True)
        rolloff_idx = (cumsum >= threshold).float().argmax(dim=1)
        rolloff = freqs[rolloff_idx.long()].float()
        features["spectral_rolloff_mean"] = float(rolloff.mean().cpu())
        features["spectral_rolloff_std"] = float(rolloff.std().cpu())

        # Spectral contrast
        #contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        #features["spectral_contrast_mean"] = float(np.mean(contrast))
        #features["spectral_contrast_std"] = float(np.std(contrast))

        n_bands = 6
        freq_bins = spec.shape[1]
        band_size = freq_bins // n_bands
        contrast_values = []
        for b in range(n_bands):
            start = b * band_size
            end = start + band_size if b < n_bands - 1 else freq_bins
            band = spec[:, start:end, :]
            peak = band.max(dim=1)[0] + 1e-10 
            valley = band.min(dim=1)[0] + 1e-10
            contrast_values.append(torch.log(peak / valley))
        contrast = torch.stack(contrast_values, dim=1)
        features["spectral_contrast_mean"] = float(contrast.mean().cpu())
        features["spectral_contrast_std"] = float(contrast.std().cpu())

        # MFCCs (Mel-frequency cepstral coefficients) - important for speech
        #mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

        ## Store each MFCC coefficient mean and std
        #for i in range(len(mfccs)):
        #    features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
        #    features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

        mfccs = self.mfcc_transform(audio_tensor).squeeze(0)  # (n_mfcc, time)
        for i in range(mfccs.shape[0]):
            features[f"mfcc{i+1}_mean"] = float(mfccs[i].mean().cpu())
            features[f"mfcc{i+1}_std"] = float(mfccs[i].std().cpu())

        # MFCC delta features (first derivative)
        #mfcc_delta = librosa.feature.delta(mfccs)
        #for i in range(len(mfcc_delta)):
        #    features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
        #    features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))

        mfcc_delta = torchaudio.functional.compute_deltas(mfccs.unsqueeze(0)).squeeze(0)
        for i in range(mfcc_delta.shape[0]):
            features[f"mfcc{i+1}_delta_mean"] = float(mfcc_delta[i].mean().cpu())
            features[f"mfcc{i+1}_delta_std"] = float(mfcc_delta[i].std().cpu())

        # Chroma features - useful for characterizing harmonic content
        #chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        #features["chroma_mean"] = float(np.mean(chroma))
        #features["chroma_std"] = float(np.std(chroma))

        ## Store individual chroma features
        #for i in range(len(chroma)):
        #    features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        #    features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        n_chroma = 12
        chroma_weights = torch.zeros(spec.shape[1], n_chroma, device=self.device)
        freqs_hz = torch.linspace(0, sr / 2, spec.shape[1], device=self.device)
        for i, f in enumerate(freqs_hz):
            if f > 0:
                pitch_class = int(torch.round(12 * torch.log2(f / 440.0 + 1e-10)).item()) % 12
                chroma_weights[i, pitch_class] = 1.0
        chroma = torch.matmul(spec.squeeze(0).T, chroma_weights).T  # (12, time)
        features["chroma_mean"] = float(chroma.mean().cpu())
        features["chroma_std"] = float(chroma.std().cpu())
        for i in range(chroma.shape[0]):
            features[f"chroma_{i+1}_mean"] = float(chroma[i].mean().cpu())
            features[f"chroma_{i+1}_std"] = float(chroma[i].std().cpu())

        # Mel spectrogram (average across frequency bands)
        #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        #features["mel_spec_mean"] = float(np.mean(mel_spec))
        #features["mel_spec_std"] = float(np.std(mel_spec))

        mel_spec = self.mel_transform(audio_tensor)
        features["mel_spec_mean"] = float(mel_spec.mean().cpu())
        features["mel_spec_std"] = float(mel_spec.std().cpu())

        # Spectral flatness - measure of the noisiness of the signal
        #flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
        #features["spectral_flatness_mean"] = float(np.mean(flatness))
        #features["spectral_flatness_std"] = float(np.std(flatness))

        geo_mean = torch.exp(torch.log(spec + 1e-10).mean(dim=1))
        arith_mean = spec.mean(dim=1) + 1e-10
        flatness = geo_mean / arith_mean
        features["spectral_flatness_mean"] = float(flatness.mean().cpu())
        features["spectral_flatness_std"] = float(flatness.std().cpu())

        # Tonnetz (tonal centroid features)
        #tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        #features["tonnetz_mean"] = float(np.mean(tonnetz))
        #features["tonnetz_std"] = float(np.std(tonnetz))

        phi = torch.tensor([7/6, 7/6, 7/6, 7/6, 7/6, 7/6], device=self.device) * torch.pi
        r = torch.arange(12, device=self.device).float()
        tonnetz_transform = torch.stack([
            torch.sin(r * phi[0]), torch.cos(r * phi[0]),
            torch.sin(r * phi[2]), torch.cos(r * phi[2]),
            torch.sin(r * phi[4]), torch.cos(r * phi[4])
        ], dim=0)  # (6, 12)
        tonnetz = torch.matmul(tonnetz_transform, chroma)  # (6, time)
        features["tonnetz_mean"] = float(tonnetz.mean().cpu())
        features["tonnetz_std"] = float(tonnetz.std().cpu())

        # Rhythm features - tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features["tempo"] = float(tempo)

        if len(beat_frames) > 0:
            # Calculate beat_stats only if beats are detected
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                beat_diffs = np.diff(beat_times)
                features["beat_mean"] = float(np.mean(beat_diffs))
                features["beat_std"] = float(np.std(beat_diffs))
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0
        else:
            features["beat_mean"] = 0.0
            features["beat_std"] = 0.0

        # Pitch and harmonics
        #pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        # For each frame, find the highest magnitude pitch
        #pitch_values = []
        #for i in range(magnitudes.shape[1]):
        #    index = magnitudes[:, i].argmax()
        #    pitch = pitches[index, i]
        #    if pitch > 0:  # Exclude zero pitch
        #        pitch_values.append(pitch)

        #if pitch_values:
        #    features["pitch_mean"] = float(np.mean(pitch_values))
        #    features["pitch_std"] = float(np.std(pitch_values))
        #else:
        #    features["pitch_mean"] = 0.0
        #    features["pitch_std"] = 0.0


        pitch = torchaudio.functional.detect_pitch_frequency(audio_tensor, sr)
        pitch_values = pitch[pitch > 0]
        if pitch_values.numel() > 0:
            features["pitch_mean"] = float(pitch_values.mean().cpu())
            features["pitch_std"] = float(pitch_values.std().cpu())
        elif pitch_values.numel() == 1:
            features["pitch_mean"] = float(pitch_values.mean().cpu())
            features["pitch_std"] = 0.0
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        
        # Speech-specific features

        # Voice Activity Detection (simplified)
        # Higher energies typically indicate voice activity
        #energy = np.array([sum(abs(audio[i:i+hop_length])) for i in range(0, len(audio), hop_length)])
        #features["energy_mean"] = float(np.mean(energy))
        #features["energy_std"] = float(np.std(energy))

        energy = audio_tensor.unfold(1, hop_length, hop_length).abs().sum(dim=2)
        features["energy_mean"] = float(energy.mean().cpu())
        features["energy_std"] = float(energy.std().cpu())

        # Harmonics-to-noise ratio (simplified approximation)
        # Using the squared magnitude of the spectrogram
        #S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        #S_squared = S**2
        #S_mean = np.mean(S_squared, axis=1)
        #S_std = np.std(S_squared, axis=1)
        #S_ratio = np.divide(S_mean, S_std, out=np.zeros_like(S_mean), where=S_std!=0)
        #features["harmonic_ratio"] = float(np.mean(S_ratio))

        S_squared = spec.squeeze(0)  # (freq, time)
        S_mean = S_squared.mean(dim=1)
        S_std = S_squared.std(dim=1) + 1e-10
        S_ratio = S_mean / S_std
        features["harmonic_ratio"] = float(S_ratio.mean().cpu())

        # Statistical features from the raw waveform
        features["audio_mean"] = float(np.mean(audio))
        features["audio_std"] = float(np.std(audio))
        features["audio_skew"] = float(scipy.stats.skew(audio))
        features["audio_kurtosis"] = float(scipy.stats.kurtosis(audio))

        return features
