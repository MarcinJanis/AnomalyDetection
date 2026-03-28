import numpy as np
import torch
import librosa


class MelSpectrogramTransform:
    def __init__(
        self,
        sample_rate=16000,
        duration=2.0,
        n_mels=128,
        n_fft=1024,
        hop_length=256,
        f_min=0.0,
        f_max=None,
        power=2.0
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power

        self.target_num_samples = int(sample_rate * duration)

    def to_mono(self, waveform):
        # waveform: numpy array
        # może być [T] albo [channels, T]
        if waveform.ndim == 1:
            return waveform

        if waveform.ndim == 2:
            return np.mean(waveform, axis=0)

        raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

    def resample_if_needed(self, waveform, sr):
        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.sample_rate
            )
            sr = self.sample_rate
        return waveform, sr

    def pad_or_trim(self, waveform):
        current_num_samples = waveform.shape[0]

        if current_num_samples > self.target_num_samples:
            waveform = waveform[:self.target_num_samples]
        elif current_num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - current_num_samples
            waveform = np.pad(waveform, (0, pad_amount), mode="constant")

        return waveform

    def __call__(self, waveform, sr):
        waveform = self.to_mono(waveform)
        waveform, sr = self.resample_if_needed(waveform, sr)
        waveform = self.pad_or_trim(waveform)

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        # CNN zwykle chce [1, n_mels, time]
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        return mel_tensor