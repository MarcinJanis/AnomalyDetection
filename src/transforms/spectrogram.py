import torch

import librosa


class AudioTransform:
  def __init__(self, sample_rate, n_fft, hop_lenght, mag, f_min, f_max):
    self.sr = sample_rate # [Hz]
    self.n_fft = n_fft # lenght of short fft window [samples]
    self.hop_lenght = hop_lenght # number of samples between successive frames - whatever is that 
    self.n_mels = n_mels # number of Mel bands to generate
    self.mag = mag # Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
    self.f_min = f_min # lowest frequency
    self.f_max = f_max # highest frequency

  def mel_spectogram(waveform):
  
    # mel-spectogram 
    mel_spectogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=self.sample_rate,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        n_mels=self.n_mels,
        fmin=self.f_min,
        fmax=self.f_max,
        power=self.mag 
    )

    mel_spec_db = librosa.power_to_db(mel_spectogram, ref=np.max)

    # norm
    mel_spectogram_norm = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8
    )

    return mel_spectogram_norm

    
  
          # Convert to log scale (dB)
          # Add small epsilon to avoid log(0)
          mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
  
          # Normalize to [0, 1] range
          mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (
              mel_spec_db.max() - mel_spec_db.min() + 1e-8
          )
  
          return mel_spec_norm
