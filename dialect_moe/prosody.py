from __future__ import annotations

import math

import torch


PROSODY_FEATURE_NAMES = [
    "log_duration",
    "rms_mean",
    "rms_std",
    "zcr",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "voiced_fraction",
]

PITCH_ENERGY_FEATURE_NAMES = [
    "log_duration",
    "rms_mean",
    "rms_std",
    "zcr",
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "voiced_fraction",
]

_PITCH_ENERGY_INDICES = [0, 1, 2, 3, 7, 8, 9, 10, 11]


def prosody_feature_names(feature_set: str = "legacy") -> list[str]:
    if feature_set == "legacy":
        return PROSODY_FEATURE_NAMES
    if feature_set == "pitch_energy":
        return PITCH_ENERGY_FEATURE_NAMES
    raise ValueError("prosody_feature_set must be one of: legacy, pitch_energy")


def _safe_standardize(features: torch.Tensor) -> torch.Tensor:
    scales = features.new_tensor(
        [1.0, 0.1, 0.1, 0.2, 4000.0, 3000.0, 8000.0, 300.0, 150.0, 100.0, 500.0, 1.0]
    )
    return torch.nan_to_num(features / scales, nan=0.0, posinf=0.0, neginf=0.0)


def _pitch_autocorrelation(
    waveform: torch.Tensor, sample_rate: int, frame_length: int, hop_length: int
) -> tuple[torch.Tensor, int]:
    pitch_frame_length = max(frame_length, int(0.040 * sample_rate))
    if waveform.numel() < pitch_frame_length:
        waveform = torch.nn.functional.pad(waveform, (0, pitch_frame_length - waveform.numel()))
    frames = waveform.unfold(0, pitch_frame_length, hop_length)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    frames = frames * torch.hann_window(pitch_frame_length)
    fft_size = 1 << math.ceil(math.log2(2 * pitch_frame_length - 1))
    spectrum = torch.fft.rfft(frames, n=fft_size)
    correlation = torch.fft.irfft(spectrum.abs().square(), n=fft_size)[..., :pitch_frame_length]
    correlation = correlation / correlation[..., :1].clamp_min(1e-8)
    min_lag = max(1, sample_rate // 600)
    max_lag = min(pitch_frame_length - 1, sample_rate // 50)
    candidates = correlation[:, min_lag : max_lag + 1]
    confidence, indices = candidates.max(dim=-1)
    pitch = sample_rate / (indices + min_lag).float()
    voiced = pitch[confidence >= 0.30]
    return voiced, pitch.numel()


@torch.no_grad()
def extract_prosody(
    waveform: torch.Tensor, sample_rate: int, feature_set: str = "legacy"
) -> torch.Tensor:
    """Extract a compact, deterministic acoustic/prosodic vector on CPU."""
    waveform = waveform.float().flatten()
    if waveform.numel() < 2:
        return torch.zeros(len(prosody_feature_names(feature_set)), dtype=torch.float32)

    waveform = waveform - waveform.mean()
    duration = waveform.numel() / sample_rate
    frame_length = max(256, int(0.025 * sample_rate))
    hop_length = max(80, int(0.010 * sample_rate))
    n_fft = 1 << math.ceil(math.log2(frame_length))

    frames = waveform.unfold(0, frame_length, hop_length) if waveform.numel() >= frame_length else waveform[None]
    rms = frames.square().mean(dim=-1).sqrt()
    zcr = (waveform[:-1] * waveform[1:] < 0).float().mean()

    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=torch.hann_window(frame_length),
        return_complex=True,
    ).abs()
    power = spectrum.square().clamp_min(1e-10)
    frequencies = torch.linspace(0, sample_rate / 2, power.shape[0])[:, None]
    normalizer = power.sum(dim=0).clamp_min(1e-10)
    centroid = (power * frequencies).sum(dim=0) / normalizer
    bandwidth = ((frequencies - centroid[None]).square() * power).sum(dim=0).div(normalizer).sqrt()
    cumulative = power.cumsum(dim=0)
    threshold = 0.85 * cumulative[-1]
    rolloff_bins = (cumulative >= threshold[None]).float().argmax(dim=0)
    rolloff = rolloff_bins.float() * (sample_rate / 2) / max(power.shape[0] - 1, 1)

    voiced, pitch_frame_count = _pitch_autocorrelation(
        waveform, sample_rate, frame_length, hop_length
    )

    if voiced.numel():
        f0_mean, f0_std = voiced.mean(), voiced.std(unbiased=False)
        f0_min, f0_max = voiced.min(), voiced.max()
    else:
        f0_mean = f0_std = f0_min = f0_max = waveform.new_tensor(0.0)

    raw = torch.stack(
        [
            waveform.new_tensor(math.log1p(duration)),
            rms.mean(),
            rms.std(unbiased=False),
            zcr,
            centroid.mean(),
            bandwidth.mean(),
            rolloff.mean(),
            f0_mean,
            f0_std,
            f0_min,
            f0_max,
            waveform.new_tensor(voiced.numel() / max(pitch_frame_count, 1)),
        ]
    )
    standardized = _safe_standardize(raw)
    if feature_set == "pitch_energy":
        standardized = standardized[_PITCH_ENERGY_INDICES]
    elif feature_set != "legacy":
        raise ValueError("prosody_feature_set must be one of: legacy, pitch_energy")
    return standardized.cpu()
