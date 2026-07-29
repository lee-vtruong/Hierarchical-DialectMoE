from __future__ import annotations

import math

import torch


NUM_SPECTRAL_BANDS = 12
SPECTRAL_FEATURE_NAMES = [
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff85_mean",
    "rolloff85_std",
    "flatness_mean",
    "flatness_std",
    "flux_mean",
    "flux_std",
    *[f"fft_band_{index:02d}" for index in range(NUM_SPECTRAL_BANDS)],
    "low_high_log_ratio",
    "spectral_entropy_mean",
]


@torch.no_grad()
def extract_spectral(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Extract deterministic FFT/STFT distribution features on CPU.

    Frequency-location features are normalized by Nyquist frequency. Band
    features are relative powers, making the vector less sensitive to volume.
    """
    waveform = waveform.float().flatten()
    if waveform.numel() < 2:
        return torch.zeros(len(SPECTRAL_FEATURE_NAMES), dtype=torch.float32)

    waveform = waveform - waveform.mean()
    frame_length = max(256, int(0.025 * sample_rate))
    hop_length = max(80, int(0.010 * sample_rate))
    if waveform.numel() < frame_length:
        waveform = torch.nn.functional.pad(
            waveform, (0, frame_length - waveform.numel())
        )
    n_fft = 1 << math.ceil(math.log2(frame_length))
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=torch.hann_window(frame_length, device=waveform.device),
        return_complex=True,
    ).abs()
    power = spectrum.square().clamp_min(1e-12)
    normalized = power / power.sum(dim=0, keepdim=True).clamp_min(1e-12)

    frequency_axis = torch.linspace(
        0.0, 1.0, power.shape[0], device=waveform.device
    )[:, None]
    centroid = (normalized * frequency_axis).sum(dim=0)
    bandwidth = (
        (normalized * (frequency_axis - centroid[None]).square())
        .sum(dim=0)
        .sqrt()
    )
    cumulative = normalized.cumsum(dim=0)
    rolloff_bins = (cumulative >= 0.85).float().argmax(dim=0)
    rolloff = rolloff_bins.float() / max(power.shape[0] - 1, 1)
    flatness = torch.exp(power.log().mean(dim=0)) / power.mean(dim=0).clamp_min(
        1e-12
    )

    if normalized.shape[1] > 1:
        flux = (
            normalized[:, 1:].sub(normalized[:, :-1]).square().sum(dim=0).sqrt()
        )
    else:
        flux = waveform.new_zeros(1)

    band_edges = torch.linspace(
        0, power.shape[0], NUM_SPECTRAL_BANDS + 1, device=waveform.device
    ).round().long()
    band_features = []
    mean_distribution = normalized.mean(dim=1)
    for index in range(NUM_SPECTRAL_BANDS):
        start = int(band_edges[index].item())
        end = max(start + 1, int(band_edges[index + 1].item()))
        band_features.append(mean_distribution[start:end].sum())
    band_features_tensor = torch.stack(band_features)
    band_features_tensor = torch.log1p(NUM_SPECTRAL_BANDS * band_features_tensor)

    frequencies_hz = frequency_axis[:, 0] * (sample_rate / 2)
    mean_power = power.mean(dim=1)
    low = mean_power[frequencies_hz <= 1000].sum()
    high = mean_power[frequencies_hz >= 3000].sum()
    low_high_ratio = torch.log((low + 1e-12) / (high + 1e-12)).clamp(-10, 10) / 10
    spectral_entropy = (
        -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=0)
        / math.log(max(normalized.shape[0], 2))
    )

    features = torch.cat(
        [
            torch.stack(
                [
                    centroid.mean(),
                    centroid.std(unbiased=False),
                    bandwidth.mean(),
                    bandwidth.std(unbiased=False),
                    rolloff.mean(),
                    rolloff.std(unbiased=False),
                    flatness.mean(),
                    flatness.std(unbiased=False),
                    flux.mean(),
                    flux.std(unbiased=False),
                ]
            ),
            band_features_tensor,
            torch.stack([low_high_ratio, spectral_entropy.mean()]),
        ]
    )
    return torch.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).float().cpu()
