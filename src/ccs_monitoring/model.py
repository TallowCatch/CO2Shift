"""Simple segmentation models for monitoring experiments."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MonitoringUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16, dropout: float = 0.15) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8, dropout)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_height, input_width = x.shape[-2:]
        pad_height = (-input_height) % 8
        pad_width = (-input_width) % 8
        if pad_height or pad_width:
            # Preserve edge amplitudes while making encoder-decoder shapes compatible.
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        if pad_height or pad_width:
            logits = logits[..., :input_height, :input_width]
        return logits


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1]),
                dtype=x.dtype,
                device=x.device,
            )
        stacked = torch.cat([x, hidden], dim=1)
        update_gate, reset_gate = torch.chunk(torch.sigmoid(self.gates(stacked)), 2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset_gate * hidden], dim=1)))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class TemporalMonitoringNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.temporal_cell = ConvGRUCell(base_channels * 2, base_channels * 2)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec = DoubleConv(base_channels * 2, base_channels, dropout)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, num_channels, input_height, input_width = sequence.shape
        pad_height = (-input_height) % 2
        pad_width = (-input_width) % 2
        if pad_height or pad_width:
            flattened = sequence.reshape(batch_size * num_steps, num_channels, input_height, input_width)
            flattened = F.pad(flattened, (0, pad_width, 0, pad_height), mode="replicate")
            sequence = flattened.reshape(batch_size, num_steps, num_channels, input_height + pad_height, input_width + pad_width)

        logits_per_step: list[torch.Tensor] = []
        hidden: torch.Tensor | None = None
        for step_index in range(num_steps):
            x = sequence[:, step_index]
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            hidden = self.temporal_cell(e2, hidden)
            decoded = self.dec(torch.cat([self.up(hidden), e1], dim=1))
            logits = self.head(decoded)
            if pad_height or pad_width:
                logits = logits[..., :input_height, :input_width]
            logits_per_step.append(logits)
        return torch.stack(logits_per_step, dim=1)


def vertical_derivative(field: torch.Tensor) -> torch.Tensor:
    kernel = field.new_tensor([[-0.5], [0.0], [0.5]], dtype=field.dtype).view(1, 1, 3, 1)
    return F.conv2d(F.pad(field, (0, 0, 1, 1), mode="replicate"), kernel)


class WaveTemporalMonitoringNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        dropout: float = 0.1,
        max_time_shift_samples: float = 3.0,
    ) -> None:
        super().__init__()
        self.max_time_shift_samples = float(max_time_shift_samples)
        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.temporal_cell = ConvGRUCell(base_channels * 2, base_channels * 2)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec = DoubleConv(base_channels * 2, base_channels, dropout)
        self.support_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.amplitude_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.time_shift_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_steps, num_channels, input_height, input_width = sequence.shape
        pad_height = (-input_height) % 2
        pad_width = (-input_width) % 2
        if pad_height or pad_width:
            flattened = sequence.reshape(batch_size * num_steps, num_channels, input_height, input_width)
            flattened = F.pad(flattened, (0, pad_width, 0, pad_height), mode="replicate")
            sequence = flattened.reshape(
                batch_size,
                num_steps,
                num_channels,
                input_height + pad_height,
                input_width + pad_width,
            )

        support_logits_per_step: list[torch.Tensor] = []
        amplitude_per_step: list[torch.Tensor] = []
        time_shift_per_step: list[torch.Tensor] = []
        residual_per_step: list[torch.Tensor] = []
        hidden: torch.Tensor | None = None

        for step_index in range(num_steps):
            x = sequence[:, step_index]
            baseline = x[:, :1]
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            hidden = self.temporal_cell(e2, hidden)
            decoded = self.dec(torch.cat([self.up(hidden), e1], dim=1))

            support_logits = self.support_head(decoded)
            amplitude = torch.tanh(self.amplitude_head(decoded))
            time_shift = self.max_time_shift_samples * torch.tanh(self.time_shift_head(decoded))
            baseline_grad = vertical_derivative(baseline)
            support_prob = torch.sigmoid(support_logits)
            predicted_residual = support_prob * (amplitude * baseline + time_shift * baseline_grad)

            if pad_height or pad_width:
                support_logits = support_logits[..., :input_height, :input_width]
                amplitude = amplitude[..., :input_height, :input_width]
                time_shift = time_shift[..., :input_height, :input_width]
                predicted_residual = predicted_residual[..., :input_height, :input_width]

            support_logits_per_step.append(support_logits)
            amplitude_per_step.append(amplitude)
            time_shift_per_step.append(time_shift)
            residual_per_step.append(predicted_residual)

        return {
            "support_logits": torch.stack(support_logits_per_step, dim=1),
            "amplitude_perturbation": torch.stack(amplitude_per_step, dim=1),
            "time_shift_field": torch.stack(time_shift_per_step, dim=1),
            "predicted_residual": torch.stack(residual_per_step, dim=1),
        }


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=sample_weight)
    probs = torch.sigmoid(logits)
    if sample_weight is not None:
        intersection = (probs * targets * sample_weight).sum(dim=(1, 2, 3))
        union = (probs * sample_weight).sum(dim=(1, 2, 3)) + (targets * sample_weight).sum(dim=(1, 2, 3))
    else:
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6))
    return bce + dice_loss.mean()
