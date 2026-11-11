"""Separate token likelihood from illumination quality; no GAN or Retinex shortcut."""
import math

import numpy as np
import torch
from torch import nn

from .data import erp_directions


class IlluminationObjective(nn.Module):
    def __init__(self, config):
        super().__init__()
        height, width = config['data']['env_size']
        edges = torch.linspace(0, math.pi, height + 1)
        row_area = (edges[:-1].cos() - edges[1:].cos()) * 2 * math.pi / width
        weights = row_area[:, None].expand(height, width)
        self.register_buffer('weights', weights[None, None] / weights.sum())
        directions = torch.from_numpy(erp_directions(height, width)).reshape(-1, 3)
        normals = torch.from_numpy(erp_directions(8, 16)).reshape(-1, 3)
        # Unoccluded Lambertian irradiance divided by pi, at a fixed normal grid.
        kernel = (normals @ directions.T).clamp_min(0) * weights.flatten()[None] / math.pi
        self.register_buffer('diffuse_kernel', kernel)
        self.scale = config['data']['hdr_scale']
        self.max_log = config['data']['max_log_value']
        self.linear_weight = config['loss']['linear']
        self.diffuse_weight = config['loss']['diffuse']
        self.edge_weight = config['loss'].get('edge', 0.0)
        self.highlight_weight = config['loss'].get('highlight', 0.0)

    def weighted_mean(self, value):
        return (value * self.weights).sum((-2, -1)).mean(1)

    def to_hdr(self, log_hdr):
        return torch.expm1(log_hdr.clamp(0, self.max_log)) * self.scale

    def diffuse(self, hdr):
        return hdr.flatten(2) @ self.diffuse_kernel.T

    def forward(self, predicted_log, target_hdr):
        target_log = torch.log1p(target_hdr / self.scale)
        prediction = self.to_hdr(predicted_log)
        importance = 1 + self.highlight_weight * (target_log.mean(1, keepdim=True) /
            (target_log.mean((1, 2, 3), keepdim=True) + 0.1)).clamp(max=10)
        log_error = self.weighted_mean((predicted_log - target_log).square() * importance)
        delta = predicted_log - target_log
        horizontal = delta - delta.roll(1, dims=-1)
        vertical = torch.nn.functional.pad(delta[:, :, 1:] - delta[:, :, :-1], (0, 0, 0, 1))
        edge_error = self.weighted_mean(horizontal.abs() + vertical.abs())
        linear_error = self.weighted_mean((prediction - target_hdr).square())
        normalizer = self.weighted_mean(target_hdr.square()).detach() + self.scale ** 2
        diffuse_error = (self.diffuse(prediction) - self.diffuse(target_hdr)).square().mean((1, 2))
        diffuse_normalizer = self.diffuse(target_hdr).square().mean((1, 2)).detach() + self.scale ** 2
        total = (log_error + self.edge_weight * edge_error + self.linear_weight * linear_error / normalizer
                 + self.diffuse_weight * diffuse_error / diffuse_normalizer).mean()
        return total

    @torch.no_grad()
    def metrics(self, predicted_log, target_hdr):
        prediction = self.to_hdr(predicted_log)
        target_log = torch.log1p(target_hdr / self.scale)
        scale = self.weighted_mean(prediction * target_hdr) / self.weighted_mean(prediction.square()).clamp_min(1e-12)
        aligned = prediction * scale[:, None, None, None]
        pred_energy = self.weighted_mean(prediction)
        target_energy = self.weighted_mean(target_hdr)
        values = {
            'log_rmse': self.weighted_mean((predicted_log - target_log).square()).sqrt(),
            'linear_rmse': self.weighted_mean((prediction - target_hdr).square()).sqrt(),
            'scale_aligned_rmse': self.weighted_mean((aligned - target_hdr).square()).sqrt(),
            'diffuse_rmse': (self.diffuse(prediction) - self.diffuse(target_hdr)).square().mean((1, 2)).sqrt(),
            'energy_relative_error': (pred_energy - target_energy).abs() / target_energy.clamp_min(1e-12),
            'inverse_clipped_fraction': (predicted_log > self.max_log).float().mean((1, 2, 3))
        }
        return [{name: float(value[i]) for name, value in values.items()}
                for i in range(target_hdr.shape[0])]

    @torch.no_grad()
    def render_spheres(self, hdr, size=64, roughness=0.25):
        """Diagnostic unoccluded spheres; not a scene renderer or shadow simulator."""
        device = hdr.device
        coords = (torch.arange(size, device=device) + 0.5) * 2 / size - 1
        y, x = torch.meshgrid(-coords, coords, indexing='ij')
        mask = x.square() + y.square() <= 1
        z = (1 - x.square() - y.square()).clamp_min(0).sqrt()
        normals = torch.stack([x[mask], y[mask], z[mask]], -1)
        lights = torch.from_numpy(erp_directions(*hdr.shape[-2:])).to(device).reshape(-1, 3)
        area = self.weights.flatten() * 4 * math.pi
        reflection = 2 * normals[:, 2:3] * normals - normals.new_tensor([0, 0, 1])
        outputs = []
        # Bound memory by integrating small batches of visible surface points.
        for direction, exponent in [(normals, 1.0), (reflection, 2 / roughness ** 2)]:
            chunks = []
            for chunk in direction.split(64):
                kernel = (chunk @ lights.T).clamp_min(0).pow(exponent) * area[None]
                kernel = kernel / kernel.sum(1, keepdim=True).clamp_min(1e-12)
                chunks.append(hdr.flatten(2) @ kernel.T)
            pixels = torch.cat(chunks, -1)
            image = hdr.new_zeros(hdr.shape[0], 3, size, size)
            image[:, :, mask] = pixels
            outputs.append(image)
        return outputs
