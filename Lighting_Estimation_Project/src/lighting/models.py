"""A spherical VQ autoencoder and image-conditioned causal token generator."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ERPConv(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, stride=stride)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        return self.conv(F.pad(x, (0, 0, 1, 1), mode='replicate'))


class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config['tokenizer']
        if c['downsample_factor'] != 16:
            raise ValueError('This encoder implements exactly four 2x downsampling stages')
        self.codebook_size = c['codebook_size']
        self.commitment = c['commitment_weight']
        channels = [3, c['base_channels'], c['base_channels'] * 2,
                    c['base_channels'] * 4, c['embedding_dim']]
        encoder, decoder = [], []
        for cin, cout in zip(channels[:-1], channels[1:]):
            encoder.extend([ERPConv(cin, cout, stride=2), nn.SiLU()])
        self.encoder = nn.Sequential(*encoder)
        for cin, cout in zip(channels[:0:-1], channels[-2::-1]):
            decoder.extend([nn.Upsample(scale_factor=2, mode='nearest'), ERPConv(cin, cout)])
            if cout != 3:
                decoder.append(nn.SiLU())
        self.decoder = nn.Sequential(*decoder)
        nn.init.constant_(self.decoder[-1].conv.bias, -2.0)
        self.codebook = nn.Embedding(self.codebook_size, c['embedding_dim'])
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)

    def quantize(self, latent):
        vectors = latent.permute(0, 2, 3, 1).contiguous()
        flat = vectors.reshape(-1, vectors.shape[-1]).float()
        codes = self.codebook.weight.float()
        distance = flat.square().sum(1, keepdim=True) + codes.square().sum(1) - 2 * flat @ codes.T
        indices = distance.argmin(1).reshape(vectors.shape[:-1])
        quantized = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(quantized, latent.detach())
        loss = loss + self.commitment * F.mse_loss(latent, quantized.detach())
        straight_through = latent + (quantized - latent).detach()
        return straight_through, indices, loss

    def encode(self, log_hdr):
        return self.quantize(self.encoder(log_hdr))[1]

    def decode(self, indices):
        latent = self.codebook(indices).permute(0, 3, 1, 2).contiguous()
        return F.softplus(self.decoder(latent))

    def forward(self, log_hdr):
        latent, indices, vq_loss = self.quantize(self.encoder(log_hdr))
        return F.softplus(self.decoder(latent)), indices, vq_loss


class AutoregressiveLighting(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        c = config['ar']
        height, width = config['data']['env_size']
        if height % 16 or width % 16:
            raise ValueError('Environment dimensions must be divisible by 16')
        self.grid = (height // 16, width // 16)
        self.length = self.grid[0] * self.grid[1]
        self.bos = config['tokenizer']['codebook_size']
        self.embedding = nn.Embedding(self.bos + 1, c['d_model'])
        # Row position plus periodic longitude; the two seam neighbours have nearby positions.
        rows = torch.arange(self.grid[0]).repeat_interleave(self.grid[1]).float()
        cols = torch.arange(self.grid[1]).repeat(self.grid[0]).float()
        phase = 2 * math.pi * (cols + 0.5) / self.grid[1]
        positions = torch.stack([(rows + 0.5) / self.grid[0], phase.sin(), phase.cos()], -1)
        self.register_buffer('positions', positions)
        self.position_projection = nn.Linear(3, c['d_model'])
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.image_encoder.requires_grad_(False)
        self.image_projection = nn.Linear(512, c['d_model'])
        self.image_position = nn.Linear(2, c['d_model'])
        layer = nn.TransformerDecoderLayer(c['d_model'], c['heads'], 4 * c['d_model'],
                                           dropout=c['dropout'], batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, c['layers'], norm=nn.LayerNorm(c['d_model']))
        # TransformerDecoder clones its template: initialise each clone independently.
        for block in self.decoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.output = nn.Linear(c['d_model'], self.bos)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()  # Frozen BatchNorm statistics as well as weights.
        return self

    def image_features(self, image):
        with torch.no_grad():
            features = self.image_encoder((image - self.image_mean) / self.image_std)
        h, w = features.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                              torch.linspace(-1, 1, w, device=image.device), indexing='ij')
        position = self.image_position(torch.stack([x, y], -1).reshape(-1, 2))
        return self.image_projection(features.flatten(2).transpose(1, 2)) + position

    def next_logits(self, prefix, memory):
        length = prefix.shape[1]
        if not 0 < length <= self.length:
            raise ValueError('Prefix length outside the configured token sequence')
        features = self.embedding(prefix) + self.position_projection(self.positions[:length])
        mask = torch.ones(length, length, dtype=torch.bool, device=prefix.device).triu(1)
        return self.output(self.decoder(features, memory, tgt_mask=mask))

    def forward(self, image, target_tokens):
        target_tokens = target_tokens.flatten(1)
        if target_tokens.shape[1] != self.length:
            raise ValueError('Target token grid does not match configuration')
        bos = torch.full((image.shape[0], 1), self.bos, dtype=torch.long, device=image.device)
        prefix = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        return self.next_logits(prefix, self.image_features(image))

    @torch.no_grad()
    def generate(self, image):
        if self.training:
            raise ValueError('Call eval() before generation')
        memory = self.image_features(image)
        prefix = torch.full((image.shape[0], 1), self.bos, dtype=torch.long, device=image.device)
        for _ in range(self.length):
            token = self.next_logits(prefix, memory)[:, -1].argmax(-1, keepdim=True)
            prefix = torch.cat([prefix, token], dim=1)
        return prefix[:, 1:].reshape(image.shape[0], *self.grid)
