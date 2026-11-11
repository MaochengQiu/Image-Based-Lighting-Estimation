"""A spherical VQ autoencoder and image-conditioned causal token generator."""
import math
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


class ERPConv(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, stride=stride)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        return self.conv(F.pad(x, (0, 0, 1, 1), mode='replicate'))


class ERPResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ERPConv(channels, channels), nn.SiLU(), ERPConv(channels, channels))

    def forward(self, x):
        return x + 0.1 * self.net(x)


class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config['tokenizer']
        factor = c['downsample_factor']
        if factor not in (4, 8, 16):
            raise ValueError('downsample_factor must be 4, 8 or 16')
        stages = int(math.log2(factor))
        self.codebook_size = c['codebook_size']
        self.commitment = c['commitment_weight']
        channels = [3] + [c['base_channels'] * min(2 ** i, 4) for i in range(stages - 1)] + [c['embedding_dim']]
        encoder, decoder = [], []
        for cin, cout in zip(channels[:-1], channels[1:]):
            encoder.extend([ERPConv(cin, cout, stride=2), nn.SiLU()])
            if c.get('residual_blocks', False):
                encoder.append(ERPResidual(cout))
        self.encoder = nn.Sequential(*encoder)
        for cin, cout in zip(channels[:0:-1], channels[-2::-1]):
            decoder.extend([nn.Upsample(scale_factor=2, mode='nearest'), ERPConv(cin, cout)])
            if cout != 3:
                decoder.append(nn.SiLU())
                if c.get('residual_blocks', False):
                    decoder.append(ERPResidual(cout))
        self.decoder = nn.Sequential(*decoder)
        nn.init.constant_(self.decoder[-1].conv.bias, -2.0)
        self.codebook = nn.Embedding(self.codebook_size, c['embedding_dim'])
        nn.init.uniform_(self.codebook.weight, -1 / self.codebook_size, 1 / self.codebook_size)

    @torch.no_grad()
    def initialize_codebook(self, log_hdr, iterations=10):
        """K-means initialization using training features only, before optimizer steps."""
        vectors = self.encoder(log_hdr).permute(0, 2, 3, 1).reshape(-1, self.codebook.embedding_dim).float()
        if vectors.shape[0] < self.codebook_size:
            raise ValueError('Initialization batch needs at least codebook_size latent vectors')
        indices = torch.randperm(vectors.shape[0], device=vectors.device)[:self.codebook_size]
        centers = vectors[indices].clone()
        for _ in range(iterations):
            assignment = torch.cdist(vectors, centers).argmin(1)
            for index in range(self.codebook_size):
                selected = vectors[assignment == index]
                if selected.numel():
                    centers[index] = selected.mean(0)
        self.codebook.weight.copy_(centers)
        return int(torch.cdist(vectors, centers).argmin(1).unique().numel())

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
        factor = config['tokenizer']['downsample_factor']
        if height % factor or width % factor:
            raise ValueError('Environment dimensions must be divisible by tokenizer factor')
        self.grid = (height // factor, width // factor)
        self.length = self.grid[0] * self.grid[1]
        self.bos = config['tokenizer']['codebook_size']
        self.configured_prefix_noise = c.get('prefix_noise_probability', 0.0)
        if not 0 <= self.configured_prefix_noise <= 1:
            raise ValueError('prefix_noise_probability must be in [0, 1]')
        self.encoder_input_size = c.get('encoder_input_size')
        self.cache_frozen_features = c.get('cache_frozen_features', False)
        self._feature_cache = {}
        self.embedding = nn.Embedding(self.bos + 1, c['d_model'])
        # Row position plus periodic longitude; the two seam neighbours have nearby positions.
        rows = torch.arange(self.grid[0]).repeat_interleave(self.grid[1]).float()
        cols = torch.arange(self.grid[1]).repeat(self.grid[0]).float()
        phase = 2 * math.pi * (cols + 0.5) / self.grid[1]
        position_features = [(rows + 0.5) / self.grid[0]]
        for frequency in c.get('position_frequencies', [1]):
            position_features.extend([(phase * frequency).sin(), (phase * frequency).cos()])
        positions = torch.stack(position_features, -1)
        order = torch.arange(self.length)
        ordering = c.get('generation_order', 'raster')
        if ordering == 'visible_first':
            longitude = phase - math.pi
            latitude = math.pi * (0.5 - (rows + 0.5) / self.grid[0])
            x = latitude.cos() * longitude.sin()
            y = latitude.sin()
            z = latitude.cos() * longitude.cos()
            half_horizontal = math.tan(math.radians(config['data']['hfov_degrees']) / 2)
            input_height, input_width = config['data']['input_size']
            half_vertical = half_horizontal * input_height / input_width
            visible = (z > 0) & (x.abs() <= z * half_horizontal) & (y.abs() <= z * half_vertical)
            order = torch.tensor(sorted(range(self.length), key=lambda i: (not bool(visible[i]), -float(z[i]), i)))
        elif ordering != 'raster':
            raise ValueError('Unknown generation_order')
        self.register_buffer('generation_order', order, persistent=False)
        self.register_buffer('positions', positions[order])
        self.position_projection = nn.Linear(positions.shape[-1], c['d_model'])
        if c.get('image_encoder', 'resnet18') == 'tiny':
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU())
            feature_dim = 64
            self.freeze_image_encoder = False
        else:
            from torchvision.models import ResNet18_Weights, resnet18
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
            feature_dim = 512
            self.freeze_image_encoder = c.get('freeze_image_encoder', c['pretrained_image_encoder'])
        self.image_encoder.requires_grad_(not self.freeze_image_encoder)
        self.image_projection = nn.Linear(feature_dim, c['d_model'])
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
        self.use_condition_prior = c.get('condition_prior', False)
        if self.use_condition_prior:
            self.condition_queries = nn.Parameter(torch.randn(self.length, c['d_model']) * 0.02)
            self.condition_decoder = nn.TransformerDecoderLayer(c['d_model'], c['heads'], 4 * c['d_model'],
                dropout=c['dropout'], batch_first=True, norm_first=True)
            self.condition_output = nn.Sequential(nn.LayerNorm(c['d_model']), nn.Linear(c['d_model'], self.bos))
        self.last_condition_logits = None
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_encoder.eval()  # Freeze BatchNorm statistics along with weights.
        return self

    def image_features(self, image):
        if self.encoder_input_size is not None:
            image = F.interpolate(image, size=self.encoder_input_size, mode='bilinear', align_corners=False)
        normalized = (image - self.image_mean) / self.image_std
        if self.cache_frozen_features and self.freeze_image_encoder:
            keys = [hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for value in image]
            missing = [index for index, key in enumerate(keys) if key not in self._feature_cache]
            if missing:
                with torch.no_grad():
                    encoded = self.image_encoder(normalized[missing])
                for index, value in zip(missing, encoded):
                    self._feature_cache[keys[index]] = value.detach().cpu().clone()
            features = torch.stack([self._feature_cache[key].to(normalized.device) for key in keys])
            while len(self._feature_cache) > 256:
                self._feature_cache.pop(next(iter(self._feature_cache)))
        else:
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_image_encoder):
                features = self.image_encoder(normalized)
        h, w = features.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                              torch.linspace(-1, 1, w, device=image.device), indexing='ij')
        position = self.image_position(torch.stack([x, y], -1).reshape(-1, 2))
        return self.image_projection(features.flatten(2).transpose(1, 2)) + position

    def condition_logits(self, memory):
        queries = self.condition_queries + self.position_projection(self.positions)
        return self.condition_output(self.condition_decoder(queries[None].expand(memory.shape[0], -1, -1), memory))

    def next_logits(self, prefix, memory):
        length = prefix.shape[1]
        if not 0 < length <= self.length:
            raise ValueError('Prefix length outside the configured token sequence')
        features = self.embedding(prefix) + self.position_projection(self.positions[:length])
        mask = torch.ones(length, length, dtype=torch.bool, device=prefix.device).triu(1)
        return self.output(self.decoder(features, memory, tgt_mask=mask))

    def sequence_targets(self, target_tokens):
        return target_tokens.flatten(1)[:, self.generation_order]

    def forward(self, image, target_tokens):
        target_tokens = self.sequence_targets(target_tokens)
        if target_tokens.shape[1] != self.length:
            raise ValueError('Target token grid does not match configuration')
        bos = torch.full((image.shape[0], 1), self.bos, dtype=torch.long, device=image.device)
        prefix = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        probability = self.configured_prefix_noise
        if self.training and probability > 0:
            previous = prefix[:, 1:]
            corrupt = torch.rand(previous.shape, device=previous.device) < probability
            previous = torch.where(corrupt, torch.randint_like(previous, self.bos), previous)
            prefix = torch.cat([bos, previous], dim=1)
        memory = self.image_features(image)
        logits = self.next_logits(prefix, memory)
        self.last_condition_logits = self.condition_logits(memory) if self.use_condition_prior else None
        return logits + self.last_condition_logits if self.use_condition_prior else logits

    @torch.no_grad()
    def generate(self, image):
        if self.training:
            raise ValueError('Call eval() before generation')
        memory = self.image_features(image)
        prefix = torch.full((image.shape[0], 1), self.bos, dtype=torch.long, device=image.device)
        prior = self.condition_logits(memory) if self.use_condition_prior else None
        for index in range(self.length):
            logits = self.next_logits(prefix, memory)[:, -1]
            if prior is not None:
                logits = logits + prior[:, index]
            token = logits.argmax(-1, keepdim=True)
            prefix = torch.cat([prefix, token], dim=1)
        grid_tokens = torch.empty_like(prefix[:, 1:])
        grid_tokens[:, self.generation_order] = prefix[:, 1:]
        return grid_tokens.reshape(image.shape[0], *self.grid)
