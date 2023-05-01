import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
from src.vit_spatial_spectral import ViTSpatialSpectral_V1, ViTSpatialSpectral


class BlockwiseToPixels(nn.Module):
    def __init__(self, dim, num_spectral_blocks, pixels_per_patch, precision):
        super().__init__()
        self.pixels_per_patch = pixels_per_patch
        self.layers = nn.ModuleList(
            [nn.Linear(dim, pixels_per_patch) for _ in range(num_spectral_blocks)]
        )
        if precision == "16-mixed":
            self.dtype = torch.float16
        elif precision == "32-true":
            self.dtype = torch.float32

    def forward(self, x, block_indices):
        # x.shape: b n d
        # block_indices.shape: b n

        out = torch.empty(
            x.shape[0],
            x.shape[1],
            self.pixels_per_patch,
            device=x.device,
            dtype=self.dtype,
        )
        # for b in range(x.shape[0]):
        #     for n in range(x.shape[1]):
        #         out[b,n] = self.layers[block_indices[b,n]](x[b,n])

        for idx, layer in enumerate(self.layers):
            spectral_idx = block_indices == idx  # .nonzero()
            out[spectral_idx] = layer(x[spectral_idx])

        return out


class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio=0.5,
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        pixel_values_per_patch = (
            self.encoder.near_band * self.encoder.image_size**2
        )  # self.patch_to_emb[1].weight.shape[-1]

        self.to_patch = encoder.patch_to_embedding[:2]
        self.patch_to_emb = encoder.patch_to_embedding[2]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device=device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1 : (num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        # return recon_loss
        return recon_loss, pred_pixel_values, masked_patches, masked_indices, encoded


class SimMIMSpatialSpectral(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio=0.5,
        mask_patch_size=1,
        tube_masking=False,
        intermediate_losses=False,
        to_pixels_per_spectral_block=False,
        precision="32-true",
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio
        self.mask_patch_size = mask_patch_size
        self.intermediate_losses = intermediate_losses
        self.to_pixels_per_spectral_block = to_pixels_per_spectral_block
        self.tube_masking = tube_masking

        if self.mask_patch_size != 1:
            self.mask_generator = MaskGenerator(
                input_size=encoder.image_size,
                mask_patch_size=mask_patch_size,
                model_patch_size=encoder.patch_height,
                mask_ratio=self.masking_ratio,
            )

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        if isinstance(encoder, ViTSpatialSpectral_V1):
            num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
            self.to_patch, self.patch_to_emb = (
                encoder.to_patch_embedding[0],
                encoder.to_patch_embedding[1:],
            )
            self.pixel_values_per_patch = self.patch_to_emb[1].weight.shape[-1]

        elif isinstance(encoder, ViTSpatialSpectral):
            encoder_dim = encoder.dim
            self.to_patch = encoder.to_patch_embedding.to_patch
            self.patch_to_emb = encoder.to_patch_embedding.embed
            self.pixel_values_per_patch = encoder.pixels_per_patch

        #        elif isinstance(encoder, SwinTransformer3D):
        #           encoder_dim = encoder.embed_dim
        #           self.to_patch = encoder.patch_embed.to_patch
        #           self.patch_to_emb = encoder.patch_embed.embed
        #           self.pixel_values_per_patch = encoder.patch_size[0] * encoder.patch_size[1] * encoder.patch_size[2]

        # simple linear head
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        if self.to_pixels_per_spectral_block:
            self.to_pixels = BlockwiseToPixels(
                encoder_dim,
                encoder.num_spectral_patches,
                self.pixel_values_per_patch,
                precision=precision,
            )
        else:
            self.to_pixels = nn.Linear(encoder_dim, self.pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes
        batch_range = torch.arange(batch, device=device)[:, None]

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)

        #       if isinstance(self.encoder, SwinTransformer3D):
        #           d,h,w = tokens.shape[2:]
        #           tokens = rearrange(tokens, 'b c d h w -> b (d h w) c', )
        #           # patches = rearrange(patches, 'b c (d p) h w -> b (c d h w) p', p=self.pixel_values_per_patch)
        #           patches = rearrange(patches, 'b c (d p0) (h p1) (w p2) -> b (c d h w) (p0 p1 p2)',
        #                               p0=self.encoder.patch_size[0],
        #                               p1=self.encoder.patch_size[1],
        #                               p2=self.encoder.patch_size[2],
        #                               )
        if isinstance(self.encoder, ViTSpatialSpectral):
            if self.encoder.blockwise_patch_embed:
                patches = rearrange(patches, "b g n d -> b (g n) d")

        num_patches = tokens.shape[1]

        # get positions
        if isinstance(self.encoder, ViTSpatialSpectral_V1):
            pos_emb = self.encoder.pos_embedding[:, 1 : (num_patches + 1)]
            tokens = tokens + pos_emb

        elif isinstance(self.encoder, ViTSpatialSpectral):
            if self.encoder.spectral_pos_embed:
                pos_embed = self.encoder.get_pos_embeddings()
            else:
                pos_embed = self.encoder.pos_embedding[:, : tokens.shape[1]]

            tokens = tokens + pos_embed

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        if isinstance(self.encoder, ViTSpatialSpectral_V1):
            mask_tokens = mask_tokens + pos_emb
        elif isinstance(self.encoder, ViTSpatialSpectral):
            mask_tokens = mask_tokens + pos_embed

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)

        if self.mask_patch_size == 1:
            masked_indices = (
                torch.rand(batch, num_patches, device=device)
                .topk(k=num_masked, dim=-1)
                .indices
            )
            masked_bool_mask = (
                torch.zeros((batch, num_patches), device=device)
                .scatter_(-1, masked_indices, 1)
                .bool()
            )
        else:
            if self.tube_masking:
                (
                    masked_bool_mask,
                    masked_indices,
                ) = self.mask_generator.get_batch_tube_masked(
                    batch_size=batch,
                    channel_tokens=self.encoder.num_spectral_patches,
                    num_masked=num_masked,
                    device=device,
                )
            else:
                masked_bool_mask, masked_indices = self.mask_generator.get_batch(
                    batch_size=batch,
                    channel_tokens=self.encoder.num_spectral_patches,
                    num_masked=num_masked,
                    device=device,
                )

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        #        if isinstance(self.encoder, SwinTransformer3D):
        #            tokens = rearrange(tokens, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        if isinstance(self.encoder, ViTSpatialSpectral_V1):
            (
                encoded,
                encoded_spatial,
                encoded_spectral,
            ) = self.encoder.transformer_forward(tokens)
        elif isinstance(self.encoder, ViTSpatialSpectral):
            encoded = self.encoder.transformer_forward(tokens)
        else:
            encoded = self.encoder.transformer_forward(tokens)
            encoded = rearrange(encoded, "b c d h w -> b (d h w) c")

        if self.intermediate_losses:
            # predict pixels from final token representation *and* from earlier spatial/spectral transformer representations
            encoded_tokens = [encoded, encoded_spatial, encoded_spectral]
        else:
            # loss only based on final representation
            encoded_tokens = [encoded]

        recon_loss = 0.0

        for encoded in encoded_tokens:
            # get the masked tokens
            encoded_mask_tokens = encoded[batch_range, masked_indices]

            # small linear projection for predicted pixel values
            if self.to_pixels_per_spectral_block:
                spectral_block_idx = torch.arange(
                    self.encoder.num_spectral_patches
                ).repeat_interleave(self.encoder.num_spatial_patches)
                spectral_block_idx = (
                    spectral_block_idx.unsqueeze(0).repeat(batch, 1).to(device)
                )
                masked_spectral_block_idx = spectral_block_idx[
                    batch_range, masked_indices
                ]

                pred_pixel_values = self.to_pixels(
                    encoded_mask_tokens, masked_spectral_block_idx
                )
            else:
                pred_pixel_values = self.to_pixels(encoded_mask_tokens)

            # get the masked patches for the final reconstruction loss
            masked_patches = patches[batch_range, masked_indices]

            # calculate reconstruction loss
            recon_loss += F.l1_loss(pred_pixel_values, masked_patches) / num_masked

        return recon_loss


class MaskGenerator:
    # adapted from from https://github.com/microsoft/SimMIM
    def __init__(
        self, input_size=16, mask_patch_size=4, model_patch_size=1, mask_ratio=0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask

    def bool_mask_to_indices(self, masked_bool_mask, batch, num_masked, device):

        masked_idx_tmp = masked_bool_mask.nonzero()
        masked_idx = torch.empty((batch, num_masked), device=device, dtype=int)

        for idx in range(batch):
            masked_idx[idx, :] = masked_idx_tmp[:, 1][
                (num_masked * idx) : (num_masked * (idx + 1))
            ]

        return masked_idx

    def get_batch(self, batch_size, channel_tokens, num_masked, device):
        bool_mask = rearrange(
            torch.stack(
                [
                    torch.tensor(self(), dtype=bool)
                    for _ in range(batch_size * channel_tokens)
                ]
            ),
            "(b c) h w -> b c h w",
            b=batch_size,
            c=channel_tokens,
        )
        bool_mask_flat = rearrange(bool_mask, "b c h w -> b (c h w)").to(device)

        idx_mask = self.bool_mask_to_indices(
            bool_mask_flat, batch_size, num_masked, device
        )

        return bool_mask_flat, idx_mask

    def get_batch_tube_masked(self, batch_size, channel_tokens, num_masked, device):
        bool_mask = (
            torch.stack([torch.tensor(self(), dtype=bool) for _ in range(batch_size)])
            .unsqueeze(1)
            .repeat(1, channel_tokens, 1, 1)
        )
        bool_mask_flat = rearrange(bool_mask, "b c h w -> b (c h w)").to(device)

        idx_mask = self.bool_mask_to_indices(
            bool_mask_flat, batch_size, num_masked, device
        )

        return bool_mask_flat, idx_mask
