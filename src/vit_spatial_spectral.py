import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import reduce
from operator import mul

from src.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT_original(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class BlockwisePatchEmbedding(nn.Module):
    def __init__(
        self, num_channels, transformer_dim, patch_depth, patch_height, patch_width
    ):
        super().__init__()
        assert (
            num_channels % patch_depth == 0
        ), f"Number of channels {num_channels=} not divisible by patch_depth {patch_depth=}"
        self.patch_depth = patch_depth
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.transformer_dim = transformer_dim

        self.patch_dim = reduce(mul, [patch_depth, patch_height, patch_width])
        self.num_blocks = num_channels // patch_depth

        self.pre_norm = nn.LayerNorm(self.patch_dim)
        self.post_norm = nn.LayerNorm(self.transformer_dim)

        self.to_patch = Rearrange(
            "b (c p0) (h p1) (w p2) -> b c (h w) (p0 p1 p2)",
            p0=self.patch_depth,
            p1=self.patch_height,
            p2=self.patch_width,
        )
        self.blockwise_embed = nn.ModuleList(
            [
                nn.Linear(self.patch_dim, self.transformer_dim)
                for _ in range(self.num_blocks)
            ]
        )

    def embed(self, patches):
        patches = self.pre_norm(patches)

        embeds = []
        for i in range(self.num_blocks):
            embeds.append(self.blockwise_embed[i](patches[:, i, :, :]))

        embeds = torch.stack(embeds, dim=1)  # .flatten(start_dim=1, end_dim=2)
        embeds = rearrange(embeds, "b g n d -> b (g n) d")

        embeds = self.post_norm(embeds)

        return embeds

    def forward(self, x):
        patches = self.to_patch(x)

        embeddings = self.embed(patches)

        return embeddings


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_dim, patch_depth, patch_height, patch_width):
        super().__init__()
        self.to_patch = nn.Sequential(
            Rearrange(
                "b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)",
                p0=patch_depth,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
        )
        self.embed = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        patches = self.to_patch(x)
        embeddings = self.embed(patches)

        return embeddings


class ViTSpatialSpectral(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        spatial_patch_size,
        spectral_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        spectral_pos_embed=True,
        pool="mean",
        blockwise_patch_embed=True,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        spectral_pos=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        spectral_only=False,
        spectral_mlp_head=False,
        pixelwise=False,
        pos_embed_len=None,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        image_depth = channels
        self.patch_height, self.patch_width = pair(spatial_patch_size)
        self.patch_depth = spectral_patch_size
        self.image_size = image_size
        self.pixels_per_patch = reduce(
            mul, [self.patch_depth, self.patch_height, self.patch_width]
        )
        self.spectral_pos = np.array(spectral_pos)
        self.spectral_pos_embed = spectral_pos_embed
        self.blockwise_patch_embed = blockwise_patch_embed
        self.spectral_only = spectral_only
        self.spectral_mlp_head = spectral_mlp_head
        self.pixelwise = pixelwise  # make one prediction per image (i.e., for center pixel, inference with sliding window)

        assert (
            image_height % self.patch_height == 0
            and image_width % self.patch_width == 0
            and image_depth % self.patch_depth == 0
        ), f"Image dimensions must be divisible by the patch size. {image_height=}, {self.patch_height=}, {image_width=}, {self.patch_width=}, {image_depth=}, {self.patch_depth=}"

        self.num_spatial_patches_sqrt = image_height // self.patch_height
        self.num_spatial_patches = self.num_spatial_patches_sqrt**2
        self.num_spectral_patches = image_depth // self.patch_depth

        self.num_patches = self.num_spatial_patches * self.num_spectral_patches
        assert pool in {
            "mean"
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        if self.blockwise_patch_embed:
            self.to_patch_embedding = BlockwisePatchEmbedding(
                channels, dim, self.patch_depth, self.patch_height, self.patch_width
            )
        else:
            patch_dim = reduce(
                mul, [self.patch_depth, self.patch_height, self.patch_width]
            )
            self.to_patch_embedding = PatchEmbed(
                dim, patch_dim, self.patch_depth, self.patch_height, self.patch_width
            )
        # b,c,h,w to b,n,d with n number of tokens and d their dimensionality
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)', p0 = self.patch_depth, p1 = self.patch_height, p2 = self.patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )

        if self.spectral_pos_embed:
            channel_embed_dim = (
                dim // 3
            )  # allot 1/3 of the positional embedding vector to the channel position embedding
            pos_embed_dim = dim - channel_embed_dim

            # spatial positional embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_spatial_patches, pos_embed_dim)
            )
            p_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.num_spatial_patches_sqrt, cls_token=False
            )
            self.pos_embed.data.copy_(
                torch.from_numpy(p_embed).float().unsqueeze(0)
            ).shape

            # spectral positional embedding
            assert (
                len(self.spectral_pos) == self.num_spectral_patches
            ), f"{self.spectral_pos.shape=}, {self.num_spectral_patches=}"
            self.channel_embed = nn.Parameter(
                torch.zeros(1, self.num_spectral_patches, channel_embed_dim)
            )
            chan_embed = get_1d_sincos_pos_embed_from_grid(
                self.channel_embed.shape[-1], self.spectral_pos
            )
            self.channel_embed.data.copy_(
                torch.from_numpy(chan_embed).float().unsqueeze(0)
            ).shape

        else:
            if pos_embed_len is not None:
                self.pos_embedding = nn.Parameter(torch.randn(1, pos_embed_len, dim))
            else:
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, self.num_patches + 1, dim)
                )

        self.dropout = nn.Dropout(emb_dropout)

        if self.spectral_only:
            self.spatial_spectral_transformer = nn.Sequential(
                Rearrange(
                    "b (c h w) d -> (b h w) c d",
                    c=self.num_spectral_patches,
                    h=self.num_spatial_patches_sqrt,
                    w=self.num_spatial_patches_sqrt,
                ),
                Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
                Rearrange(
                    "(b h w) c d -> b (c h w) d",
                    c=self.num_spectral_patches,
                    h=self.num_spatial_patches_sqrt,
                    w=self.num_spatial_patches_sqrt,
                ),
            )
        else:
            self.spatial_spectral_transformer = nn.Sequential(
                Rearrange(
                    "b (c h w) d -> (b c) (h w) d",
                    c=self.num_spectral_patches,
                    h=self.num_spatial_patches_sqrt,
                    w=self.num_spatial_patches_sqrt,
                ),
                Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
                Rearrange(
                    "(b c) (h w) d -> (b h w) c d",
                    c=self.num_spectral_patches,
                    h=self.num_spatial_patches_sqrt,
                    w=self.num_spatial_patches_sqrt,
                ),
                Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
                Rearrange(
                    "(b h w) c d -> b (c h w) d",
                    c=self.num_spectral_patches,
                    h=self.num_spatial_patches_sqrt,
                    w=self.num_spatial_patches_sqrt,
                ),
            )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.dim = dim

        num_out_pixels = self.patch_width * self.patch_height

        # pixelwise classification
        if self.spectral_mlp_head:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim * self.num_spectral_patches),
                nn.Linear(
                    dim * self.num_spectral_patches, num_classes * num_out_pixels
                ),  # map spatial patch to classes per pixels
                Rearrange(
                    "b h w (p1 p2 num_classes) -> b (h p1) (w p2) num_classes",
                    p1=self.patch_height,
                    p2=self.patch_width,
                    num_classes=num_classes,
                ),
                MoveAxis((-1, 1)),
            )
        else:
            if self.pixelwise:
                # with avg pooling
                # self.mlp_head = nn.Sequential(
                #     nn.LayerNorm(dim),
                #     Mean(axis=(1,2)), # average pool over h,w
                #     nn.Linear(dim, num_classes),
                #     Rearrange('b (p1 p2 num_classes) -> b p1 p2 num_classes', p1=self.patch_height, p2=self.patch_width, num_classes=num_classes),
                #     MoveAxis((-1,1)),
                #     Squeeze(),
                #  )

                # with large fc layer
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    Flatten(start_dim=1, end_dim=-1),  # flatten over n,d dimensions
                    nn.Linear(dim * self.num_spatial_patches, num_classes),
                    Rearrange(
                        "b (p1 p2 num_classes) -> b p1 p2 num_classes",
                        p1=self.patch_height,
                        p2=self.patch_width,
                        num_classes=num_classes,
                    ),
                    MoveAxis((-1, 1)),
                    Squeeze(),
                )
            else:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(
                        dim, num_classes * num_out_pixels
                    ),  # map spatial patch to classes per pixels
                    Rearrange(
                        "b h w (p1 p2 num_classes) -> b (h p1) (w p2) num_classes",
                        p1=self.patch_height,
                        p2=self.patch_width,
                        num_classes=num_classes,
                    ),
                    MoveAxis((-1, 1)),
                )

    def transformer_forward(self, x):

        x = self.spatial_spectral_transformer(x)

        return x

    def get_pos_embeddings(self):
        # add channel embed
        channel_embed = self.channel_embed.unsqueeze(2)  # (1, c, 1, cD)
        pos_embed = self.pos_embed.unsqueeze(1)

        # Channel embed same across (x,y) position, and pos embed same across channel (c)
        channel_embed = channel_embed.expand(
            -1, -1, pos_embed.shape[2], -1
        )  # (1, c, L, cD)
        pos_embed = pos_embed.expand(
            -1, channel_embed.shape[1], -1, -1
        )  # (1, c, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, c, L, D)
        pos_channel = rearrange(pos_channel, "b g n d -> b (g n) d")

        return pos_channel

    def forward_features(self, img):
        # tokenize/embed, add pos encoding, pass through transformer
        x = self.to_patch_embedding(img)

        if self.spectral_pos_embed:
            pos_embed = self.get_pos_embeddings()
        else:
            pos_embed = self.pos_embedding[:, : x.shape[1]]

        # add pos embed w/o cls token
        x = x + pos_embed  #

        x = self.dropout(x)

        x = self.transformer_forward(x)

        return x

    def forward(self, img):
        x = self.forward_features(img)

        if self.spectral_mlp_head:
            x = rearrange(
                x,
                "b (c h w) d -> b h w (c d)",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches_sqrt,
                w=self.num_spatial_patches_sqrt,
            )

        else:
            # map tokens back into spatial-spectral cube
            x = rearrange(
                x,
                "b (c h w) d -> b c h w d",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches_sqrt,
                w=self.num_spatial_patches_sqrt,
            )

            # avg pool / extract cls token and pass through mlp head
            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x


class AvgPoolMerge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # (b,n,d)
        x = torch.stack((x1, x2), dim=1)
        x = x.mean(axis=1)

        return x


class LinearMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(2 * dim, dim)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=2)
        x = self.fc(x)

        return x


class MoveAxis(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return torch.moveaxis(x, *self.axes)


class ViTSpatialSpectral_V1(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        spatial_patch_size,
        spectral_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="mean",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        merge="avgpool",
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        image_depth = channels
        self.patch_height, self.patch_width = pair(spatial_patch_size)
        self.patch_depth = spectral_patch_size
        self.image_size = image_size

        assert (
            image_height % self.patch_height == 0
            and image_width % self.patch_width == 0
            and image_depth % self.patch_depth == 0
        ), "Image dimensions must be divisible by the patch size."

        self.num_spatial_patches = image_height // self.patch_height
        self.num_spectral_patches = image_depth // self.patch_depth

        self.num_patches = self.num_spatial_patches**2 * self.num_spectral_patches
        patch_dim = self.patch_depth * self.patch_height * self.patch_width
        assert pool in {
            "mean"
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # b,c,h,w to b,n,d with n number of tokens and d their dimensionality
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b (c p0) (h p1) (w p2) -> b (c h w) (p0 p1 p2)",
                p0=self.patch_depth,
                p1=self.patch_height,
                p2=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.spatial_transformer = nn.Sequential(
        #     # move spectral dim into batch dim
        #     Rearrange('b (c h w) d -> (b c) (h w) d', c=self.num_spectral_patches, h=self.num_spatial_patches, w=self.num_spatial_patches),
        #     Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
        #     Rearrange('(b c) (h w) d -> b (c h w) d', c=self.num_spectral_patches, h=self.num_spatial_patches, w=self.num_spatial_patches),
        # )
        # self.spectral_transformer = nn.Sequential(
        #     # move spatial dims into batch dim
        #     Rearrange('b (c h w) d -> (b h w) c d', c=self.num_spectral_patches, h=self.num_spatial_patches, w=self.num_spatial_patches),
        #     Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
        #     Rearrange('(b h w) c d -> b (c h w) d', c=self.num_spectral_patches, h=self.num_spatial_patches, w=self.num_spatial_patches),
        #  )

        self.spatial_spectral_transformer = nn.Sequential(
            Rearrange(
                "b (c h w) d -> (b c) (h w) d",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches,
                w=self.num_spatial_patches,
            ),
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
            Rearrange(
                "(b c) (h w) d -> (b h w) c d",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches,
                w=self.num_spatial_patches,
            ),
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout),
            Rearrange(
                "(b h w) c d -> b (c h w) d",
                c=self.num_spectral_patches,
                h=self.num_spatial_patches,
                w=self.num_spatial_patches,
            ),
        )

        if merge == "avgpool":
            self.merge = AvgPoolMerge()
        elif merge == "linear":
            self.merge = LinearMerge(dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        # patch classification
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes),
        # )

        # pixelwise classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(
                dim, num_classes * (self.patch_width * self.patch_height)
            ),  # map spatial patch to classes per pixels
            Rearrange(
                "b h w (p1 p2 num_classes) -> b (h p1) (w p2) num_classes",
                p1=self.patch_height,
                p2=self.patch_width,
                num_classes=num_classes,
            ),
            MoveAxis((-1, 1)),
        )

    def transformer_forward(self, x):
        # x1 = self.spatial_transformer(x)
        # x2 = self.spectral_transformer(x)

        # x = self.merge(x1, x2)

        x = self.spatial_spectral_transformer(x)

        # return x, x1, x2
        return x, x, x

    def forward_features(self, img):
        # tokenize/embed, add cls token and pos encoding, pass through transformer
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x, x1, x2 = self.transformer_forward(x)

        return x, x1, x2

    def forward(self, img):
        x, _, _ = self.forward_features(img)

        # map tokens back into spatial-spectral cube
        x = rearrange(
            x,
            "b (c h w) d -> b c h w d",
            c=self.num_spectral_patches,
            h=self.num_spatial_patches,
            w=self.num_spatial_patches,
        )

        # avg pool / extract cls token and pass through mlp head
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def get_pos_for_spectral_embedding(
    spectral_patch_depth, wavelengths, reference_wavelengths
):
    """For each spectral block in wavelengths, return the index of the
    closest spectral block in reference_wavelengths
    usecase: model pre-trained on reference_wavelengths with positional encodings of the
    sequence of spectral blocks. Encode the spectral range of a different sensor based on
    the learned embeddings (e.g. 5th corresponds to avg. wavelength 700 in pre-training).
    """
    wavelengths = np.array(wavelengths)
    reference_wavelengths = np.array(reference_wavelengths)
    block_means = []
    total = len(wavelengths)
    if total % spectral_patch_depth != 0:
        total = len(wavelengths) + (
            spectral_patch_depth - len(wavelengths) % spectral_patch_depth
        )
    for i in range(0, total, spectral_patch_depth):

        block_means.append(wavelengths[i : i + spectral_patch_depth].mean())

    reference_block_means = []
    total = len(reference_wavelengths)
    if total % spectral_patch_depth != 0:
        total = len(reference_wavelengths) + (
            spectral_patch_depth - len(reference_wavelengths) % spectral_patch_depth
        )
    for i in range(0, total, spectral_patch_depth):
        reference_block_means.append(
            reference_wavelengths[i : i + spectral_patch_depth].mean()
        )

    # for each mean, find the idx of the closest mean of the reference wavelengths
    return [np.argmin(np.abs(reference_block_means - m)) for m in block_means]


class Mean(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.mean(axis=self.axis)


class Flatten(nn.Module):
    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()
