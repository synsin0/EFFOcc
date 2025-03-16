import torch
import torch.nn as nn
from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.vision_transformer import PatchEmbed

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import numpy as np

import torch
import torch.nn.functional as F


from scipy.cluster.vq import kmeans2
from torch import nn
import torch.distributed.nn.functional
import torch.distributed
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN

# from .video_swin_transformer import PatchEmbed3D
# from .video_swin_transformer import BasicLayer as BasicLayer3D

from einops import rearrange

# from .positional_encoding import get_3d_sincos_pos_embed

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BidirectionalTransformer(nn.Module):
    def __init__(self, n_e, e_dim, img_size, hidden_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.decoder_embed = nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
        token_size = img_size**2
        self.pos_embed = nn.Parameter(torch.zeros(1, token_size, hidden_dim), requires_grad=False)
        self.blocks = BasicLayer(
            hidden_dim,
            (img_size, img_size),
            depth,
            num_heads=num_heads,
            window_size=10,
            downsample=None,
            use_checkpoint=True,
        )
        self.norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())
        self.pred = nn.Linear(hidden_dim, n_e, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, (self.img_size, self.img_size), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        return x


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class BidirectionalVideoTransformer(nn.Module):
#     def __init__(self, n_e, e_dim, img_size, hidden_dim=512, depth=24, num_heads=16):
#         super().__init__()
#         self.n_e = n_e
#         self.e_dim = e_dim
#         self.img_size = img_size
#         self.hidden_dim = hidden_dim
#         self.decoder_embed = nn.Linear(e_dim, hidden_dim, bias=True)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
#         token_size = img_size**2
#         # self.pos_embed = nn.Parameter(torch.zeros(1, token_size, hidden_dim), requires_grad=False)

#         self.blocks = [
#             BasicLayer3D(
#                 dim = hidden_dim,
#                 depth = depth,
#                 num_heads=num_heads,
#                 window_size=(2, 8, 8),
#                 downsample=None,
#                 use_checkpoint=True,
#             ),
#         ]

#         self.blocks = nn.Sequential(*self.blocks)
#         self.norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())
#         self.pred = nn.Linear(hidden_dim, n_e, bias=True)
#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_3d_sincos_pos_embed(embed_dim = self.hidden_dim, grid_size = (self.img_size[1], self.img_size[2]), t_size = self.img_size[0], cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.mask_token, std=0.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         # embed tokens
#         x = self.decoder_embed(x)

#         B, N, C = x.shape
#         img_size = self.img_size
#         n_frames = int(N / img_size / img_size)
#         x = x.view(B, n_frames, img_size, img_size, C).permute(0,4,1,2,3)
#         # add pos embed
#         x = x + self.pos_embed

#         # apply Transformer blocks
#         x = self.blocks(x)

#         x = rearrange(x, 'n c d h w -> n d h w c')
#         x = self.norm(x)
#         x = rearrange(x, 'n d h w c -> n c d h w')

#         x = rearrange(x, 'n c d h w -> n (d h w) c')

#         # predictor projection
#         x = self.pred(x)

#         return x


# @PLUGIN_LAYERS.register_module()
# class VectorQuantizer(nn.Module):
#     def __init__(self, n_e, e_dim, beta, cosine_similarity=False, dead_limit=256):
#         super().__init__()
#         self.n_e = n_e
#         self.e_dim = e_dim
#         self.beta = beta
#         self.cosine_similarity = cosine_similarity
#         self.dead_limit = dead_limit

#         self.embedding = nn.Embedding(self.n_e, self.e_dim)
#         self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

#         self.register_buffer("global_iter", torch.zeros(1))
#         self.register_buffer("num_iter", torch.zeros(1))
#         self.register_buffer("data_initialized", torch.zeros(1))
#         self.register_buffer("reservoir", torch.zeros(self.n_e * 10, e_dim))

#     def train_codebook(self, z, code_age, code_usage):
#         assert z.shape[-1] == self.e_dim
#         z_flattened = z.reshape(-1, self.e_dim)

#         if self.cosine_similarity:
#             z_flattened = F.normalize(z_flattened, p=2, dim=-1)

#         self.update_reservoir(z_flattened, code_age, code_usage)

#         if self.cosine_similarity:
#             min_encoding_indices = torch.matmul(z_flattened, F.normalize(self.embedding.weight, p=2, dim=-1).T).max(
#                 dim=-1
#             )[1]
#         else:
#             # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

#             z_dist = torch.cdist(z_flattened, self.embedding.weight)
#             min_encoding_indices = torch.argmin(z_dist, dim=1)
            

#         z_q = self.embedding(min_encoding_indices).view(z.shape)

#         if self.cosine_similarity:
#             z_q = F.normalize(z_q, p=2, dim=-1)
#             z_norm = F.normalize(z, p=2, dim=-1)
#             loss = (
#                 self.beta * torch.mean(1 - (z_q.detach() * z_norm).sum(dim=-1)),
#                 torch.mean(1 - (z_q * z_norm.detach()).sum(dim=-1)),
#             )
#         else:
#             loss = (self.beta * torch.mean((z_q.detach() - z) ** 2), torch.mean((z_q - z.detach()) ** 2))

#         # preserve gradients
#         z_q = z + (z_q - z).detach()

#         if code_age is not None and code_usage is not None:
#             code_idx = min_encoding_indices
#             if torch.distributed.is_initialized():
#                 code_idx = torch.cat(torch.distributed.nn.functional.all_gather(code_idx))
#             code_age += 1
#             code_age[code_idx] = 0
#             code_usage.index_add_(0, code_idx, torch.ones_like(code_idx, dtype=code_usage.dtype))



#         return z_q, loss, min_encoding_indices



#     def forward(self, z, code_age=None, code_usage=None):
#         z_q, loss, min_encoding_indices = self.train_codebook(z, code_age, code_usage)
#         return z_q, loss, min_encoding_indices

#     def get_codebook_entry(self, indices, shape=None):
#         # shape specifying (batch, height, width, channel)

#         # get quantized latent vectors
#         z_q = self.embedding(indices)

#         if shape is not None:
#             z_q = z_q.view(shape)
#             # reshape back to match original input shape
#             z_q = z_q.permute(0, 3, 1, 2).contiguous()
#         if self.cosine_similarity:
#             z_q = F.normalize(z_q, p=2, dim=-1)
#         return z_q

#     def update_reservoir(self, z, code_age, code_usage):
#         if not (self.embedding.weight.requires_grad and self.training):
#             return

#         assert z.shape[-1] == self.e_dim
#         z_flattened = z.reshape(-1, self.e_dim)

#         rp = torch.randperm(z_flattened.size(0))
#         num_sample: int = self.reservoir.shape[0] // 100  # pylint: disable=access-member-before-definition
#         self.reservoir: torch.Tensor = torch.cat([self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data])

#         self.num_iter += 1
#         self.global_iter += 1

#         if ((code_age >= self.dead_limit).sum() / self.n_e) > 0.03 and (
#             self.data_initialized.item() == 0 or self.num_iter.item() > 1000
#         ):
#             self.update_codebook(code_age, code_usage)
#             if self.data_initialized.item() == 0:
#                 self.data_initialized.fill_(1)

#             self.num_iter.fill_(0)

#     def update_codebook(self, code_age, code_usage):
#         if not (self.embedding.weight.requires_grad and self.training):
#             return

#         if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#             live_code = self.embedding.weight[code_age < self.dead_limit].data
#             live_code_num = live_code.shape[0]
#             if self.cosine_similarity:
#                 live_code = F.normalize(live_code, p=2, dim=-1)

#             all_z = torch.cat([self.reservoir, live_code])
#             rp = torch.randperm(all_z.shape[0])
#             all_z = all_z[rp]

#             init = torch.cat(
#                 [live_code, self.reservoir[torch.randperm(self.reservoir.shape[0])[: (self.n_e - live_code_num)]]]
#             )
#             init = init.data.cpu().numpy()
#             print(
#                 "running kmeans!!", self.n_e, live_code_num, self.data_initialized.item()
#             )  # data driven initialization for the embeddings
#             centroid, assignment = kmeans2(
#                 all_z.cpu().numpy(),
#                 init,
#                 minit="matrix",
#                 iter=50,
#             )
#             z_dist = (all_z - torch.from_numpy(centroid[assignment]).to(all_z.device)).norm(dim=1).sum().item()
#             self.embedding.weight.data = torch.from_numpy(centroid).to(self.embedding.weight.device)

#             print("finish kmeans", z_dist)

#         if torch.distributed.is_initialized():
#             torch.distributed.nn.functional.broadcast(self.embedding.weight, src=0)

#         code_age.fill_(0)
#         code_usage.fill_(0)
#         # self.data_initialized.fill_(1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class VQEncoder(nn.Module):
#     def __init__(
#         self,
#         img_size,
#         patch_size=8,
#         in_chans=40,
#         embed_dim=512,
#         num_heads=16,
#         depth=12,
#         codebook_dim=1024,
#     ):
#         super().__init__()

#         norm_layer = nn.LayerNorm
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer=norm_layer)
#         num_patches = self.patch_embed.num_patches

#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding

#         self.h = img_size // patch_size
#         self.w = img_size // patch_size

#         self.blocks = [
#             BasicLayer(
#                 embed_dim,
#                 (img_size // patch_size, img_size // patch_size),
#                 depth,
#                 num_heads=num_heads,
#                 window_size=8,
#                 downsample=None,
#                 use_checkpoint=True,
#             ),
#         ]

#         self.blocks = nn.Sequential(*self.blocks)

#         self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
#         self.pre_quant = nn.Linear(embed_dim, codebook_dim)

#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#         # nn.init.constant_(self.pre_quant.weight, 0)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         # embed patches
#         x = self.patch_embed(x)

#         # add pos embed w/o cls token
#         x = x + self.pos_embed

#         # apply Transformer blocks
#         x = self.blocks(x)
#         x = self.norm(x)
#         x = self.pre_quant(x)

#         return x


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class VQEncoder3D(nn.Module):
#     def __init__(
#         self,
#         img_size,
#         patch_size=(2,8,8),
#         in_chans=40,
#         embed_dim=512,
#         num_heads=16,
#         depth=12,
#         codebook_dim=1024,
#     ):
#         super().__init__()

#         norm_layer = nn.LayerNorm
#         self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim, norm_layer=norm_layer)
#         # num_patches = self.patch_embed.num_patches
#         num_patches = img_size[0] * img_size[1] * img_size[2]
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding

#         # self.h = img_size // patch_size
#         # self.w = img_size // patch_size
#         self.embed_dim = embed_dim
#         self.img_size = img_size
#         self.blocks = [
#             BasicLayer3D(
#                 dim = embed_dim,
#                 depth = depth,
#                 num_heads=num_heads,
#                 window_size=(2, 8, 8),
#                 downsample=None,
#                 use_checkpoint=True,
#             ),
#         ]

#         self.blocks = nn.Sequential(*self.blocks)

#         self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
#         self.pre_quant = nn.Linear(embed_dim, codebook_dim)
    
#         self.initialize_weights()


#     def initialize_weights(self):
       
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding

#         pos_embed = get_3d_sincos_pos_embed(embed_dim = self.embed_dim, grid_size = (self.img_size[1], self.img_size[2]), t_size = self.img_size[0], cls_token=False)

#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#         # nn.init.constant_(self.pre_quant.weight, 0)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         # embed patches
#         x = self.patch_embed(x)
        
#         import ipdb
#         ipdb.set_trace()
#         # add pos embed w/o cls token
#         x = x + self.pos_embed

#         # apply Transformer blocks
#         x = self.blocks(x)

#         x = rearrange(x, 'n c d h w -> n d h w c')
#         x = self.norm(x)
#         x = rearrange(x, 'n d h w c -> n c d h w')

#         x = rearrange(x, 'n c d h w -> n (d h w) c')

#         x = self.pre_quant(x)

#         return x

# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class VQDecoder3D(nn.Module):
#     def __init__(
#         self,
#         img_size,
#         # num_patches,
#         patch_size=(2,8,8),
#         in_chans=40,
#         embed_dim=512,
#         num_heads=16,
#         depth=12,
#         codebook_dim=1024,
#         bias_init=-3,
#     ):
#         super().__init__()

#         if isinstance(img_size, int):
#             img_size = (img_size, img_size)

#         norm_layer = nn.LayerNorm
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.h = img_size[1] // patch_size[1]
#         self.w = img_size[2] // patch_size[2]
#         self.t = img_size[0] // patch_size[0]
#         num_patches = self.h * self.w * self.t
#         self.num_patches = num_patches
#         self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding

#         self.blocks = [
#             BasicLayer3D(
#                 dim = embed_dim,
#                 depth = depth,
#                 num_heads=num_heads,
#                 window_size=(2, 8, 8),
#                 use_checkpoint=True,
#             ),
#         ]
#         self.blocks = nn.Sequential(*self.blocks)

#         self.embed_dim = embed_dim

#         self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
#         self.pred = nn.Linear(embed_dim, (patch_size[0] * patch_size[1] * patch_size[2]) * in_chans, bias=True)
#         self.initialize_weights()
#         nn.init.constant_(self.pred.bias, bias_init)

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_3d_sincos_pos_embed(embed_dim = self.embed_dim, grid_size = (self.h, self.w), t_size = self.t, cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def unpatchify(self, x):
#         p = self.patch_size
#         # h = w = int(x.shape[1] ** 0.5)
#         h, w, t = self.h, self.w, self.t    

#         x = x.reshape(shape=(x.shape[0], t, h, w, p[0], p[1], p[2], self.in_chans))
#         x = torch.einsum("nthwrpqc->nctrhpwq", x)
#         imgs = x.reshape(shape=(x.shape[0], self.in_chans, t * p[0], h * p[1], w * p[2]))

#         return imgs

#     def forward(self, x):
#         # embed tokens
#         x = self.decoder_embed(x)

#         # add pos embed
#         x = x + self.pos_embed


#         B, N, C = x.shape
#         n_frames = int(N / self.h / self.w)
#         x = x.view(B, n_frames, self.h, self.w, C).permute(0,4,1,2,3)
#         # add pos embed
        
#         # apply Transformer blocks
#         x = self.blocks(x)
#         x = rearrange(x, 'n c d h w -> n d h w c')
#         x = self.norm(x)
#         x = rearrange(x, 'n d h w c -> n c d h w')
#         x = rearrange(x, 'n c d h w -> n (d h w) c')
#         # predictor projection
#         x = self.pred(x)
#         x = self.unpatchify(x)

#         return x

# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class VQDecoder(nn.Module):
#     def __init__(
#         self,
#         img_size,
#         num_patches,
#         patch_size=8,
#         in_chans=40,
#         embed_dim=512,
#         num_heads=16,
#         depth=12,
#         codebook_dim=1024,
#         bias_init=-3,
#     ):
#         super().__init__()

#         if isinstance(img_size, int):
#             img_size = (img_size, img_size)

#         norm_layer = nn.LayerNorm
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.h = img_size[0] // patch_size
#         self.w = img_size[1] // patch_size
#         self.num_patches = num_patches
#         self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches, embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding

#         self.blocks = BasicLayer(
#             embed_dim,
#             (img_size[0] // patch_size, img_size[1] // patch_size),
#             depth=depth,
#             num_heads=num_heads,
#             window_size=8,
#             use_checkpoint=True,
#         )

#         self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
#         self.pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
#         self.initialize_weights()
#         nn.init.constant_(self.pred.bias, bias_init)

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def unpatchify(self, x):
#         p = self.patch_size
#         # h = w = int(x.shape[1] ** 0.5)
#         h, w = self.h, self.w
#         assert h * w == x.shape[1]

#         x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
#         x = torch.einsum("nhwpqc->nchpwq", x)
#         imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))

#         return imgs

#     def forward(self, x):
#         # embed tokens
#         x = self.decoder_embed(x)

#         # add pos embed
#         x = x + self.pos_embed

#         # apply Transformer blocks
#         x = self.blocks(x)
#         x = self.norm(x)

#         # predictor projection
#         x = self.pred(x)
#         x = self.unpatchify(x)

#         return x
