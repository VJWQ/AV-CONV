import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, embed_dim, ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.0,
        mlp_ratio=4.0,
        mlp_dropout=0.0,
        attn_type="",
        num_frame=8,
        num_sub=4,
    ):
        super().__init__()
        self.attention_type = attn_type
        self.types = self.attention_type.split("_")
        self.num_frame = num_frame
        self.num_sub = num_sub
        print("self.types", self.types, len(self.types), self.num_sub)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        if "T" in self.types:
            self.temporal_norm1 = nn.LayerNorm(embed_dim)
            self.temporal_attn = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
            )
            if len(self.types) > 1:
                self.temporal_fc = nn.Linear(embed_dim, embed_dim)
        if "N" in self.types:
            self.subject_norm1 = nn.LayerNorm(embed_dim)
            self.subject_attn = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
            )
            if len(self.types) > 1:
                self.subject_fc = nn.Linear(embed_dim, embed_dim)
        if "S" in self.types:
            self.spatial_norm1 = nn.LayerNorm(embed_dim)
            self.spatial_attn = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
            )
            # self.spatial_fc = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, ratio=mlp_ratio, dropout=mlp_dropout)

    def forward(self, x):
        """
        input x shape: [BN, TS, MC]
        """
        BN, TS, MC = x.shape  # [BN, TS, MC]
        N = self.num_sub
        B = BN // N
        T = self.num_frame
        S = TS // T
        x_out = x

        if self.attention_type == 'T_N_S':
            # ============= Temporal =============
            # [BN, TS, MC] -> [BNS, T, MC]
            xt = x_out.reshape(BN, T, S, MC).permute(0, 2, 1, 3).reshape(BN * S, T, MC)
            res_temporal = self.dropout(self.temporal_attn(self.temporal_norm1(xt)))
            # [BNS, T, MC] -> [BN, S, T, MC] -> [BN, T, S, MC] -> [BN, TS, MC]
            res_temporal = res_temporal.reshape(BN, S, T, MC).permute(0, 2, 1, 3).reshape(BN, TS, MC)
            res_temporal = self.temporal_fc(res_temporal)
            x_out = x_out + res_temporal

            # ============= Subject =============
            # [BN, TS, MC] -> [B, N, T, S, MC] -> [B, T, S, N, MC] -> [BTS, N, MC]
            xn = x_out.reshape(B, N, T, S, MC).permute(0, 2, 3, 1, 4).reshape(B * T * S, N, MC)
            res_subject = self.dropout(self.subject_attn(self.subject_norm1(xn)))
            # [BTS, N, MC] -> [B, T, S, N, MC] -> [B, N, T, S, MC] -> [BN, TS, MC]
            res_subject = res_subject.reshape(B, T, S, N, MC).permute(0, 3, 1, 2, 4).reshape(BN, TS, MC)
            res_subject = self.subject_fc(res_subject)
            x_out = x_out + res_subject

            # ============= Spatial =============
            # [BN, TS, MC] -> [BNT, S, MC]
            xs = x_out.reshape(B, N, T, S, MC).reshape(B * T * N, S, MC)
            res_spatial = self.dropout(self.spatial_attn(self.spatial_norm1(xs)))
            # [BNT, S, MC] -> [B, N, T, S, MC] -> [BN, TS, MC]
            res_spatial = res_spatial.reshape(B, N, T, S, MC).reshape(BN, TS, MC)
            # res_spatial = self.spatial_fc(res_spatial)
            x_out = x_out + res_spatial

            # ============= MLP =============
            x = x_out + self.dropout(self.mlp(self.norm2(x_out)))

        else:     # without using TNS attention
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class CONV_ATTN(nn.Module):
    def __init__(
        self,
        num_blocks,
        embed_dim,
        num_heads=8,
        dropout=0,
        mlp_ratio=4.0,
        mlp_dropout=0,
        attn_type="",
        num_frame=8,
        num_sub=4,
    ):
        super(CONV_ATTN, self).__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(
                TransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attn_type=attn_type,
                    num_frame=num_frame,
                    num_sub=num_sub,
                )
            )

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x
