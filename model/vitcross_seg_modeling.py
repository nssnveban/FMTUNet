# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import model.vit_seg_configs as configs
from model.vit_seg_modeling_resnet_skip import FuseResNetV2


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 结合最大值和平均值池化生成空间响应图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 经七乘七卷积与Sigmoid激活产生空间权重图
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttentionModule(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 多层感知机或一乘一卷积生成通道权重向量
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化和最大池化压缩上下文
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # 经Sigmoid激活后逐通道加权突出任务相关特征维度
        return self.sigmoid(out)

class FusionConv(nn.Module):
    """融合卷积模块，实现多尺度特征融合"""
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        # 一乘一卷积压缩通道降低复杂度
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 三组不同尺度的深度可分离卷积并行提取多感受野空间上下文
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        # 空间注意力和通道注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x3):
        # 拼接三个输入特征
        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.down(x_fused)
        
        # 通道路径：逐通道加权突出任务相关特征维度
        x_fused_c = x_fused * self.channel_attention(x_fused)
        
        # 空间路径：多尺度卷积提取空间上下文
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        # 逐元素加权强调关键区域如物体边缘
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        # 双路特征融合：将聚焦位置的空间精炼特征与聚焦内容的通道精炼特征合并
        x_out = self.up(x_fused_s + x_fused_c)
        return x_out

class MSAA(nn.Module):
    """多尺度自适应注意力模块，通过融合相邻三层编码器特征整合细节与语义信息"""
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        """
        Args:
            x1: 当前层特征
            x2: 低层特征（传递语义信息）
            x4: 高层特征（传递边缘问题特征补充）
        """
        x_fused = self.fusion_conv(x1, x2, x3)
        return x_fused

class CMUNetSkipConnection(nn.Module):
    """CM-UNet风格的跳跃连接模块"""
    def __init__(self, skip_channels, decoder_channels):
        super(CMUNetSkipConnection, self).__init__()
        self.skip_channels = skip_channels
        self.decoder_channels = decoder_channels
        
        # 只为前三层创建MSAA模块（浅层融合）
        self.msaa_modules = nn.ModuleList()
        for i in range(min(3, len(skip_channels))):
            # 三个浅层特征拼接后的通道数
            # 使用 x1, x2, x3
            in_channels = skip_channels[0] + skip_channels[1] + skip_channels[2]
            
            # 修复：输出通道数改为原始跳跃连接的通道数
            self.msaa_modules.append(
                MSAA(in_channels, skip_channels[i])  # 改为skip_channels[i]
            )

    def forward(self, skip_features):
        """
        Args:
            skip_features: 来自编码器的跳跃连接特征列表 [x1, x2, x3, x4]
        Returns:
            processed_features: 经过MSAA处理的特征列表
        """
        processed_features = []
        
        # 确保至少有3个特征用于浅层融合
        if len(skip_features) < 3:
            # 如果特征数量不足，直接返回原特征
            return skip_features
        
        # 提取前三个浅层特征
        x1, x2, x3 = skip_features[0], skip_features[1], skip_features[2]
        
        # 对每一层进行处理
        for i in range(len(skip_features)):
            if i < 3:  # 前三层使用MSAA处理
                # 获取当前层的目标尺寸
                target_size = skip_features[i].shape[2:]
                
                # 将三个特征调整到相同尺寸
                x1_resized = F.interpolate(x1, size=target_size, mode='bilinear', align_corners=False) if x1.shape[2:] != target_size else x1
                x2_resized = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False) if x2.shape[2:] != target_size else x2
                x3_resized = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False) if x3.shape[2:] != target_size else x3
                
                # 使用MSAA模块处理三个特征
                processed_feature = self.msaa_modules[i](x1_resized, x2_resized, x3_resized)
                processed_features.append(processed_feature)
            else:
                # 其他层直接使用原特征
                processed_features.append(skip_features[i])
        
        return processed_features


class Attention(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.mode = mode
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        
        self.queryd = Linear(config.hidden_size, self.all_head_size)
        self.keyd = Linear(config.hidden_size, self.all_head_size)
        self.valued = Linear(config.hidden_size, self.all_head_size)
        self.outd = Linear(config.hidden_size, config.hidden_size)

        if self.mode == 'mba':
            self.w11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_statesx, hidden_statesy):
        mixed_query_layer = self.query(hidden_statesx)
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        mixed_queryd_layer = self.queryd(hidden_statesy)
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)

        ## Self Attention x: Qx, Kx, Vx
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sx = self.out(context_layer)
        attention_sx = self.proj_dropout(attention_sx)
        
        ## Self Attention y: Qy, Ky, Vy
        attention_scores = torch.matmul(queryd_layer, keyd_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, valued_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sy = self.outd(context_layer)
        attention_sy = self.proj_dropout(attention_sy)
        
        if self.mode == 'mba':
            ## Cross Attention x: Qx, Ky, Vy
            attention_scores = torch.matmul(query_layer, keyd_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, valued_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cx = self.out(context_layer)
            attention_cx = self.proj_dropout(attention_cx)
            
            ## Cross Attention y: Qy, Kx, Vx
            attention_scores = torch.matmul(queryd_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cy = self.outd(context_layer)
            attention_cy = self.proj_dropout(attention_cy)
        
            # Adaptative MBA
            attention_sx = self.w11 * attention_sx + self.w12 * attention_cx
            attention_sy = self.w21 * attention_sy + self.w22 * attention_cy
        
        return attention_sx, attention_sy, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DeformableCrossAttentionModule(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        inc: input channel number
        outc: output channel number
        """
        super(DeformableCrossAttentionModule, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc * kernel_size * kernel_size, outc, kernel_size=1, bias=bias)

        self.offset_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0.)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulation = modulation
        if modulation:
            self.modulator_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.modulator_conv.weight, 0.)
            if self.modulator_conv.bias is not None:
                nn.init.constant_(self.modulator_conv.bias, 0.)

    def forward(self, query, value):
        # query: (B, C, H, W)
        # value: (B, C, H, W)
        offset = self.offset_conv(query)
        modulator = torch.sigmoid(self.modulator_conv(query)) if self.modulation else None

        B, C, H, W = query.shape
        ks = self.kernel_size
        N = ks * ks
        dtype = offset.dtype
        device = offset.device

        # Get sampling grid for deformable conv
        value_padded = self.zero_padding(value)
        H_pad, W_pad = value_padded.shape[2:]

        # p_n: (1, 2N, 1, 1) - relative offsets for kernel points
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(ks - 1) // 2, (ks - 1) // 2 + 1, dtype=dtype, device=device),
            torch.arange(-(ks - 1) // 2, (ks - 1) // 2 + 1, dtype=dtype, device=device),
            indexing="ij")
        p_n = torch.cat([p_n_x.flatten(), p_n_y.flatten()], 0).view(1, 2 * N, 1, 1)

        # p_0: (1, 2N, H, W) - base grid for sampling points
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, H, self.stride, dtype=dtype, device=device),
            torch.arange(0, W, self.stride, dtype=dtype, device=device),
            indexing="ij")
        p_0_x = p_0_x.view(1, 1, H, W).repeat(1, N, 1, 1)
        p_0_y = p_0_y.view(1, 1, H, W).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1)

        # p: (B, 2N, H, W) - final sampling locations
        p = p_0.to(device) + p_n.to(device) + offset

        # Normalize p to [-1, 1] for grid_sample
        # p is (B, 2N, H, W), where 2N is [y1, y2, ..., x1, x2, ...]
        p_x = p[:, N:, :, :] # (B, N, H, W)
        p_y = p[:, :N, :, :] # (B, N, H, W)
        p_x_norm = 2.0 * p_x / (W_pad - 1) - 1.0
        p_y_norm = 2.0 * p_y / (H_pad - 1) - 1.0

        # grid: (B, H, W, N, 2) - stack and permute to match grid_sample format
        grid = torch.stack([p_x_norm, p_y_norm], dim=4) # (B, N, H, W, 2)
        grid = grid.permute(0, 2, 3, 1, 4) # (B, H, W, N, 2)

        # Reshape grid for grid_sample: (B, H, W, N, 2) -> (B, H, W*N, 2)
        grid = grid.reshape(B, H, W * N, 2)

        # Sample features from value tensor
        # sampled_feat: (B, C, H, W*N)
        sampled_feat = F.grid_sample(value_padded, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # Reshape back to 5D tensor: (B, C, H, W, N)
        sampled_feat = sampled_feat.view(B, C, H, W, N)

        if self.modulation:
            # modulator: (B, N, H, W) -> (B, 1, H, W, N)
            modulator = modulator.permute(0, 2, 3, 1).unsqueeze(1)
            # Apply modulation: (B, C, H, W, N) * (B, 1, H, W, N) -> (B, C, H, W, N)
            sampled_feat = sampled_feat * modulator

        # Reshape for final convolution
        # sampled_feat: (B, C, H, W, N) -> (B, C, N, H, W) -> (B, C*N, H, W)
        sampled_feat = sampled_feat.permute(0, 1, 4, 2, 3).contiguous().view(B, C * N, H, W)

        out = self.conv(sampled_feat)
        return out

class DeformableCrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(DeformableCrossAttentionBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attention_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn1 = Mlp(config)
        self.ffn_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn2 = Mlp(config)
        self.attn_x_enhances_y = DeformableCrossAttentionModule(inc=config.hidden_size, outc=config.hidden_size)
        self.attn_y_enhances_x = DeformableCrossAttentionModule(inc=config.hidden_size, outc=config.hidden_size)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        H = W = int(math.sqrt(N))

        # Normalize inputs
        x1_norm = self.attention_norm1(x1)
        x2_norm = self.attention_norm2(x2)

        # Reshape for Conv2d: (B, N, C) -> (B, C, H, W)
        x1_2d = x1_norm.transpose(1, 2).contiguous().view(B, C, H, W)
        x2_2d = x2_norm.transpose(1, 2).contiguous().view(B, C, H, W)

        # Branch 1: x1 enhances x2
        h2_enhanced_2d = self.attn_x_enhances_y(x1_2d, x2_2d)
        h2_enhanced = h2_enhanced_2d.view(B, C, N).transpose(1, 2)
        x2_out = x2 + h2_enhanced
        x2_out = x2_out + self.ffn2(self.ffn_norm2(x2_out))

        # Branch 2: x2 enhances x1
        h1_enhanced_2d = self.attn_y_enhances_x(x2_2d, x1_2d)
        h1_enhanced = h1_enhanced_2d.view(B, C, N).transpose(1, 2)
        x1_out = x1 + h1_enhanced
        x1_out = x1_out + self.ffn1(self.ffn_norm1(x1_out))

        return x1_out, x2_out, None

    def load_from(self, weights, n_block):
        pass

    def load_from(self, weights, n_block):
        pass



class HybridDeformableAttentionBlock(nn.Module):
    """混合可变形注意力模块：融合MBA的自适应加权与DeformableCrossAttentionModule的强大特征提取"""
    def __init__(self, config, vis):
        super(HybridDeformableAttentionBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.vis = vis
        
        # LayerNorm层
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(config.hidden_size, eps=1e-6)
        
        # FFN层
        self.ffn = Mlp(config)
        self.ffnd = Mlp(config)
        
        # 混合注意力机制
        self.hybrid_attn = HybridAttention(config, vis)

    def forward(self, x, y):
        # 保存残差
        hx = x
        hy = y
        
        # 注意力层
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y, weights = self.hybrid_attn(x, y)
        x = x + hx
        y = y + hy

        # FFN层
        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        
        return x, y, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # 加载自注意力部分的权重 (query, key, value, out)
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.hybrid_attn.query.weight.copy_(query_weight)
            self.hybrid_attn.key.weight.copy_(key_weight)
            self.hybrid_attn.value.weight.copy_(value_weight)
            self.hybrid_attn.out.weight.copy_(out_weight)
            self.hybrid_attn.query.bias.copy_(query_bias)
            self.hybrid_attn.key.bias.copy_(key_bias)
            self.hybrid_attn.value.bias.copy_(value_bias)
            self.hybrid_attn.out.bias.copy_(out_bias)
            
            self.hybrid_attn.queryd.weight.copy_(query_weight)
            self.hybrid_attn.keyd.weight.copy_(key_weight)
            self.hybrid_attn.valued.weight.copy_(value_weight)
            self.hybrid_attn.outd.weight.copy_(out_weight)
            self.hybrid_attn.queryd.bias.copy_(query_bias)
            self.hybrid_attn.keyd.bias.copy_(key_bias)
            self.hybrid_attn.valued.bias.copy_(value_bias)
            self.hybrid_attn.outd.bias.copy_(out_bias)

            # 加载FFN层的权重
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            
            self.ffnd.fc1.weight.copy_(mlp_weight_0)
            self.ffnd.fc2.weight.copy_(mlp_weight_1)
            self.ffnd.fc1.bias.copy_(mlp_bias_0)
            self.ffnd.fc2.bias.copy_(mlp_bias_1)

            # 加载LayerNorm层的权重
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.attention_normd.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_normd.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.ffn_normd.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_normd.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class HybridAttention(nn.Module):
    """混合注意力：结合自注意力、可变形跨注意力和MBA的自适应加权机制"""
    def __init__(self, config, vis):
        super(HybridAttention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 自注意力分支的Q、K、V投影
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.queryd = Linear(config.hidden_size, self.all_head_size)
        self.keyd = Linear(config.hidden_size, self.all_head_size)
        self.valued = Linear(config.hidden_size, self.all_head_size)

        # 可变形跨注意力分支
        self.deformable_ca = DeformableCrossAttentionModule(
            inc=config.hidden_size, 
            outc=config.hidden_size
        )

        # 输出投影
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.outd = Linear(config.hidden_size, config.hidden_size)
        
        # 动态加权门控参数
        self.gate_x = nn.Parameter(torch.zeros(1))
        self.gate_y = nn.Parameter(torch.zeros(1))

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_statesx, hidden_statesy):
        B, N, C = hidden_statesx.shape
        H = W = int(math.sqrt(N))
        
        # === 自注意力分支 ===
        # x的自注意力
        mixed_query_layer = self.query(hidden_statesx)
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sx = self.out(context_layer)
        attention_sx = self.proj_dropout(attention_sx)

        # y的自注意力
        mixed_queryd_layer = self.queryd(hidden_statesy)
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)

        attention_scores = torch.matmul(queryd_layer, keyd_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, valued_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sy = self.outd(context_layer)
        attention_sy = self.proj_dropout(attention_sy)

        # === 可变形跨注意力分支 ===
        # 转换为2D格式
        x_2d = hidden_statesx.transpose(1, 2).contiguous().view(B, C, H, W)
        y_2d = hidden_statesy.transpose(1, 2).contiguous().view(B, C, H, W)
        
        # x增强：query=x, value=y
        x_enhanced_2d = self.deformable_ca(x_2d, y_2d)
        attention_cx = x_enhanced_2d.view(B, C, N).transpose(1, 2)
        
        # y增强：query=y, value=x
        y_enhanced_2d = self.deformable_ca(y_2d, x_2d)
        attention_cy = y_enhanced_2d.view(B, C, N).transpose(1, 2)

        # === 动态加权融合 ===
        gate_x = torch.sigmoid(self.gate_x)
        gate_y = torch.sigmoid(self.gate_y)
        attention_x = gate_x * attention_sx + (1 - gate_x) * attention_cx
        attention_y = gate_y * attention_sy + (1 - gate_y) * attention_cy

        return attention_x, attention_y, weights


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = FuseResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.patch_embeddingsd = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x, y):
        y = y.unsqueeze(1)
        if self.hybrid:
            x, y, features = self.hybrid_model(x, y)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        y = self.patch_embeddingsd(y)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        y = y.flatten(2)
        y = y.transpose(-1, -2)
        
        embeddingsx = x + self.position_embeddings
        embeddingsx = self.dropout(embeddingsx)
        embeddingsy = y + self.position_embeddings
        embeddingsy = self.dropout(embeddingsy)
        return embeddingsx, embeddingsy, features


class Block(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.ffnd = Mlp(config)
        self.attn = Attention(config, vis, mode=mode)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y, weights = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            
            self.attn.queryd.weight.copy_(query_weight)
            self.attn.keyd.weight.copy_(key_weight)
            self.attn.valued.weight.copy_(value_weight)
            self.attn.outd.weight.copy_(out_weight)
            self.attn.queryd.bias.copy_(query_bias)
            self.attn.keyd.bias.copy_(key_bias)
            self.attn.valued.bias.copy_(value_bias)
            self.attn.outd.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            
            self.ffnd.fc1.weight.copy_(mlp_weight_0)
            self.ffnd.fc2.weight.copy_(mlp_weight_1)
            self.ffnd.fc1.bias.copy_(mlp_bias_0)
            self.ffnd.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.attention_normd.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_normd.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_normd.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_normd.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        
        # 原始架构：3SA + 6MBA + 3SA
        # 前3层：纯自注意力
        self.layer = nn.ModuleList()
        for i in range(3):
            layer = Block(config, vis, mode='sa')
            self.layer.append(copy.deepcopy(layer))
        
        # 中间6层：HybridDeformableAttentionBlock（融合MBA + DeformableCrossAttentionModule）
        for i in range(6):
            layer = HybridDeformableAttentionBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))
        
        # 后3层：纯自注意力
        for i in range(3):
            layer = Block(config, vis, mode='sa')
            self.layer.append(copy.deepcopy(layer))
        
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder_normd = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_statesx, hidden_statesy):
        attn_weights = []
        for layer_block in self.layer:
            hidden_statesx, hidden_statesy, weights = layer_block(hidden_statesx, hidden_statesy)
            if self.vis:
                attn_weights.append(weights)
        encodedx = self.encoder_norm(hidden_statesx)
        encodedy = self.encoder_normd(hidden_statesy)
        return encodedx, encodedy, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, dsm_ids):
        embeddingsx, embeddingsy, features = self.embeddings(input_ids, dsm_ids)
        encodedx, encodedy, attn_weights = self.encoder(embeddingsx, embeddingsy)  # (B, n_patch, hidden)
        return encodedx, encodedy, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)




class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)




class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,  # 恢复为原始的hidden_size
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        # 统一使用config中的decoder_channels
        decoder_channels = config.decoder_channels  # (256, 128, 64, 16)
        
        # 添加CM-UNet风格的跳跃连接预处理
        if config.n_skip != 0:
            skip_channels = config.skip_channels
            # 让MSAA输出与原始跳跃连接通道数匹配
            self.skip_connection_processor = CMUNetSkipConnection(skip_channels, decoder_channels)
        else:
            self.skip_connection_processor = None
        
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # 保持原有的 `skip_channels` 参数不变：
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels  # [512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None, use_basic_skip=False):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        if use_basic_skip:
            # 使用基础跳跃连接
            for i, decoder_block in enumerate(self.blocks):
                if features is not None and i < len(features):
                    # 基础跳跃连接：第1, 2, 3个joint mamba输出连接到第2, 3, 4个解码器
                    if i > 0:
                        skip = features[i-1]
                    else:
                        skip = None
                else:
                    skip = None
                x = decoder_block(x, skip=skip)
        else:
            # 使用MSAA跳跃链接
            if features is not None and self.skip_connection_processor is not None:
                features = self.skip_connection_processor(features)
            
            for i, decoder_block in enumerate(self.blocks):
                if features is not None:
                    skip = features[i] if (i < self.config.n_skip) else None
                else:
                    skip = None
                x = decoder_block(x, skip=skip)
        return x




class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, skip_deep_fusion=False, use_basic_skip=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.skip_deep_fusion = skip_deep_fusion  # 新增参数：是否跳过深层融合
        self.use_basic_skip = use_basic_skip
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x, y):
        if self.skip_deep_fusion:
            # 跳过深层融合，只执行浅层融合
            embeddingsx, embeddingsy, features = self.transformer.embeddings(x, y)
            # 直接使用浅层融合的输出，跳过Encoder（transformer堆叠部分）
            # 第四层浅层融合SqueezeAndExciteFusionAdd的结果作为UNet解码器的输入
            x = embeddingsx + embeddingsy
        else:
            # 正常执行完整的融合流程
            x, y, attn_weights, features = self.transformer(x, y)  # (B, n_patch, hidden)
            # 恢复为简单相加融合
            x = x + y  # (B, n_patch, hidden)
        
        x = self.decoder(x, features, self.use_basic_skip)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.patch_embeddingsd.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddingsd.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            self.transformer.encoder.encoder_normd.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_normd.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            processed_blocks = set()  # 避免重复处理同一个块
            for bname, weight in res_weight.items():
                if bname.startswith('Transformer/encoderblock_'):
                    # 从键名中提取块编号，例如 "Transformer/encoderblock_0/LayerNorm_0/bias" -> 0
                    parts = bname.split('/')
                    if len(parts) >= 2 and parts[1].startswith('encoderblock_'):
                        blocknum = int(parts[1].split('_')[1])
                        if blocknum < len(self.transformer.encoder.layer) and blocknum not in processed_blocks:
                            # 直接传递完整的权重字典和块编号
                            self.transformer.encoder.layer[blocknum].load_from(res_weight, blocknum)
                            processed_blocks.add(blocknum)

            if self.transformer.embeddings.hybrid:
                ws = res_weight["conv_root/kernel"]
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(ws, conv=True))
                ws = np.expand_dims(np.mean(ws, axis=2), axis=2)
                self.transformer.embeddings.hybrid_model.rootd.conv.weight.copy_(np2th(ws, conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                self.transformer.embeddings.hybrid_model.rootd.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.rootd.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                for bname, block in self.transformer.embeddings.hybrid_model.bodyd.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
            print('Load pretrained done.')

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

# 便捷函数：创建集成MSAA门控机制的模型
def create_msaa_model(config_name='R50-ViT-B_16', img_size=224, num_classes=6, skip_deep_fusion=False, use_basic_skip=False):
    """
    创建集成MSAA门控机制的TransUNet模型
    
    Args:
        config_name: 配置名称，默认'R50-ViT-B_16'
        img_size: 输入图像尺寸，默认224
        num_classes: 分类数量，默认6
        skip_deep_fusion: 是否跳过深层融合，默认False
        use_basic_skip: 是否使用基础跳跃连接，默认False
    
    Returns:
        VisionTransformer: 集成MSAA门控的模型
    """
    config = CONFIGS[config_name]
    config.n_classes = num_classes
    
    model = VisionTransformer(
        config, 
        img_size=img_size, 
        num_classes=num_classes, 
        vis=False,
        skip_deep_fusion=skip_deep_fusion,
        use_basic_skip=use_basic_skip
    )
    
    return model

# 便捷函数：创建禁用深层融合的模型
def create_shallow_fusion_model(config_name='R50-ViT-B_16', img_size=224, num_classes=6, use_basic_skip=False):
    """
    创建只使用浅层融合的TransUNet模型
    
    Args:
        config_name: 配置名称，默认'R50-ViT-B_16'
        img_size: 输入图像尺寸，默认224
        num_classes: 分类数量，默认6
        use_basic_skip: 是否使用基础跳跃连接，默认False
    
    Returns:
        VisionTransformer: 禁用深层融合的模型
    """
    config = CONFIGS[config_name]
    config['n_classes'] = num_classes
    
    model = VisionTransformer(
        config=config,
        img_size=img_size,
        num_classes=num_classes,
        skip_deep_fusion=True,  # 禁用深层融合
        use_basic_skip=use_basic_skip
    )
    
    return model


