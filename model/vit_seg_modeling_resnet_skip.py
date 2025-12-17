import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 直接导入mamba_ssm
from mamba_ssm import Mamba

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out
# Joint Mamba扫描函数
def scan_joint_mamba(feat_a, feat_b, step_size=2):
    """联合扫描两个特征图"""
    B, C, H, W = feat_a.shape
    
    # 详细调试输出
    # print(f"\n=== scan_joint_mamba 调试信息 ===")
    # print(f"输入特征图形状: feat_a={feat_a.shape}, feat_b={feat_b.shape}")
    # print(f"输入参数: step_size={step_size}")
    # print(f"原始尺寸: B={B}, C={C}, H={H}, W={W}")
    
    # 计算扫描后的尺寸
    H_scan = H // step_size
    W_scan = W // step_size
    # print(f"扫描后尺寸: H_scan={H_scan}, W_scan={W_scan}")
    
    # 水平联接和垂直联接
    feat_h_concat = torch.cat([feat_a, feat_b], dim=3)  # 水平联接 [B, C, H, 2W]
    feat_v_concat = torch.cat([feat_a, feat_b], dim=2)  # 垂直联接 [B, C, 2H, W]
    # print(f"联接后形状: feat_h_concat={feat_h_concat.shape}, feat_v_concat={feat_v_concat.shape}")
    
    # 使用更精确的扫描方式 - 确保精确匹配理论长度
    # print(f"\n=== 扫描过程 ===")
    # 水平联接的扫描 - 精确裁剪到理论尺寸
    h_sampled = feat_h_concat[:, :, :H_scan*step_size:step_size, :2*W_scan*step_size:step_size]  # [B, C, H_scan, 2*W_scan]
    # print(f"水平采样后形状: h_sampled={h_sampled.shape}")
    h_forward = h_sampled.contiguous().view(B, C, -1)  # [B, C, H_scan * 2*W_scan]
    # print(f"水平正向扫描形状: h_forward={h_forward.shape}")
    h_backward = h_forward.flip([2])  # 反向扫描
    # print(f"水平反向扫描形状: h_backward={h_backward.shape}")
    
    # 垂直联接的扫描 - 精确裁剪到理论尺寸
    v_sampled = feat_v_concat[:, :, :2*H_scan*step_size:step_size, :W_scan*step_size:step_size]  # [B, C, 2*H_scan, W_scan]
    # print(f"垂直采样后形状: v_sampled={v_sampled.shape}")
    v_forward = v_sampled.transpose(2, 3).contiguous().view(B, C, -1)  # [B, C, 2*H_scan * W_scan]
    # print(f"垂直正向扫描形状: v_forward={v_forward.shape}")
    v_backward = v_forward.flip([2])  # 反向扫描
    # print(f"垂直反向扫描形状: v_backward={v_backward.shape}")
    
    # 获取实际扫描长度
    h_scan_len = h_forward.shape[2]
    v_scan_len = v_forward.shape[2]
    # print(f"\n实际扫描长度: h_scan_len={h_scan_len}, v_scan_len={v_scan_len}")
    
    # 计算理论扫描长度
    h_theory_len = H_scan * (2 * W_scan)
    v_theory_len = (2 * H_scan) * W_scan
    # print(f"理论扫描长度: h_theory_len={h_theory_len}, v_theory_len={v_theory_len}")
    # print(f"长度差异: h_diff={h_scan_len-h_theory_len}, v_diff={v_scan_len-v_theory_len}")
    
    # 使用最大长度来统一尺寸
    max_scan_len = max(h_scan_len, v_scan_len)
    # print(f"\n=== 统一尺寸过程 ===")
    # print(f"最大扫描长度: max_scan_len={max_scan_len}")
    
    # 初始化扫描序列 [B, 4, C, max_scan_len]
    scans = feat_a.new_zeros((B, 4, C, max_scan_len))
    # print(f"初始化scans形状: {scans.shape}")
    
    # 将扫描结果填入统一的张量中，使用插值对齐长度
    if h_scan_len != max_scan_len:
        # print(f"需要对水平扫描进行插值对齐: {h_scan_len} -> {max_scan_len}")
        # 对于[B, C, L]的张量，需要转置为[B, L, C]进行插值，然后再转回来
        h_forward = F.interpolate(h_forward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        h_backward = F.interpolate(h_backward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        # print(f"插值后水平扫描形状: h_forward={h_forward.shape}, h_backward={h_backward.shape}")
    else:
        pass
        # print(f"水平扫描长度已匹配，无需插值")
        
    if v_scan_len != max_scan_len:
        # print(f"需要对垂直扫描进行插值对齐: {v_scan_len} -> {max_scan_len}")
        v_forward = F.interpolate(v_forward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        v_backward = F.interpolate(v_backward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        # print(f"插值后垂直扫描形状: v_forward={v_forward.shape}, v_backward={v_backward.shape}")
    else:
        pass
        # print(f"垂直扫描长度已匹配，无需插值")
    
    # print(f"\n=== 填充扫描结果 ===")
    scans[:, 0, :, :] = h_forward
    scans[:, 1, :, :] = h_backward
    scans[:, 2, :, :] = v_forward
    scans[:, 3, :, :] = v_backward
    # print(f"填充后scans形状: {scans.shape}")
    
    # 转换为 [B, 4, L, C] 格式供Mamba处理
    scans = scans.transpose(2, 3)
    # print(f"最终输出scans形状: {scans.shape}")
    # print(f"返回原始尺寸: H={H}, W={W}")
    # print(f"=== scan_joint_mamba 完成 ===\n")
    
    return scans, H, W

def merge_joint_mamba(scans, ori_h, ori_w, step_size=2):
    """合并扫描结果"""
    B, K, L, C = scans.shape
    
    # 详细调试输出
    # print(f"\n=== merge_joint_mamba 调试信息 ===")
    # print(f"输入参数: ori_h={ori_h}, ori_w={ori_w}, step_size={step_size}")
    # print(f"输入scans形状: {scans.shape} (B={B}, K={K}, L={L}, C={C})")
    
    # 转换回 [B, 4, C, L]
    scans = scans.transpose(2, 3)
    # print(f"转换后scans形状: {scans.shape}")
    
    H_scan = ori_h // step_size
    W_scan = ori_w // step_size
    # print(f"计算的扫描尺寸: H_scan={H_scan}, W_scan={W_scan}")
    
    # 计算理论扫描长度
    h_scan_len = H_scan * (2 * W_scan)
    v_scan_len = (2 * H_scan) * W_scan
    # print(f"理论扫描长度: h_scan_len={h_scan_len}, v_scan_len={v_scan_len}")
    # print(f"实际扫描长度: L={L}")
    # print(f"长度差异: h_diff={L-h_scan_len}, v_diff={L-v_scan_len}")
    
    # 从统一长度的扫描结果中恢复到原始长度
    h_forward_scan = scans[:, 0, :, :]
    h_backward_scan = scans[:, 1, :, :]
    v_forward_scan = scans[:, 2, :, :]
    v_backward_scan = scans[:, 3, :, :]
    
    # print(f"\n提取的扫描结果形状:")
    # print(f"h_forward_scan: {h_forward_scan.shape}")
    # print(f"h_backward_scan: {h_backward_scan.shape}")
    # print(f"v_forward_scan: {v_forward_scan.shape}")
    # print(f"v_backward_scan: {v_backward_scan.shape}")
    
    # 使用插值将所有扫描调整到正确的理论长度
    if L != h_scan_len:
        # print(f"\n需要对水平扫描进行插值: {L} -> {h_scan_len}")
        h_forward_scan = F.interpolate(h_forward_scan.unsqueeze(1), size=h_scan_len, mode='linear', align_corners=False).squeeze(1)
        h_backward_scan = F.interpolate(h_backward_scan.unsqueeze(1), size=h_scan_len, mode='linear', align_corners=False).squeeze(1)
        # print(f"插值后水平扫描形状: {h_forward_scan.shape}")
    else:
        pass
        # print(f"\n水平扫描长度匹配，无需插值")
    
    if L != v_scan_len:
        # print(f"\n需要对垂直扫描进行插值: {L} -> {v_scan_len}")
        v_forward_scan = F.interpolate(v_forward_scan.unsqueeze(1), size=v_scan_len, mode='linear', align_corners=False).squeeze(1)
        v_backward_scan = F.interpolate(v_backward_scan.unsqueeze(1), size=v_scan_len, mode='linear', align_corners=False).squeeze(1)
        # print(f"插值后垂直扫描形状: {v_forward_scan.shape}")
    else:
        pass
        # print(f"\n垂直扫描长度匹配，无需插值")
    
    # 反转反向扫描
    h_backward_scan = h_backward_scan.flip([2])
    v_backward_scan = v_backward_scan.flip([2])
    # print(f"\n反转后扫描形状:")
    # print(f"h_backward_scan: {h_backward_scan.shape}")
    # print(f"v_backward_scan: {v_backward_scan.shape}")
    
    # 直接进行reshape，现在长度应该是精确匹配的
    # print(f"\n=== 水平扫描reshape操作 ===")
    # print(f"目标reshape形状: ({B}, {C}, {H_scan}, {2 * W_scan})")
    # print(f"需要的元素总数: {B * C * H_scan * 2 * W_scan}")
    # print(f"h_forward_scan实际形状: {h_forward_scan.shape}")
    # print(f"h_forward_scan实际元素数: {h_forward_scan.numel()}")
    
    try:
        feat_h_f = h_forward_scan.reshape(B, C, H_scan, 2 * W_scan)
        feat_h_b = h_backward_scan.reshape(B, C, H_scan, 2 * W_scan)
        # print(f"水平扫描reshape成功!")
    except RuntimeError as e:
        # print(f"\n!!! 水平扫描reshape失败 !!!")
        # print(f"错误信息: {e}")
        # print(f"h_forward_scan.shape: {h_forward_scan.shape}")
        # print(f"期望reshape: ({B}, {C}, {H_scan}, {2 * W_scan})")
        # print(f"期望元素数: {H_scan * 2 * W_scan}")
        # print(f"实际元素数: {h_forward_scan.shape[2]}")
        # print(f"元素数差异: {h_forward_scan.shape[2] - H_scan * 2 * W_scan}")
        
        # 强制调整到正确尺寸
        target_h_len = H_scan * 2 * W_scan
        # print(f"强制插值调整到长度: {target_h_len}")
        h_forward_scan = F.interpolate(h_forward_scan.unsqueeze(1), size=target_h_len, mode='linear', align_corners=False).squeeze(1)
        h_backward_scan = F.interpolate(h_backward_scan.unsqueeze(1), size=target_h_len, mode='linear', align_corners=False).squeeze(1)
        # print(f"强制调整后形状: {h_forward_scan.shape}")
        feat_h_f = h_forward_scan.reshape(B, C, H_scan, 2 * W_scan)
        feat_h_b = h_backward_scan.reshape(B, C, H_scan, 2 * W_scan)
        # print(f"强制调整后reshape成功!")
    
    # print(f"\n=== 垂直扫描reshape操作 ===")
    # print(f"目标reshape形状: ({B}, {C}, {W_scan}, {2 * H_scan})")
    # print(f"需要的元素总数: {B * C * W_scan * 2 * H_scan}")
    # print(f"v_forward_scan实际形状: {v_forward_scan.shape}")
    # print(f"v_forward_scan实际元素数: {v_forward_scan.numel()}")
    
    try:
        feat_v_f = v_forward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        feat_v_b = v_backward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        # print(f"垂直扫描reshape成功!")
    except RuntimeError as e:
        # print(f"\n!!! 垂直扫描reshape失败 !!!")
        # print(f"错误信息: {e}")
        # print(f"v_forward_scan.shape: {v_forward_scan.shape}")
        # print(f"期望reshape: ({B}, {C}, {W_scan}, {2 * H_scan})")
        # print(f"期望元素数: {W_scan * 2 * H_scan}")
        # print(f"实际元素数: {v_forward_scan.shape[2]}")
        # print(f"元素数差异: {v_forward_scan.shape[2] - W_scan * 2 * H_scan}")
        
        # 强制调整到正确尺寸
        target_v_len = W_scan * 2 * H_scan
        # print(f"强制插值调整到长度: {target_v_len}")
        v_forward_scan = F.interpolate(v_forward_scan.unsqueeze(1), size=target_v_len, mode='linear', align_corners=False).squeeze(1)
        v_backward_scan = F.interpolate(v_backward_scan.unsqueeze(1), size=target_v_len, mode='linear', align_corners=False).squeeze(1)
        # print(f"强制调整后形状: {v_forward_scan.shape}")
        feat_v_f = v_forward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        feat_v_b = v_backward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        # print(f"强制调整后reshape成功!")
    
    # 融合正向和反向扫描结果
    feat_h_merged = feat_h_f + feat_h_b
    feat_v_merged = feat_v_f + feat_v_b
    
    # 分离特征图
    feat_a_h, feat_b_h = torch.chunk(feat_h_merged, 2, dim=3)
    feat_a_v, feat_b_v = torch.chunk(feat_v_merged, 2, dim=2)
    
    # 最终融合
    feat_a_final = feat_a_h + feat_a_v
    feat_b_final = feat_b_h + feat_b_v
    
    # 确保输出尺寸与原始输入一致
    if feat_a_final.shape[2] != ori_h or feat_a_final.shape[3] != ori_w:
        feat_a_final = F.interpolate(feat_a_final, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        feat_b_final = F.interpolate(feat_b_final, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
    
    return feat_a_final, feat_b_final



# Joint Mamba融合模块
class JointMambaFusion(nn.Module):
    def __init__(self, channels_in, depth=1, step_size=2):
        super(JointMambaFusion, self).__init__()
        self.channels_in = channels_in
        self.depth = depth
        self.step_size = step_size
        
        # 创建Mamba层
        self.mamba_layers = nn.ModuleList()
        for i in range(4):  # 4个扫描方向
            self.mamba_layers.append(Mamba(d_model=channels_in))
        
        # 插值层用于对齐反向扫描
        self.interpolate_align = True
        
    def forward(self, feat_a, feat_b):
        # 检查输入设备并确保模型在正确设备上
        device = feat_a.device
        
        # 确保Mamba层在正确的设备上
        if not next(self.mamba_layers.parameters()).device == device:
            self.mamba_layers = self.mamba_layers.to(device)
        
        # 保存原始相加结果
        original_add = feat_a + feat_b
        
        # 联合扫描
        scans, ori_h, ori_w = scan_joint_mamba(feat_a, feat_b, self.step_size)
        
        # 确保扫描结果在正确设备上
        scans = scans.to(device)
        
        # 通过Mamba层处理每个扫描方向
        enhanced_scans = []
        for i in range(4):
            scan_i = scans[:, i]  # [B, L, C]
            
            # 确保scan_i在正确设备上
            scan_i = scan_i.to(device)
            
            # 对反向扫描进行步长处理和插值对齐
            if i in [1, 3] and self.step_size == 2:  # 反向扫描
                # 步长为2的采样
                L = scan_i.shape[1]
                scan_i_sampled = scan_i[:, ::2, :]  # 步长为2采样
                
                # 通过Mamba处理
                enhanced_scan = self.mamba_layers[i](scan_i_sampled)
                
                # 插值对齐到原始长度
                if self.interpolate_align:
                    # enhanced_scan形状: [B, L_sampled, C]
                    # 需要转换为[B, C, L_sampled]进行插值，然后转回[B, L, C]
                    enhanced_scan = F.interpolate(
                        enhanced_scan.transpose(1, 2), 
                        size=L, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    # 简单重复填充
                    enhanced_scan = enhanced_scan.repeat_interleave(2, dim=1)[:, :L, :]
            else:
                # 正向扫描直接处理
                enhanced_scan = self.mamba_layers[i](scan_i)
            
            enhanced_scans.append(enhanced_scan)
        
        # 重新组合扫描结果
        enhanced_scans = torch.stack(enhanced_scans, dim=1)  # [B, 4, L, C]
        
        # 合并扫描结果
        feat_a_enhanced, feat_b_enhanced = merge_joint_mamba(enhanced_scans, ori_h, ori_w, self.step_size)
        
        # 确保增强特征与原始特征尺寸一致
        if feat_a_enhanced.shape != feat_a.shape:
            feat_a_enhanced = F.interpolate(feat_a_enhanced, size=(feat_a.shape[2], feat_a.shape[3]), mode='bilinear', align_corners=False)
        if feat_b_enhanced.shape != feat_b.shape:
            feat_b_enhanced = F.interpolate(feat_b_enhanced, size=(feat_b.shape[2], feat_b.shape[3]), mode='bilinear', align_corners=False)
        
        # 融合增强特征
        joint_result = feat_a_enhanced + feat_b_enhanced
        
        # 与原始相加结果结合（残差连接）
        final_result = joint_result + original_add
        
        return final_result
    
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

class FuseResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.activation=nn.ReLU(inplace=True)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        
        self.rootd = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        # 浅层融合层配置：SE + Joint Mamba
        self.se_layer0 = SqueezeAndExciteFusionAdd(width, activation=self.activation)
        self.se_layer1 = SqueezeAndExciteFusionAdd(width*4, activation=self.activation)
        self.se_layer2 = SqueezeAndExciteFusionAdd(width*8, activation=self.activation)
        self.se_layer3 = SqueezeAndExciteFusionAdd(width*16, activation=self.activation)
        
        # Joint Mamba融合层
        self.joint_mamba0 = JointMambaFusion(width, depth=1, step_size=2)
        self.joint_mamba1 = JointMambaFusion(width*4, depth=1, step_size=2)
        self.joint_mamba2 = JointMambaFusion(width*8, depth=1, step_size=2)
        self.joint_mamba3 = JointMambaFusion(width*16, depth=1, step_size=2)
        
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))
    
        self.bodyd = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x, y):
        SE = True
        # SE = False
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x) #64*128
        y = self.rootd(y) #64*128
        if SE:
            # 先经过SE处理两个分支
            x_se = self.se_layer0(x, y)
            y_se = self.se_layer0(y, x)
        
            x = self.joint_mamba0(x_se, y_se)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        for i in range(len(self.body)-1):
            x = self.body[i](x) #256*63, 512*32
            y = self.bodyd[i](y) #256*63, 512*32
            if SE:
                if i == 0:
                    # SE层处理后使用Joint Mamba融合
                    x_se = self.se_layer1.se_rgb(x)
                    y_se = self.se_layer1.se_depth(y)
                    x = self.joint_mamba1(x_se, y_se)
                if i == 1:
                    # SE层处理后使用Joint Mamba融合
                    x_se = self.se_layer2.se_rgb(x)
                    y_se = self.se_layer2.se_depth(y)
                    x = self.joint_mamba2(x_se, y_se)
                
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x) # 1024*16
        y = self.bodyd[-1](y) # 1024*16
        if SE:
            # 最后一层SE处理后使用Joint Mamba融合
            x_se = self.se_layer3(x, y)
            y_se = self.se_layer3(y, x)
            x = self.joint_mamba3(x_se, y_se)
        return x, y, features[::-1]
