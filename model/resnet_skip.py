import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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
# Joint Mamba
def scan_joint_mamba(feat_a, feat_b, step_size=2):
    B, C, H, W = feat_a.shape
    
    H_scan = H // step_size
    W_scan = W // step_size

    feat_h_concat = torch.cat([feat_a, feat_b], dim=3)  # 水平联接 [B, C, H, 2W]
    feat_v_concat = torch.cat([feat_a, feat_b], dim=2)  # 垂直联接 [B, C, 2H, W]

    h_sampled = feat_h_concat[:, :, :H_scan*step_size:step_size, :2*W_scan*step_size:step_size]  # [B, C, H_scan, 2*W_scan]
    h_forward = h_sampled.contiguous().view(B, C, -1)  # [B, C, H_scan * 2*W_scan]
    h_backward = h_forward.flip([2])  
    
    v_sampled = feat_v_concat[:, :, :2*H_scan*step_size:step_size, :W_scan*step_size:step_size]  # [B, C, 2*H_scan, W_scan]
    v_forward = v_sampled.transpose(2, 3).contiguous().view(B, C, -1)  # [B, C, 2*H_scan * W_scan]
    v_backward = v_forward.flip([2]) 
    
    h_scan_len = h_forward.shape[2]
    v_scan_len = v_forward.shape[2]

    h_theory_len = H_scan * (2 * W_scan)
    v_theory_len = (2 * H_scan) * W_scan

    max_scan_len = max(h_scan_len, v_scan_len)

    scans = feat_a.new_zeros((B, 4, C, max_scan_len))

    if h_scan_len != max_scan_len:
        h_forward = F.interpolate(h_forward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        h_backward = F.interpolate(h_backward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
    else:
        pass
        
    if v_scan_len != max_scan_len:
        v_forward = F.interpolate(v_forward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
        v_backward = F.interpolate(v_backward.transpose(1, 2), size=max_scan_len, mode='linear', align_corners=False).transpose(1, 2)
    else:
        pass
    
    scans[:, 0, :, :] = h_forward
    scans[:, 1, :, :] = h_backward
    scans[:, 2, :, :] = v_forward
    scans[:, 3, :, :] = v_backward

    scans = scans.transpose(2, 3)
    
    return scans, H, W

def merge_joint_mamba(scans, ori_h, ori_w, step_size=2):
    B, K, L, C = scans.shape
    
    scans = scans.transpose(2, 3)
    
    H_scan = ori_h // step_size
    W_scan = ori_w // step_size

    h_scan_len = H_scan * (2 * W_scan)
    v_scan_len = (2 * H_scan) * W_scan

    h_forward_scan = scans[:, 0, :, :]
    h_backward_scan = scans[:, 1, :, :]
    v_forward_scan = scans[:, 2, :, :]
    v_backward_scan = scans[:, 3, :, :]

    if L != h_scan_len:
        h_forward_scan = F.interpolate(h_forward_scan.unsqueeze(1), size=h_scan_len, mode='linear', align_corners=False).squeeze(1)
        h_backward_scan = F.interpolate(h_backward_scan.unsqueeze(1), size=h_scan_len, mode='linear', align_corners=False).squeeze(1)
    else:
        pass
    
    if L != v_scan_len:
        v_forward_scan = F.interpolate(v_forward_scan.unsqueeze(1), size=v_scan_len, mode='linear', align_corners=False).squeeze(1)
        v_backward_scan = F.interpolate(v_backward_scan.unsqueeze(1), size=v_scan_len, mode='linear', align_corners=False).squeeze(1)
    else:
        pass
    
    h_backward_scan = h_backward_scan.flip([2])
    v_backward_scan = v_backward_scan.flip([2])
    
    try:
        feat_h_f = h_forward_scan.reshape(B, C, H_scan, 2 * W_scan)
        feat_h_b = h_backward_scan.reshape(B, C, H_scan, 2 * W_scan)
    except RuntimeError as e:
        target_h_len = H_scan * 2 * W_scan
        h_forward_scan = F.interpolate(h_forward_scan.unsqueeze(1), size=target_h_len, mode='linear', align_corners=False).squeeze(1)
        h_backward_scan = F.interpolate(h_backward_scan.unsqueeze(1), size=target_h_len, mode='linear', align_corners=False).squeeze(1)
        feat_h_f = h_forward_scan.reshape(B, C, H_scan, 2 * W_scan)
        feat_h_b = h_backward_scan.reshape(B, C, H_scan, 2 * W_scan)
    
    try:
        feat_v_f = v_forward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        feat_v_b = v_backward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
    except RuntimeError as e:
        target_v_len = W_scan * 2 * H_scan
        v_forward_scan = F.interpolate(v_forward_scan.unsqueeze(1), size=target_v_len, mode='linear', align_corners=False).squeeze(1)
        v_backward_scan = F.interpolate(v_backward_scan.unsqueeze(1), size=target_v_len, mode='linear', align_corners=False).squeeze(1)
        feat_v_f = v_forward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
        feat_v_b = v_backward_scan.reshape(B, C, W_scan, 2 * H_scan).transpose(2, 3)
    
    feat_h_merged = feat_h_f + feat_h_b
    feat_v_merged = feat_v_f + feat_v_b
    
    feat_a_h, feat_b_h = torch.chunk(feat_h_merged, 2, dim=3)
    feat_a_v, feat_b_v = torch.chunk(feat_v_merged, 2, dim=2)
    
    feat_a_final = feat_a_h + feat_a_v
    feat_b_final = feat_b_h + feat_b_v
    
    if feat_a_final.shape[2] != ori_h or feat_a_final.shape[3] != ori_w:
        feat_a_final = F.interpolate(feat_a_final, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        feat_b_final = F.interpolate(feat_b_final, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
    
    return feat_a_final, feat_b_final


class JointMambaFusion(nn.Module):
    def __init__(self, channels_in, depth=1, step_size=2):
        super(JointMambaFusion, self).__init__()
        self.channels_in = channels_in
        self.depth = depth
        self.step_size = step_size
        
        self.mamba_layers = nn.ModuleList()
        for i in range(4):  
            self.mamba_layers.append(Mamba(d_model=channels_in))
        
        self.interpolate_align = True
        
    def forward(self, feat_a, feat_b):
        device = feat_a.device
        
        if not next(self.mamba_layers.parameters()).device == device:
            self.mamba_layers = self.mamba_layers.to(device)
        
        original_add = feat_a + feat_b
        
        scans, ori_h, ori_w = scan_joint_mamba(feat_a, feat_b, self.step_size)
        
        scans = scans.to(device)
        
        enhanced_scans = []
        for i in range(4):
            scan_i = scans[:, i]  # [B, L, C]
            
            scan_i = scan_i.to(device)
            
            if i in [1, 3] and self.step_size == 2:  
                L = scan_i.shape[1]
                scan_i_sampled = scan_i[:, ::2, :]  
                
                enhanced_scan = self.mamba_layers[i](scan_i_sampled)

                if self.interpolate_align:
                    enhanced_scan = F.interpolate(
                        enhanced_scan.transpose(1, 2), 
                        size=L, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    enhanced_scan = enhanced_scan.repeat_interleave(2, dim=1)[:, :L, :]
            else:
                enhanced_scan = self.mamba_layers[i](scan_i)
            
            enhanced_scans.append(enhanced_scan)
        
        enhanced_scans = torch.stack(enhanced_scans, dim=1)  # [B, 4, L, C]

        feat_a_enhanced, feat_b_enhanced = merge_joint_mamba(enhanced_scans, ori_h, ori_w, self.step_size)

        if feat_a_enhanced.shape != feat_a.shape:
            feat_a_enhanced = F.interpolate(feat_a_enhanced, size=(feat_a.shape[2], feat_a.shape[3]), mode='bilinear', align_corners=False)
        if feat_b_enhanced.shape != feat_b.shape:
            feat_b_enhanced = F.interpolate(feat_b_enhanced, size=(feat_b.shape[2], feat_b.shape[3]), mode='bilinear', align_corners=False)

        joint_result = feat_a_enhanced + feat_b_enhanced

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

        self.se_layer0 = SqueezeAndExciteFusionAdd(width, activation=self.activation)
        self.se_layer1 = SqueezeAndExciteFusionAdd(width*4, activation=self.activation)
        self.se_layer2 = SqueezeAndExciteFusionAdd(width*8, activation=self.activation)
        self.se_layer3 = SqueezeAndExciteFusionAdd(width*16, activation=self.activation)

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
                    x_se = self.se_layer1.se_rgb(x)
                    y_se = self.se_layer1.se_depth(y)
                    x = self.joint_mamba1(x_se, y_se)
                if i == 1:
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
            x_se = self.se_layer3(x, y)
            y_se = self.se_layer3(y, x)
            x = self.joint_mamba3(x_se, y_se)
        return x, y, features[::-1]

