import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import at_cuda as at_cuda
import math
import time
import sys
import torchvision
from utils import *


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def padding(input, pad_size):
    B, H, W, C = input.shape
    t_input = torch.zeros((B, H + 2*pad_size, W + 2*pad_size, C), dtype=torch.float32, device=input.device)
    t_input[:, pad_size:pad_size+H, pad_size:pad_size+W, :] = input.clone()
    return t_input.contiguous()

class ChannelAttention(Function):
    kernel_size = 5
    pad_size = 2
    stride = 1

    def __init__(self, kernel_size = 3, pad_size = 1, stride = 1):
        super(ChannelAttention, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride = stride

    @staticmethod
    def forward(ctx, input1, input2, kernel_size = 5, pad_size = 2, stride = 1):
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.pad_size = pad_size
        ctx.stride = stride
        B, H1, W1, C1 = input1.shape
        B, C2, H2, W2 = input2.shape

        input2 = input2.permute(0, 2, 3, 1)
        output = torch.zeros_like(input2).contiguous()

        t_input2 = padding(input2, pad_size)
        at_cuda.channel_forward(input1.contiguous(), t_input2.contiguous(), output, kernel_size, pad_size, stride)
        
        return output.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size = ctx.kernel_size
        pad_size = ctx.pad_size
        stride = ctx.stride
        input1, input2 = ctx.saved_tensors
        B, H1, W1, C1 = input1.shape
        B, C2, H2, W2 = input2.shape
        input2 = input2.permute(0, 2, 3, 1)

        grad_input = grad_output.permute(0, 2, 3, 1)
        grad_output1 = torch.zeros_like(input1).contiguous()
        grad_output2 = torch.zeros_like(input2).contiguous()

        t_input2 = padding(input2, pad_size)

        at_cuda.channel_backward(input1.contiguous(), t_input2.contiguous(), grad_input.contiguous(), grad_output1.contiguous(), grad_output2.contiguous(), 
            kernel_size, pad_size, stride)
        return grad_output1, grad_output2.permute(0, 3, 1, 2), None, None, None


# class Correlation(Function):
#
#     def __init__(self, kernel_size=3, pad_size=1, stride=1):
#         super(Correlation, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.stride = 1
#
#     @staticmethod
#     def forward(ctx, input1, input2, kernel_size = 5, pad_size = 2, stride = 1):
#         ctx.save_for_backward(input1, input2)
#         ctx.kernel_size = kernel_size
#         ctx.pad_size = pad_size
#         ctx.stride = stride
#
#         B, C, H, W = input1.shape
#         out_H = (H - kernel_size + 2 * pad_size) // stride + 1
#         out_W = (W - kernel_size + 2 * pad_size) // stride + 1
#         out_C = kernel_size * kernel_size
#
#         output = torch.zeros((B, out_H, out_W, out_C), dtype=torch.float32, device=input1.device)
#         input1 = input1.permute(0, 2, 3, 1).contiguous()
#         input2 = input2.permute(0, 2, 3, 1).contiguous()
#
#         t_input1 = padding(input1, pad_size)
#         t_input2 = padding(input2, pad_size)
#
#         at_cuda.forward(t_input1.contiguous(), t_input2.contiguous(), output, kernel_size, pad_size, stride)
#
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input1, input2 = ctx.saved_tensors
#         kernel_size = ctx.kernel_size
#         pad_size = ctx.pad_size
#         stride = ctx.stride
#
#         B, C, H, W = input1.shape
#         out_H = (H - kernel_size + 2 * pad_size) // stride + 1
#         out_W = (W - kernel_size + 2 * pad_size) // stride + 1
#         out_C = kernel_size * kernel_size
#
#         input1 = input1.permute(0, 2, 3, 1).contiguous()
#         input2 = input2.permute(0, 2, 3, 1).contiguous()
#
#         t_input1 = padding(input1, pad_size)
#         t_input2 = padding(input2, pad_size)
#
#         g1 = torch.zeros_like(t_input1).contiguous()
#         g2 = torch.zeros_like(input1).contiguous()
#         grad_input = grad_output
#         grad_input_padding = padding(grad_input, pad_size)
#
#         at_cuda.backward(t_input1.contiguous(), t_input2.contiguous(), grad_input.contiguous(), grad_input_padding.contiguous(),
#             g1, g2, kernel_size, pad_size, stride)
#
#         return g1[:, pad_size:-pad_size, pad_size:-pad_size, :].permute(0, 3, 1, 2), g2.permute(0, 3, 1, 2), None, None, None

class CorrBlock:
    def __init__(self):
        super(CorrBlock, self).__init__()

    def __call__(self, input1, input2, kernel_size=5, pad_size=2, stride=1):
        # input
        B, C, H, W = input1.shape
        out_H = (H - kernel_size + 2 * pad_size) // stride + 1
        out_W = (W - kernel_size + 2 * pad_size) // stride + 1
        out_C = kernel_size * kernel_size

        # output = torch.zeros((B, out_H, out_W, out_C), dtype=torch.float32, device=input1.device)
        # input1 = input1.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)
        # input2 = input2.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)

        # t_input1 = padding(input1, pad_size) # (B, H + 2*pad_size, W + 2*pad_size, C)
        # t_input2 = padding(input2, pad_size) # (B, H + 2*pad_size, W + 2*pad_size, C)

        # calculate corr_pyramid
        corr = CorrBlock.corr(input1, input2)
        batch, h1, w1, dim, h2, w2 = corr.shape # (B, H, W, 1, H, W)
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        # make coordinate
        coords = coords_grid(batch, h1, w1).to(input1.device)
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape # (B, H, W, 2)

        dx = torch.linspace(-pad_size, pad_size, 2 * pad_size + 1)
        dy = torch.linspace(-pad_size, pad_size, 2 * pad_size + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * pad_size + 1, 2 * pad_size + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        output = corr.view(batch, h1, w1, -1) # (B, H, W, 2r+1, 2r+1)

        return output

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.relu(torch.matmul(fmap1.transpose(1, 2), fmap2))
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        return corr

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, norm=False):
        super().__init__()
        self.w_1 = nn.Conv2d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv2d(d_hid, d_in, 1)  # position-wise
        self.ln = LayerNorm2d(d_in)
        self.norm = norm
        
    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        if self.norm:
            x = self.ln(x)
        x += residual
        return x


class single_head_local_attention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.corr_fn = CorrBlock()

    def forward(self, q, k, v, kernel, pad, show=False):
        corr = self.corr_fn(q, k, kernel, pad)
        corr = torch.softmax(corr / self.temperature, dim=3)
        result = ChannelAttention.apply(corr, v, kernel, pad)
        if show:
            return result, corr
        else:
            return result


def single_head_global_attention(q, k, v, show=False):
    # q, k, v: [B, H*W, C]
    [B, C, H, W] = q.shape
    q = q.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    k = k.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    v = v.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, H*W, H*W]
    attn = torch.softmax(scores, dim=2)  # [B, H*W, H*W]
    out = torch.matmul(attn, v)  # [B, H*W, C]
    out = out.permute(0, 2, 1)
    out = out.reshape([B, C, H, W])

    if show:
        return out, attn
    else:
        return out


class Multi_head_focus_attention(nn.Module):
    """
        multi_head_focus_attention
    """
    def __init__(self, n_head, d_model, d_k, d_v, norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Conv2d(d_model, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(d_model, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(d_model, n_head * d_v, 1, bias=False)
        self.fc = nn.Conv2d(n_head * d_v, d_model, 1, bias=False)

        self.ln = LayerNorm2d(d_model)
        self.norm = norm
        
    def forward(self, q, k, v, kernel, pad, show=False):
          
        residual = q

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        
        if kernel == 0:
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            res_q = []
            att_list = []
            for i in range(n_head):
                q_t, k_t, v_t = q[:, i*d_k:(i+1)*d_k], k[:, i*d_k:(i+1)*d_k], v[:, i*d_v:(i+1)*d_v]
                if show:
                    q_t, att = single_head_global_attention(q_t, k_t, v_t, show)
                    att_list.append(att)
                else:
                    q_t = single_head_global_attention(q_t, k_t, v_t)
                res_q.append(q_t)
            q = torch.cat(res_q, dim=1)
        else:
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            attention = single_head_local_attention(d_k ** 0.5)
            res_q = []
            att_list = []
            for i in range(n_head):
                q_t, k_t, v_t = q[:, i*d_k:(i+1)*d_k], k[:, i*d_k:(i+1)*d_k], v[:, i*d_v:(i+1)*d_v]
                if show:
                    q_t, att = attention(q_t, k_t, v_t, kernel, pad, show)
                    att_list.append(att)
                else:
                    q_t = attention(q_t, k_t, v_t, kernel, pad)
                res_q.append(q_t)
            q = torch.cat(res_q, dim=1)
         
        q = self.fc(q)
        q += residual
        if self.norm:
            q = self.ln(q)
        if show:
            return q, att_list
        else:
            return q


class FocusFormer_Attention(nn.Module):
    """
        self_attn + cross_attn + ffn
    """
    def __init__(self, in_planes, n_head, d_k, d_v):
        super(FocusFormer_Attention, self).__init__()
        self.slf_attn = Multi_head_focus_attention(n_head, in_planes, d_k, d_v, norm=True)
        self.crs_attn = Multi_head_focus_attention(n_head, in_planes, d_k, d_v, norm=True)
        self.pos_ffn = PositionwiseFeedForward(in_planes, in_planes, norm=True)

    def forward(self, input1, input2, kernel, pad, show = False):
        if show:
            slf_1, slf_att_1 = self.slf_attn(input1, input1, input1, kernel, pad, show)
            slf_2, slf_att_2 = self.slf_attn(input2, input2, input2, kernel, pad, show)
            crs_1, crs_att_1 = self.crs_attn(slf_1,  slf_2,  slf_2, kernel, pad, show)
            crs_2, crs_att_2 = self.crs_attn(slf_2,  slf_1,  slf_1, kernel, pad, show)
            crs_1 = self.pos_ffn(crs_1)
            crs_2 = self.pos_ffn(crs_2)
            return crs_1, crs_2, slf_att_1, slf_att_2, crs_att_1, crs_att_2
        else:
            slf_1 = self.slf_attn(input1, input1, input1, kernel, pad)
            slf_2 = self.slf_attn(input2, input2, input2, kernel, pad)
            crs_1 = self.crs_attn(slf_1,  slf_2,  slf_2, kernel, pad)
            crs_2 = self.crs_attn(slf_2,  slf_1,  slf_1, kernel, pad)
            crs_1 = self.pos_ffn(crs_1)
            crs_2 = self.pos_ffn(crs_2)
            return crs_1, crs_2
