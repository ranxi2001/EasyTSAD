from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from torch.utils.data import DataLoader, Dataset
import tqdm
from EasyTSAD.Controller import TSADController
import time
import copy
import pandas as pd
from torch.optim import lr_scheduler
from torch.nn.functional import gumbel_softmax
from sklearn.preprocessing import StandardScaler
from einops import rearrange

from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import TSData


# ============================== CATCH 依赖组件 ==============================

# RevIN 层
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# 动态对比损失
class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask

        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))

        eye = torch.eye(attn_mask.shape[-1]).unsqueeze(0).repeat(b, 1, 1).to(attn_mask.device)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(eye.reshape(b, -1) - attn_mask.reshape((b, -1)),
                                                                p=1, dim=-1)
        loss = clustering_loss.mean(1) + self.k * regular_loss

        mean_loss = loss.mean()
        return mean_loss


# 通道掩码生成器
class channel_mask_generator(torch.nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(torch.nn.Linear(input_size * 2, n_vars, bias=False), nn.Sigmoid())
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)
        inverse_eye = 1 - torch.eye(self.n_vars).to(x.device)
        diag = torch.eye(self.n_vars).to(x.device)
        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag
        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        # 只取第一个通道（对应选中的概率）
        resample_matrix = resample_matrix[..., 0]
        resample_matrix = rearrange(resample_matrix, '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix


# Transformer 组件
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(k=regular_lambda, temperature=temperature)

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        
        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores

            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, regular_lambda=regular_lambda,
                                    temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = 0
        for attn, ff in self.layers:
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)
            if dcloss is not None:
                total_loss += dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers) if total_loss != 0 else torch.tensor(0.0)
        return x, attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()

        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, regular_lambda=regular_lambda,
                                         temperature=temperature)

        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss


# 展平头部
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)

        return x


# CATCH 主模型
class CATCHModel(nn.Module):
    def __init__(self, configs, **kwargs):
        super(CATCHModel, self).__init__()

        self.revin_layer = RevIN(configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last)
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        
        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=configs.c_in)
        self.frequency_transformer = Trans_C(dim=configs.cf_dim, depth=configs.e_layers, heads=configs.n_heads,
                                       mlp_dim=configs.d_ff,
                                       dim_head=configs.head_dim, dropout=configs.dropout,
                                       patch_dim=configs.patch_size * 2,
                                       horizon=self.horizon * 2, d_model=configs.d_model * 2,
                                       regular_lambda=configs.regular_lambda, temperature=configs.temperature)

        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
                                    head_dropout=configs.head_dropout)

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)

        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):
        z = self.revin_layer(z, 'norm')

        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)

        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')

        return z, complex_z.permute(0, 2, 1), dcloss


# 频率损失函数
class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        self.mask = None

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
            
        if self.configs.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        else:
            raise NotImplementedError

        if self.configs.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean(dim=self.dim, keepdim=self.keep_dim) if self.configs.module_first else loss_auxi.mean(dim=self.dim, keepdim=self.keep_dim).abs()
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim, keepdim=self.keep_dim) if self.configs.module_first else (loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
        else:
            raise NotImplementedError
        return loss_auxi


class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):
        output_patch = outputs.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor([range(start_indices[i], end_indices[i]) for i in range(n)]).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c)).to(main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / (non_zero_cnt + 1e-8)

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss


# 工具函数
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: 
            print('Updating learning rate to {}'.format(lr))


# 数据集类
class TSDataset(Dataset):
    def __init__(self, data, win_size, step, mode='train'):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        
        # 处理多种数据格式
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                self.data = data.reshape(-1, 1)
            else:
                self.data = data
        else:
            self.data = np.array(data)
            if len(self.data.shape) == 1:
                self.data = self.data.reshape(-1, 1)
        
        if mode == 'train':
            self.sample_num = max(0, (len(self.data) - self.win_size) // self.step + 1)
        else:
            self.sample_num = max(0, len(self.data) - self.win_size + 1)
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        if self.mode == 'train':
            start = index * self.step
        else:
            start = index
        end = start + self.win_size
        
        sample = self.data[start:end]
        
        if self.mode == 'thre':
            return torch.FloatTensor(sample), torch.zeros(1)
        else:
            return torch.FloatTensor(sample)


def train_val_split(data, train_ratio, val_ratio):
    """分割训练和验证数据"""
    n = len(data)
    train_end = int(n * train_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:]
    
    return train_data, val_data


# 配置类
class TransformerConfig:
    def __init__(self, **kwargs):
        # 默认超参数
        defaults = {
            "lr": 0.0001,
            "Mlr": 0.00001,
            "e_layers": 3,
            "n_heads": 2,
            "cf_dim": 64,
            "d_ff": 256,
            "d_model": 128,
            "head_dim": 64,
            "individual": 0,
            "dropout": 0.2,
            "head_dropout": 0.1,
            "auxi_loss": "MAE",
            "auxi_type": "complex",
            "auxi_mode": "fft",
            "auxi_lambda": 0.005,
            "score_lambda": 0.05,
            "regular_lambda": 0.5,
            "temperature": 0.07,
            "patch_stride": 8,
            "patch_size": 16,
            "inference_patch_stride": 1,
            "inference_patch_size": 32,
            "dc_lambda": 0.005,
            "module_first": True,
            "mask": False,
            "pretrained_model": None,
            "num_epochs": 3,
            "batch_size": 32,
            "patience": 3,
            "seq_len": 96,
            "pct_start": 0.3,
            "revin": 1,
            "affine": 0,
            "subtract_last": 0,
            "lradj": "type1",
        }
        
        for key, value in defaults.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


# ============================== CATCH 算法实现 ==============================

class Catch(BaseMethod):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.__anomaly_score = None
        
        # 配置参数，使用传入的hparams和默认值
        self.config = TransformerConfig(**hparams)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.config)
        self.seq_len = self.config.seq_len
        
        print(f"[LOG] CATCH算法初始化完成，设备: {self.device}")

    def _convert_to_dataframe(self, data):
        """将数据转换为pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(np.array(data))

    def _hyper_param_tune(self, train_data):
        """根据训练数据调整超参数"""
        if isinstance(train_data, pd.DataFrame):
            column_num = train_data.shape[1]
        else:
            column_num = train_data.shape[1] if len(train_data.shape) > 1 else 1
            
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.c_in = column_num
        self.config.label_len = 48

    def _validate(self, valid_loader):
        """验证模型"""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_input in valid_loader:
                batch_input = batch_input.to(self.device)
                output, _, _ = self.model(batch_input)
                loss = self.criterion(output, batch_input).detach().cpu().numpy()
                total_loss.append(loss)
        
        self.model.train()
        return np.mean(total_loss) if total_loss else 0.0
    
    def train_valid_phase(self, tsTrain: TSData):
        '''
        Define train and valid phase for naive mode. All time series needed are saved in tsTrain. 
        
        tsTrain's property :
            train (np.ndarray):
                The training set in numpy format;
            valid (np.ndarray):
                The validation set in numpy format;
            test (np.ndarray):
                The test set in numpy format;
            train_label (np.ndarray):
                The labels of training set in numpy format;
            test_label (np.ndarray):
                The labels of test set in numpy format;
            valid_label (np.ndarray):
                The labels of validation set in numpy format;
            info (dict):
                Some informations about the dataset, which might be useful.
            
        NOTE : test and test_label are not accessible in training phase
        '''
        print(f"[LOG] CATCH开始训练，训练数据形状: {tsTrain.train.shape}")
        
        # 转换数据格式
        train_data = self._convert_to_dataframe(tsTrain.train)
        valid_data = self._convert_to_dataframe(tsTrain.valid)
        
        # 调整超参数
        self._hyper_param_tune(train_data)
        
        # 标准化
        self.scaler.fit(train_data.values)
        train_data_scaled = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns if hasattr(train_data, 'columns') else None
        )
        valid_data_scaled = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns if hasattr(valid_data, 'columns') else None
        )
        
        # 创建模型
        self.model = CATCHModel(self.config)
        self.model.to(self.device)
        
        # 创建数据加载器
        train_dataset = TSDataset(train_data_scaled, self.config.seq_len, 1, mode='train')
        valid_dataset = TSDataset(valid_data_scaled, self.config.seq_len, 1, mode='valid')
        
        if len(train_dataset) == 0 or len(valid_dataset) == 0:
            print("[WARNING] 数据集太小，无法进行训练")
            return
        
        train_loader = DataLoader(train_dataset, batch_size=min(self.config.batch_size, len(train_dataset)), shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=min(self.config.batch_size, len(valid_dataset)), shuffle=False)
        
        # 初始化优化器
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]
        self.optimizer = torch.optim.Adam(main_params, lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)
        
        train_steps = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        
        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )
        
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            train_loss = []
            self.model.train()
            
            step = min(int(len(train_loader) / 10), 100) if len(train_loader) > 0 else 1
            for i, batch_input in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                batch_input = batch_input.float().to(self.device)
                output, output_complex, dcloss = self.model(batch_input)
                
                rec_loss = self.criterion(output, batch_input)
                norm_input = self.model.revin_layer(batch_input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)
                
                loss = rec_loss + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxi_loss
                train_loss.append(loss.item())
                
                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()
                
                loss.backward()
                self.optimizer.step()
            
            # 验证
            valid_loss = self._validate(valid_loader)
            train_loss = np.average(train_loss) if train_loss else 0.0
            
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}")
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)
        
        return
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        '''
        Define train and valid phase for all-in-one mode. All time series needed are saved in tsTrains. 
        
        tsTrain's structure:
            {
                "name of time series 1": tsData1,
                "name of time series 2": tsData2,
                ...
            }
            
        '''
        print("[LOG] CATCH 不支持 all-in-one 模式，使用 naive 模式代替")
        return
        
    def test_phase(self, tsData: TSData):
        '''
        Define test phase for each time series. 
        '''
        print(f"[LOG] CATCH开始测试，测试数据形状: {tsData.test.shape}")
        
        # 转换并标准化测试数据
        test_data = self._convert_to_dataframe(tsData.test)
        test_data_scaled = pd.DataFrame(
            self.scaler.transform(test_data.values),
            columns=test_data.columns if hasattr(test_data, 'columns') else None
        )
        
        # 加载最佳模型
        if hasattr(self, 'early_stopping') and hasattr(self.early_stopping, 'check_point'):
            self.model.load_state_dict(self.early_stopping.check_point)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 创建测试数据加载器
        test_dataset = TSDataset(test_data_scaled, self.config.seq_len, 1, mode='thre')
        
        if len(test_dataset) == 0:
            print("[WARNING] 测试数据集为空")
            self.__anomaly_score = np.array([])
            return
            
        test_loader = DataLoader(test_dataset, batch_size=min(self.config.batch_size, len(test_dataset)), shuffle=False)
        
        # 计算异常分数
        temp_anomaly_criterion = nn.MSELoss(reduce=False)
        freq_anomaly_criterion = frequency_criterion(self.config)
        
        attens_energy = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, _, _ = self.model(batch_x)
                
                temp_score = torch.mean(temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + self.config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
        
        if attens_energy:
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        else:
            attens_energy = np.array([])
            
        self.__anomaly_score = np.array(attens_energy)
        
        print(f"[LOG] 异常分数计算完成，长度: {len(self.__anomaly_score)}")
         
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        if hasattr(self, 'model'):
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_info = f"CATCH Model - Total trainable parameters: {total_params}\n"
            param_info += f"Configuration:\n"
            for key, value in self.config.__dict__.items():
                param_info += f"  {key}: {value}\n"
        else:
            param_info = "CATCH Model - Model not initialized yet\n"
        
        with open(save_file, 'w') as f:
            f.write(param_info)


# ============================== 主执行程序 ==============================

if __name__ == "__main__":
    
    # 创建全局控制器
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["machine-1", "machine-2", "machine-3"]
    dataset_types = "MTS"
    
    print("[LOG] 开始设置数据集")
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="./datasets",
        datasets=datasets,
    )
    print("[LOG] 数据集设置完成")

    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import TSData
    from typing import Dict

    class Catch(BaseMethod):
        def __init__(self, hparams) -> None:
            super().__init__()
            self.__anomaly_score = None
            
            # 配置参数，使用传入的hparams和默认值
            self.config = TransformerConfig(**hparams)
            self.scaler = StandardScaler()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.criterion = nn.MSELoss()
            self.auxi_loss = frequency_loss(self.config)
            self.seq_len = self.config.seq_len
            
            print(f"[LOG] CATCH算法初始化完成，设备: {self.device}")

        def _convert_to_dataframe(self, data):
            """将数据转换为pandas DataFrame"""
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(np.array(data))

        def _hyper_param_tune(self, train_data):
            """根据训练数据调整超参数"""
            if isinstance(train_data, pd.DataFrame):
                column_num = train_data.shape[1]
            else:
                column_num = train_data.shape[1] if len(train_data.shape) > 1 else 1
                
            self.config.enc_in = column_num
            self.config.dec_in = column_num
            self.config.c_out = column_num
            self.config.c_in = column_num
            self.config.label_len = 48

        def _validate(self, valid_loader):
            """验证模型"""
            total_loss = []
            self.model.eval()
            
            with torch.no_grad():
                for batch_input in valid_loader:
                    batch_input = batch_input.to(self.device)
                    output, _, _ = self.model(batch_input)
                    loss = self.criterion(output, batch_input).detach().cpu().numpy()
                    total_loss.append(loss)
            
            self.model.train()
            return np.mean(total_loss) if total_loss else 0.0
        
        def train_valid_phase(self, tsTrain: TSData):
            '''
            Define train and valid phase for naive mode. All time series needed are saved in tsTrain. 
            
            tsTrain's property :
                train (np.ndarray):
                    The training set in numpy format;
                valid (np.ndarray):
                    The validation set in numpy format;
                test (np.ndarray):
                    The test set in numpy format;
                train_label (np.ndarray):
                    The labels of training set in numpy format;
                test_label (np.ndarray):
                    The labels of test set in numpy format;
                valid_label (np.ndarray):
                    The labels of validation set in numpy format;
                info (dict):
                    Some informations about the dataset, which might be useful.
                    
            NOTE : test and test_label are not accessible in training phase
            '''
            print(f"[LOG] CATCH开始训练，训练数据形状: {tsTrain.train.shape}")
            
            # 转换数据格式
            train_data = self._convert_to_dataframe(tsTrain.train)
            valid_data = self._convert_to_dataframe(tsTrain.valid)
            
            # 调整超参数
            self._hyper_param_tune(train_data)
            
            # 标准化
            self.scaler.fit(train_data.values)
            train_data_scaled = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns if hasattr(train_data, 'columns') else None
            )
            valid_data_scaled = pd.DataFrame(
                self.scaler.transform(valid_data.values),
                columns=valid_data.columns if hasattr(valid_data, 'columns') else None
            )
            
            # 创建模型
            self.model = CATCHModel(self.config)
            self.model.to(self.device)
            
            # 创建数据加载器
            train_dataset = TSDataset(train_data_scaled, self.config.seq_len, 1, mode='train')
            valid_dataset = TSDataset(valid_data_scaled, self.config.seq_len, 1, mode='valid')
            
            if len(train_dataset) == 0 or len(valid_dataset) == 0:
                print("[WARNING] 数据集太小，无法进行训练")
                return
            
            train_loader = DataLoader(train_dataset, batch_size=min(self.config.batch_size, len(train_dataset)), shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=min(self.config.batch_size, len(valid_dataset)), shuffle=False)
            
            # 初始化优化器
            main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]
            self.optimizer = torch.optim.Adam(main_params, lr=self.config.lr)
            self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)
            
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=train_steps,
                pct_start=self.config.pct_start,
                epochs=self.config.num_epochs,
                max_lr=self.config.lr,
            )
            
            schedulerM = lr_scheduler.OneCycleLR(
                optimizer=self.optimizerM,
                steps_per_epoch=train_steps,
                pct_start=self.config.pct_start,
                epochs=self.config.num_epochs,
                max_lr=self.config.Mlr,
            )
            
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
            
            # 训练循环
            for epoch in range(self.config.num_epochs):
                train_loss = []
                self.model.train()
                
                step = min(int(len(train_loader) / 10), 100) if len(train_loader) > 0 else 1
                for i, batch_input in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    
                    batch_input = batch_input.float().to(self.device)
                    output, output_complex, dcloss = self.model(batch_input)
                    
                    rec_loss = self.criterion(output, batch_input)
                    norm_input = self.model.revin_layer(batch_input, 'transform')
                    auxi_loss = self.auxi_loss(output_complex, norm_input)
                    
                    loss = rec_loss + self.config.dc_lambda * dcloss + self.config.auxi_lambda * auxi_loss
                    train_loss.append(loss.item())
                    
                    if (i + 1) % step == 0:
                        self.optimizerM.step()
                        self.optimizerM.zero_grad()
                    
                    loss.backward()
                    self.optimizer.step()
                
                # 验证
                valid_loss = self._validate(valid_loader)
                train_loss = np.average(train_loss) if train_loss else 0.0
                
                print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Train Loss: {train_loss:.7f}, Valid Loss: {valid_loss:.7f}")
                
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
                adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
                adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)
            
            return
                
        def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            '''
            Define train and valid phase for all-in-one mode. All time series needed are saved in tsTrains. 
            
            tsTrain's structure:
                {
                    "name of time series 1": tsData1,
                    "name of time series 2": tsData2,
                    ...
                }
                
            '''
            print("[LOG] CATCH 不支持 all-in-one 模式，使用 naive 模式代替")
            return
            
        def test_phase(self, tsData: TSData):
            '''
            Define test phase for each time series. 
            '''
            print(f"[LOG] CATCH开始测试，测试数据形状: {tsData.test.shape}")
            
            # 转换并标准化测试数据
            test_data = self._convert_to_dataframe(tsData.test)
            test_data_scaled = pd.DataFrame(
                self.scaler.transform(test_data.values),
                columns=test_data.columns if hasattr(test_data, 'columns') else None
            )
            
            # 加载最佳模型
            if hasattr(self, 'early_stopping') and hasattr(self.early_stopping, 'check_point'):
                self.model.load_state_dict(self.early_stopping.check_point)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 创建测试数据加载器
            test_dataset = TSDataset(test_data_scaled, self.config.seq_len, 1, mode='thre')
            
            if len(test_dataset) == 0:
                print("[WARNING] 测试数据集为空")
                self.__anomaly_score = np.array([])
                return
                
            test_loader = DataLoader(test_dataset, batch_size=min(self.config.batch_size, len(test_dataset)), shuffle=False)
            
            # 计算异常分数
            temp_anomaly_criterion = nn.MSELoss(reduce=False)
            freq_anomaly_criterion = frequency_criterion(self.config)
            
            attens_energy = []
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = batch_x.float().to(self.device)
                    outputs, _, _ = self.model(batch_x)
                    
                    temp_score = torch.mean(temp_anomaly_criterion(batch_x, outputs), dim=-1)
                    freq_score = torch.mean(freq_anomaly_criterion(batch_x, outputs), dim=-1)
                    score = (temp_score + self.config.score_lambda * freq_score).detach().cpu().numpy()
                    attens_energy.append(score)
            
            if attens_energy:
                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            else:
                attens_energy = np.array([])
                
            self.__anomaly_score = np.array(attens_energy)
            
            print(f"[LOG] 异常分数计算完成，长度: {len(self.__anomaly_score)}")
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            if hasattr(self, 'model'):
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                param_info = f"CATCH Model - Total trainable parameters: {total_params}\n"
                param_info += f"Configuration:\n"
                for key, value in self.config.__dict__.items():
                    param_info += f"  {key}: {value}\n"
            else:
                param_info = "CATCH Model - Model not initialized yet\n"
            
            with open(save_file, 'w') as f:
                f.write(param_info)
    """============= [算法运行] ============="""
    # 指定方法和训练模式
    training_schema = "naive"
    method = "Catch"
    
    # 运行模型
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "seq_len": 96,
            "batch_size": 32,
            "num_epochs": 3,
            "lr": 0.0001,
            "patience": 3,
            "e_layers": 2,
            "n_heads": 2,
            "cf_dim": 64,
            "d_ff": 256,
            "d_model": 128,
            "head_dim": 64,
            "dropout": 0.2,
            "auxi_lambda": 0.005,
            "score_lambda": 0.05,
            "regular_lambda": 0.5,
            "temperature": 0.07,
            "patch_stride": 8,
            "patch_size": 16,
            "dc_lambda": 0.005,
        },
        preprocess="z-score",
    )
       
        
    """============= [评估设置] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # 指定评估协议
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
        
    """============= [绘图设置] ============="""
    
    # 为每条曲线绘制异常分数
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
 