"""
MIXAD (Memory-Induced Explainable Time Series Anomaly Detection) 算法实现
基于 EasyTSAD 框架，适用于多元时序异常检测

核心技术：记忆增强、图卷积网络、时空建模、对比学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")


def get_default_device():
    """选择可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class MIXADDataset(Dataset):
    """MIXAD专用数据集类"""
    
    def __init__(self, data: np.ndarray, window_size: int, stride: int = 1):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.stride = stride
        self.num_samples = max(0, (len(data) - window_size) // stride + 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        window = self.data[start_idx:start_idx + self.window_size]
        return window, window


# ============= MIXAD核心组件实现 =============

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    """Gumbel Softmax采样"""
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GC(nn.Module):
    """图卷积层"""
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GC, self).__init__()
        self.cheb_k = cheb_k 
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []

        # 聚合邻居信息
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)

        # 变换
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


class STRGCCell(nn.Module):
    """时空递归图卷积单元"""
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(STRGCCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GC(dim_in+self.hidden_dim, 2*dim_out, cheb_k) 
        self.update = GC(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, supports)) 
        z, r = torch.split(z_r, self.hidden_dim, dim=-1) 
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class STRGC_Encoder(nn.Module):
    """时空递归图卷积编码器"""
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num 
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x 
        output_hidden = []

        for i in range(self.num_layers): 
            state = init_state[i] 
            inner_states = []

            for t in range(seq_length): 
                state = self.strgc_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)

            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden 
    
    def init_hidden(self, batch_size): 
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.strgc_cells[i].init_hidden_state(batch_size)) 
        return init_states


class STRGC_Decoder(nn.Module):
    """时空递归图卷积解码器"""
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports): 
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt 
        output_hidden = []

        for i in range(self.num_layers): 
            state = self.strgc_cells[i](current_inputs, init_state[i], supports) 
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, output_hidden


class MIXADModel(nn.Module):
    """MIXAD主模型"""
    def __init__(self, args):
        super(MIXADModel, self).__init__()
        self.num_nodes = args.num_nodes 
        self.input_dim = args.input_dim
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim 
        self.horizon = args.seq_len
        self.num_layers = args.num_rnn_layers
        self.cheb_k = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning 
        
        self.mem_num = args.mem_num
        self.mem_dim = args.mem_dim
        self.memory = self.construct_memory()

        self.encoder = STRGC_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = STRGC_Decoder(self.num_nodes, self.output_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        """构建记忆模块"""
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t: torch.Tensor): 
        """查询记忆模块"""
        query = torch.matmul(h_t, self.memory['Wq']) 
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory['Memory'])

        _, ind = torch.topk(att_score, k=min(2, self.mem_num), dim=-1) 
        pos = self.memory['Memory'][ind[:, :, 0]] 
        neg = self.memory['Memory'][ind[:, :, 1]] if self.mem_num > 1 else pos
        return value, query, pos, neg, att_score
    
    def scaled_laplacian(self, node_embeddings1, node_embeddings2, is_eval=False):
        """归一化图拉普拉斯函数"""
        node_num = self.num_nodes
        learned_graph = torch.mm(node_embeddings1, node_embeddings2.transpose(0, 1))
        
        # 归一化图
        norm1 = torch.norm(node_embeddings1, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(node_embeddings2, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm1, norm2.transpose(0, 1))
        learned_graph = learned_graph / (norm + 1e-6)
        learned_graph = (learned_graph + 1) / 2.
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
       
        # 制作稀疏邻接矩阵
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        mask = torch.eye(node_num, node_num).bool().to(node_embeddings1.device)
        adj.masked_fill_(mask, 0)
       
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        L = -W
        L[range(len(L)), range(len(L))] = d
        
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("特征值计算错误: {}".format(e))
            lambda_max = 1.0
       
        tilde = (2 * L / lambda_max - torch.eye(n).to(node_embeddings1.device))
        return adj, tilde
            
    def forward(self, x, batches_seen=None):
        # 节点嵌入和图构建
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        
        if self.training:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=False)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=False)
        else:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=True)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=True)
        supports = [learned_tilde1, learned_tilde2] 

        # 编码
        init_state = self.encoder.init_hidden(x.shape[0]) 
        h_en, state_en = self.encoder(x, init_state, supports) 
        h_t = h_en[:, -1, :, :] 

        # 记忆查询
        h_att, query, pos, neg, att_score = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
        
        # 解码
        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device) 
        out = []

        for t in range(self.horizon):
            h_de, ht_list = self.decoder(go, ht_list, supports)
            go = self.proj(h_de)
            out.append(go)

        output = torch.stack(out, dim=1)

        return output, h_att, query, pos, neg, att_score, (adj1, adj2)


def contrastive_loss(query, pos, neg, temperature=0.5):
    """对比损失函数"""
    # 计算相似度
    pos_sim = F.cosine_similarity(query, pos, dim=-1) / temperature
    neg_sim = F.cosine_similarity(query, neg, dim=-1) / temperature
    
    # 对比损失
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8))
    return loss.mean()


def consistency_loss(adj1, adj2):
    """一致性损失函数"""
    return F.mse_loss(adj1, adj2)


def kl_loss(att_score):
    """KL散度损失"""
    uniform = torch.ones_like(att_score) / att_score.shape[-1]
    return F.kl_div(torch.log(att_score + 1e-8), uniform, reduction='batchmean')


class MIXAD(BaseMethod):
    """MIXAD异常检测方法"""
    
    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = get_default_device()
        self.model = None
        
        if params is None:
            params = {}
        
        # MIXAD参数配置
        self.config = {
            'window_size': params.get('window', 30),
            'batch_size': params.get('batch_size', 32),
            'epochs': params.get('epochs', 10),
            'learning_rate': params.get('lr', 1e-3),
            'rnn_units': params.get('rnn_units', 64),
            'max_diffusion_step': params.get('max_diffusion_step', 2),
            'num_rnn_layers': params.get('num_rnn_layers', 1),
            'mem_num': params.get('mem_num', 3),
            'mem_dim': params.get('mem_dim', 32),
            'lamb_cont': params.get('lamb_cont', 0.01),
            'lamb_cons': params.get('lamb_cons', 0.1),
            'lamb_kl': params.get('lamb_kl', 0.0001),
            'stride': params.get('stride', 1),
            'cl_decay_steps': params.get('cl_decay_steps', 2000),
            'use_curriculum_learning': params.get('use_curriculum_learning', True)
        }
        
        print(f"[LOG] MIXAD初始化完成，使用设备: {self.device}")
        print(f"[LOG] 核心特性: 记忆增强、图卷积网络、时空建模")
    
    def train_valid_phase(self, tsTrain: MTSData):
        """训练阶段"""
        print(f"\n[LOG] ========== MIXAD训练开始 ==========")
        
        train_data = tsTrain.train
        self.n_features = train_data.shape[1]
        
        print(f"[LOG] 训练数据: {train_data.shape}")
        
        # 准备参数
        class Args:
            def __init__(self, config, n_features):
                self.num_nodes = n_features
                self.input_dim = 1
                self.output_dim = 1
                self.seq_len = config['window_size']
                self.rnn_units = config['rnn_units']
                self.max_diffusion_step = config['max_diffusion_step']
                self.num_rnn_layers = config['num_rnn_layers']
                self.mem_num = config['mem_num']
                self.mem_dim = config['mem_dim']
                self.cl_decay_steps = config['cl_decay_steps']
                self.use_curriculum_learning = config['use_curriculum_learning']
        
        args = Args(self.config, self.n_features)
        
        # 创建模型
        self.model = MIXADModel(args).to(self.device)
        
        # 创建数据集
        dataset = MIXADDataset(train_data, self.config['window_size'], self.config['stride'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=0)
        
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        print(f"[LOG] 开始训练，{self.config['epochs']}轮")
        
        batches_seen = 0
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_data, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                try:
                    batch_data = batch_data.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # 重塑数据 [B, T, N, 1]
                    batch_data = batch_data.unsqueeze(-1)
                    batch_targets = batch_targets.unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    output, h_att, query, pos, neg, att_score, (adj1, adj2) = self.model(batch_data, batches_seen)
                    
                    # 重构损失
                    rec_loss = criterion(output, batch_targets)
                    
                    # 对比损失
                    cont_loss = contrastive_loss(query, pos, neg)
                    
                    # 一致性损失
                    cons_loss = consistency_loss(adj1, adj2)
                    
                    # KL损失
                    kl_loss_val = kl_loss(att_score)
                    
                    # 总损失
                    total_batch_loss = (rec_loss + 
                                      self.config['lamb_cont'] * cont_loss +
                                      self.config['lamb_cons'] * cons_loss +
                                      self.config['lamb_kl'] * kl_loss_val)
                    
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                    
                    total_loss += total_batch_loss.item()
                    num_batches += 1
                    batches_seen += 1
                    
                except Exception as e:
                    print(f"[WARNING] 批次训练错误: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        print(f"[LOG] ========== MIXAD训练完成 ==========\n")
    
    def test_phase(self, tsData: MTSData):
        """测试阶段"""
        print(f"\n[LOG] ========== MIXAD测试开始 ==========")
        
        test_data = tsData.test
        test_dataset = MIXADDataset(test_data, self.config['window_size'], stride=1)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch_data, batch_targets in tqdm(test_loader, desc="MIXAD测试"):
                try:
                    batch_data = batch_data.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # 重塑数据
                    batch_data = batch_data.unsqueeze(-1)
                    batch_targets = batch_targets.unsqueeze(-1)
                    
                    # 前向传播
                    output, _, _, _, _, _, _ = self.model(batch_data)
                    
                    # 重构误差
                    rec_error = torch.mean((output - batch_targets) ** 2, dim=(1, 2, 3))
                    scores.extend(rec_error.cpu().numpy())
                    
                except Exception as e:
                    print(f"[WARNING] 测试批次错误: {e}")
                    continue
        
        # 处理分数长度
        full_scores = np.zeros(len(test_data))
        if len(scores) > 0:
            full_scores[:self.config['window_size']-1] = scores[0] if scores else 0.0
            end_idx = min(len(scores), len(full_scores) - self.config['window_size'] + 1)
            if end_idx > 0:
                full_scores[self.config['window_size']-1:self.config['window_size']-1+end_idx] = scores[:end_idx]
        
        self.__anomaly_score = full_scores
        print(f"[LOG] 异常分数范围: [{np.min(full_scores):.4f}, {np.max(full_scores):.4f}]")
        print(f"[LOG] ========== MIXAD测试完成 ==========\n")
    
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def param_statistic(self, save_file):
        """参数统计"""
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"""
                MIXAD (Memory-Induced Explainable Time Series Anomaly Detection) 参数统计:
                ==================================================================
                总参数数量: {total_params:,}
                可训练参数数量: {trainable_params:,}
                窗口大小: {self.config['window_size']}
                批量大小: {self.config['batch_size']}
                训练轮数: {self.config['epochs']}
                学习率: {self.config['learning_rate']}
                RNN单元数: {self.config['rnn_units']}
                记忆单元数: {self.config['mem_num']}
                记忆维度: {self.config['mem_dim']}
                对比损失权重: {self.config['lamb_cont']}
                一致性损失权重: {self.config['lamb_cons']}
                KL损失权重: {self.config['lamb_kl']}
                ==================================================================
                MIXAD核心特性:
                ✅ 记忆增强机制 (可学习的记忆模块)
                ✅ 时空递归图卷积 (STRGC)
                ✅ 自适应图学习 (Gumbel Softmax)
                ✅ 对比学习 (正负样本对比)
                ✅ 多任务学习 (重构+对比+一致性+KL)
                ✅ 课程学习 (渐进式训练策略)
                ==================================================================
            """
        else:
            param_info = "MIXAD模型尚未初始化"
            
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(param_info)


# ============= 主程序入口 =============
if __name__ == "__main__":
    print("🚀 ========== MIXAD: 记忆增强时序异常检测 ==========")
    
    gctrl = TSADController()
    
    datasets = ["machine-1", "machine-2", "machine-3"]
    gctrl.set_dataset(dataset_type="MTS", dirname="./datasets", datasets=datasets)

    method = "MIXAD"

    # MIXAD配置
    gctrl.run_exps(
        method=method,
        training_schema="mts",
        hparams={
            "window": 30,              # 窗口大小
            "batch_size": 32,          # 批次大小
            "epochs": 10,              # 训练轮数
            "lr": 1e-3,               # 学习率
            "rnn_units": 64,          # RNN单元数
            "max_diffusion_step": 2,   # 最大扩散步数
            "num_rnn_layers": 1,      # RNN层数
            "mem_num": 3,             # 记忆单元数
            "mem_dim": 32,            # 记忆维度
            "lamb_cont": 0.01,        # 对比损失权重
            "lamb_cons": 0.1,         # 一致性损失权重
            "lamb_kl": 0.0001,        # KL损失权重
            "stride": 1,              # 步长
            "cl_decay_steps": 2000,   # 课程学习衰减步数
            "use_curriculum_learning": True  # 是否使用课程学习
        },
        preprocess="z-score",
    )
       
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    
    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema="mts")
    gctrl.plots(method=method, training_schema="mts")
    
    print("🎉 ========== MIXAD执行完毕 ==========") 