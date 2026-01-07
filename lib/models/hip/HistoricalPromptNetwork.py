import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 引入同目录下自定义模块（编码器相关组件）
from lib.models.hip.modules import *
import skimage
from thop import profile

# 自定义 softmax，保留 top-k 元素，其余置零，用于注意力稀疏化
def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x

class HistoricalPromptDecoder(nn.Module):
    def __init__(self, isEval=False):
        super().__init__()

    def set_eval(self, mem_max=150):
        # 推理阶段重置解码器内部状态，包括临时与持久记忆空间
        self.topk = 20
        self.CK = None
        self.CV = None
        self.mem_k = None
        self.mem_v = None
        self.mem_max = mem_max
        self.mem_cnt = 0
        self.imgs = None
        self.path = None

    def setPath(self, path, seqName, frameId):
        # 在可视化时保存路径信息（未使用时保持 None）
        self.path = f"{path}/{seqName}/{frameId}"

    def add_memory(self, key, value, is_temp=False, searchRegionImg=None):
        # 将新的记忆（key/value）写入记忆池，可选择保存对应的搜索区域图像
        if searchRegionImg is not None:
            if self.imgs is None:
                self.imgs = torch.from_numpy(searchRegionImg[:,:,::-1].copy())
            else:
                img = torch.from_numpy(searchRegionImg[:,:,::-1].copy())
                self.imgs = torch.cat([self.imgs, img], dim=1)
        self.temp_k = None
        self.temp_v = None
        # 将特征展平成 (B, C, HW) 形式，以便后续矩阵运算
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.mem_cnt = 1
        else:
            # 支持临时记忆缓冲（仅当前帧使用，不写入常驻记忆）
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                # 若记忆已满，则丢弃最旧的一个 slot（FIFO 策略）
                if self.mem_cnt == self.mem_max:
                    self.mem_k = self.mem_k[:, :, key.shape[2]:]
                    self.mem_v = self.mem_v[:, :, value.shape[2]:]
                    self.mem_cnt -= 1
                
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                self.mem_cnt += 1
 
    def match_memory(self, qk):
        # 用于旧版接口：根据查询 key 与记忆 key 计算匹配，并输出记忆 value
        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = torch.bmm(affinity, mv)

        return readout_mem.view(qk.shape[0], self.CV, -1)
    
    def _global_matching(self, mk, qk):
        # 计算全局匹配注意力得分（使用 L2 范数近似 cosine）
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def get_affinity(self, mk, qk, eval=False, visualize=False):
        """
         @Attention : mk [B C T H W]
        训练/推理共用接口：根据记忆 key 与查询 key 计算注意力权重
        """
        #B, CK, THW = mk.shape
        if not eval:
            mk = mk.flatten(start_dim=2)
        B, CK, THW = mk.shape
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum # Normalize
        return affinity

    def readout(self, affinity, mv, qv, eval=False):
        # 根据注意力权重，从记忆 value 中读取并融合当前特征
        if not eval:
            mv = mv.flatten(start_dim=2)
        B, CV, THW = mv.shape
        _, _, H, W = qv.shape
        #mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        # 拼接记忆读出特征与当前特征，作为历史提示
        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out # [B (Cv+Cv) H W]


class HistoricalPromptNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器负责生成 key/value；解码器负责记忆匹配与读出
        self.encoder = HistoricalPromptEncoder() 
        self.decoder = HistoricalPromptDecoder()

    def set_eval(self, mem_max):
        # 进入推理模式，重置解码器状态
        self.decoder.set_eval(mem_max)

    def addMemory(self, addMemKey, addMemValue, searchRegionImg):
        # 对外提供写入记忆的接口
        self.decoder.add_memory(addMemKey, addMemValue, searchRegionImg=searchRegionImg)

    def encode(self, frame, kf16, mask, other_mask=None): 
        # 将当前帧图像、关键特征和前景 mask 编码为记忆 value
        f16 = self.encoder(frame, kf16, mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def eval_decode(self, queryFrame_key, queryFrame_value):
        # 推理时：仅使用持久记忆，读取历史提示
        affinity = self.decoder.get_affinity(self.decoder.mem_k, queryFrame_key, eval=True, visualize=True)
        return self.decoder.readout(affinity=affinity, mv=self.decoder.mem_v, qv=queryFrame_value, eval=True)


    def decode(self, queryFrame_key, queryFrame_value, memory_key, memory_value):
        """
            queryFrame_key : [B Ck H W]
            queryFrame_value : [B C_v H W]
            memory_key : [B Ck T H W]
            memory_value : [B C_v T H W]
        """
        # 训练阶段：支持将模板或历史帧作为记忆，与当前查询帧进行匹配
        affinity = self.decoder.get_affinity(memory_key, queryFrame_key, visualize=False)
        return self.decoder.readout(affinity=affinity, mv=memory_value, qv=queryFrame_value)

    def forward(self, mode, *args, **kwargs):
        # 根据模式选择编码或解码分支
        if mode == 'encode':
            return self.encode(*args, **kwargs)
        elif mode == 'train_decode':
            return self.decode(*args, **kwargs)
        elif mode == 'eval_decode':
            return self.eval_decode(*args, **kwargs)
        else:
            raise NotImplementedError

