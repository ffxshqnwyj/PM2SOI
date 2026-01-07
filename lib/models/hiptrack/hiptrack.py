"""
Basic HIPTrack model.
"""
# --- 标准库 ---
import math
import os
from typing import List

# --- 第三方库 ---
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from thop import profile
from thop.utils import clever_format

# --- 项目内模块 ---
from lib.models.layers.head import build_box_head
from lib.models.hiptrack.vit import vit_base_patch16_224
from lib.models.hiptrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import numpy as np
import cv2
import random
from lib.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, visualizeDuringTraining
import torchvision.ops as ops
from lib.models.hip import HistoricalPromptNetwork
from lib.models.hip.modules import KeyProjection
from lib.models.hip import ResBlock
import thop

class HIPTrack(nn.Module):
    """ This is the base class for HIPTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", vis_during_train=False, new_hip=False, memory_max=150, update_interval=20):
        super().__init__()
        
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # 历史提示模块：负责存储历史帧并提供动态提示
        self.HIP = HistoricalPromptNetwork()
        # 将 backbone 输出的高维特征压缩到 提示key 维度  768 -> 64
        self.key_proj = KeyProjection(768, keydim=64)
        # 额外通道压缩，用于减轻解码计算量
        self.key_comp = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        # 将历史提示与当前搜索特征融合的残差块
        self.searchRegionFusion = ResBlock(768, 768)
        self.new_hip = new_hip
        # 历史提示更新间隔
        self.update_interval = update_interval
        if self.new_hip:
            # new_hip 为 True 时，对模板特征/图像进行上采样以提高 mask 精度
            self.upsample = nn.Upsample(scale_factor=2.0, align_corners=True, mode="bilinear")
        self.memorys = []
        # 历史提示最大容量
        self.mem_max = memory_max
        # motion-aware memory gating attributes
        self.template_app_feat = None
        self.memory_score_thresh = 0.0
        self.memory_motion_thresh = 0.0
        self.second_peak_ratio = 0.85
        self.second_peak_min_score = 0.25
        self.second_peak_min_dist = 2.0
        self.memory_force_interval = 3
        self.memory_skip_count = 0

    def set_eval(self):
        # 推理前重置历史提示网络，设定最大记忆数
        self.HIP.set_eval(mem_max=self.mem_max)
        self.memory_skip_count = 0

    def forward_track(self, index: int, template: torch.Tensor, template_boxes: torch.Tensor, search: torch.Tensor, ce_template_mask=None, ce_keep_rate=None, searchRegionImg=None, info=None, motion_bbox=None):
        #self.HIP.memory.setPath("./visualizeMemBank", info, index)
        # 推理阶段入口：index 表示当前帧序号
        if index <= 10:
            # 前10帧：优先用模板帧构建初始记忆，不依赖历史提示

            # 1.backbone
            # 输入：
            #   template : [B, 3, H_t, W_t]（默认 192x192）模板图像
            #   search   : [B, 3, H_s, W_s]（默认 384x384）搜索区域
            # 输出：
            #   x : [B, (H_t/16)^2 + (H_s/16)^2, C]（C=768），拼接后的模板+搜索token  [B, 720, 768]
            #   aux_dict : 包含 CE 相关索引、注意力等辅助信息
            x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            
            # 2.历史提示网络
            # 将模板图像上采样到原尺度的两倍，提升掩码精度 [B,3,192,192] -> [B,3,384,384]
            upsampled_template = self.upsample(template)
            template_mask = self.generateMask([None, None, None], 
                                              template_boxes, 
                                              upsampled_template, x, 
                                              visualizeMask=False, cxcywh=False)

            # 从 backbone 输出中取出模板部分特征，并恢复到空间布局
            # 取出模板 token，view 成 [B, C, Ht/16, Wt/16]  （默认12×12） [B,768,12,12]
            template_feature = x[:, :(Ht // 16)**2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)

            # 同样对模板特征进行上采样，以对齐更精细的掩码 [B,768,24,24]
            template_feature = self.upsample(template_feature)
            if self.template_app_feat is None:
                template_vec = self._get_app_feature(template_feature)
                if template_vec is not None:
                    self.template_app_feat = template_vec.detach()

            # 使用模板图像/特征/掩码送入encoder生成 prompt value(注意力计算中的V)
            ref_v_template = self.HIP('encode', 
                            upsampled_template, 
                            template_feature, 
                            template_mask.unsqueeze(1)) # unsqueeze(1)将[B, H, W] -> [B, 1, H, W]，单通道 mask 与 RGB 图像拼接
            
            # 将模板特征降维压缩成 prompt key(注意力计算中的K)
            k16_template = self.key_proj(template_feature)

            # 提取当前帧（search image）在 ViT 输出中的部分作为搜索特征
            # 取出搜索 token，view 成 [B, C, Hs/16, Ws/16]  （默认24×24） [B,768,24,24]
            searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            search_app_feat = self._get_app_feature(searchRegionFeature_1)
            # 同样投影到低维空间生成 当前搜索帧的query key(注意力计算中的Q) [B, 64, Hs/16, Ws/16]  [B,64,24,24]
            k16 = self.key_proj(searchRegionFeature_1)         

            # 当前搜索帧经过压缩后的特征(query value)，用于与注意力计算后的结果进行拼接，形成最终的历史提示 [B, 768, Hs/16, Ws/16] -> [B, 384, Hs/16, Ws/16]  [B,384,24,24]
            # 它是query的自身特征，不参与注意力计算，用于和记忆信息融合
            searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)

            # 将模板记忆作为提示，解码出当前帧的历史提示特征
            historical_prompt = self.HIP('train_decode', 
                            k16, # queryFrame_key
                            searchRegionFeature_1_thin, # queryFrame value
                            k16_template.unsqueeze(2), # memoryKey
                            ref_v_template) # memoryValue

            B, C, H, W = historical_prompt.shape  #[B,768,24,24]

            # 3.预测头
            # 变成匹配检测头的形状 (B, HW, C)  [B,576,768]
            historical_prompt = historical_prompt.view(B, C, H*W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            # 将原始搜索特征和历史提示堆叠后送入检测头
            out = self.forward_head(
                torch.stack([feat_last[:, -self.feat_len_s:], historical_prompt], dim=0), 
                None, return_topk_boxes=False)
            
            out.update(aux_dict)
            out['backbone_feat'] = x

            if index == 5 or index == 10:
                force_write = self._memory_is_empty()
                if self._memory_should_write(out['score_map'], out['pred_boxes'], motion_bbox, search_app_feat, force=force_write):
                    #print(self.memorys)
                    B, _, Ht, Wt = template.shape
                    _, _, C = x.shape
                    _, _, Hs, Ws = search.shape

                    # 对预测结果生成二值掩码，用于抽取当前帧的记忆 value
                    mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search, x, visualizeMask=False, frame=index, seqName=info)

                    ref_v = self.HIP('encode', 
                                        search, 
                                        searchRegionFeature_1, 
                                        mask.unsqueeze(1))

                    # 在第 5/10 帧写入记忆，为后续帧提供历史提示
                    # searchRegionImg用于可视化记忆库，把搜索图像和对应的 key/value 一起保存下来
                    self.HIP.addMemory(k16, ref_v, searchRegionImg)
            return out

        else:
            # 进入常规跟踪阶段：记忆池已建立，直接利用历史提示
            #flops1, params1 = thop.profile(self.backbone, inputs=(template, search, ce_template_mask, ce_keep_rate, None, False, None, None))
            
            x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate)

            B, _, Ht, Wt = template.shape
            _, _, C = x.shape
            _, _, Hs, Ws = search.shape
            
            # 1.从 ViT 输出中取出当前搜索帧 token，并投影成 query key（Q）
            k16 = self.key_proj(x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))
            #flops2, params2 = thop.profile(self.key_proj, inputs=(x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16),))
            

            # 2.还原搜索帧特征为 [B, C, Hs/16, Ws/16]，并压缩通道作为 query value
            searchRegionFeature = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
            search_app_feat = self._get_app_feature(searchRegionFeature)
            searchRegionFeature_thin = self.key_comp(searchRegionFeature)
            #flops3, params3 = thop.profile(self.key_comp, inputs=(searchRegionFeature,))

            # 3.调用解码器 eval_decode：使用记忆库中的 prompt key/value(KV) 与当前 Q 做匹配
            historicalPrompt = self.HIP('eval_decode', 
                            k16, # queryFrame_key
                            searchRegionFeature_thin, #queryFrame_value
                        )
            # eval_decode：只读历史记忆，输出融合后的动态提示
            
            B, C, H, W = historicalPrompt.shape

            # 4.预测头
            historicalPrompt = historicalPrompt.view(B, C, H*W).permute(0, 2, 1)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]

            out = self.forward_head(
                torch.stack([feat_last[:, -self.feat_len_s:], historicalPrompt], dim=0), 
                None, return_topk_boxes=False) 

            out.update(aux_dict)
            out['backbone_feat'] = x

            if index % self.update_interval == 0:
                force_write = self._memory_is_empty()
                if self._memory_should_write(out['score_map'], out['pred_boxes'], motion_bbox, search_app_feat, force=force_write):
                    # 每隔 update_interval 帧更新一次记忆，避免过频导致噪声积累
                    mask = self.generateMask(aux_dict['removed_indexes_s'], 
                                             out['pred_boxes'].squeeze(1), 
                                             search, x, visualizeMask=False, frame=index, seqName=info)

                    # 用最新帧的预测框生成掩码，再编码成新的记忆 value/key
                    ref_v = self.HIP('encode', 
                                        search, 
                                        searchRegionFeature, 
                                       mask.unsqueeze(1)) 
                    
                    # 将当前帧的 key/value 写入记忆库，同时可选保存裁剪图像用于可视化
                    self.HIP.addMemory(k16, ref_v, searchRegionImg)
            
            return out

    def forward(self, template: torch.Tensor,
                search: list,
                search_after: torch.Tensor=None,
                previous: torch.Tensor=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gtBoxes=None,
                previousBoxes=None,
                template_boxes=None
                ):
        """
        参数说明：
            template : 模板图像张量，形状 [B, 3, H_z, W_z]。
            search : 搜索帧列表，默认包含 5 帧，每帧形状 [B, 3, H_x, W_x]。
            search_after : 备用的后续搜索帧（当前未使用，兼容旧接口）。
            previous : 历史帧堆叠张量，用于多帧建模（默认 None）。
            ce_template_mask : 候选消融生成的模板 mask，指导 backbone 丢弃 token。
            ce_keep_rate : 候选消融保留比例，配合 mask 使用。
            return_last_attn : 是否返回 backbone 最后一层注意力权重，用于可视化调试。
            gtBoxes : 搜索帧对应的 GT 框（训练监督用，当前未直接使用）。
            previousBoxes : 历史帧的 GT 框，用于多帧设置下的辅助监督。
            template_boxes : 模板帧的 GT 框（cxcywh），用于生成模板掩码。
        """
        '''
            template : [B 3 H_z W_z]
            search : [3 * [B 3 H_x W_x]]
            previous : [B L 3 H_x W_x]
        '''
        # 训练阶段入口：接受一个模板和多个搜索帧（默认 5 帧）
        x, aux_dict = self.backbone(z=template, x=search[0],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        B, _, Ht, Wt = template.shape
        _, _, C = x.shape
        _, _, Hs, Ws = search[0].shape

        # ===== 第一帧：使用模板构建初始记忆 =====
        upsampled_template = self.upsample(template)
        template_mask = self.generateMask([None, None, None], 
                                          template_boxes.squeeze(0), 
                                          upsampled_template, x, 
                                          visualizeMask=False, cxcywh=False)
        template_feature = x[:, :(Ht // 16)**2, :].permute(0, 2, 1).view(B, C, Ht // 16, Wt // 16)
        template_feature = self.upsample(template_feature)
        ref_v_template = self.HIP('encode', 
                        upsampled_template, 
                        template_feature, 
                        template_mask.unsqueeze(1)) 
        
        k16_template = self.key_proj(template_feature)
        searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        k16 = self.key_proj(searchRegionFeature_1)         
        searchRegionFeature_1_thin = self.key_comp(searchRegionFeature_1)
        historical_prompt = self.HIP('train_decode', 
                        k16, # queryFrame_key
                        searchRegionFeature_1_thin, # queryFrame value
                        k16_template.unsqueeze(2), # memoryKey
                        ref_v_template) # memoryValue
        B, C, H, W = historical_prompt.shape

        historical_prompt = historical_prompt.view(B, C, H*W).permute(0, 2, 1)
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(
            torch.stack([feat_last[:, -self.feat_len_s:], historical_prompt], dim=0), 
            None, return_topk_boxes=False) 
        out.update(aux_dict)
        out['backbone_feat'] = x

        mask = self.generateMask(aux_dict['removed_indexes_s'], out['pred_boxes'].squeeze(1), search[0], x, visualizeMask=False)
        searchRegionFeature_1 = x[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)

        
        ref_v = self.HIP('encode', 
                            search[0],
                            searchRegionFeature_1, 
                            mask.unsqueeze(1))

        k16 = self.key_proj(searchRegionFeature_1)
        #k16 = k16.reshape(B, *k16.shape[-3:]).transpose(1, 2).contiguous()

        x_2, aux_dict_2 = self.backbone(z=template, x=search[1],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        # ===== 第二帧：使用第一帧写入的记忆作为提示 =====
        k16_2 = self.key_proj(x_2[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_2 = x_2[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_2_thin = self.key_comp(searchRegionFeature_2)

        historicalPrompt_2 = self.HIP('train_decode', 
                            k16_2, # queryFrame_key
                            searchRegionFeature_2_thin, # queryFrame value
                            k16.unsqueeze(2), # memoryKey
                            ref_v) # memoryValue
        
        B, C, H, W = historicalPrompt_2.shape

        historicalPrompt_2 = historicalPrompt_2.view(B, C, H*W).permute(0, 2, 1)
        
        feat_x2_last = x_2
        if isinstance(x_2, list):
            feat_x2_last = x_2[-1]

        out_2 = self.forward_head(
            torch.stack([feat_x2_last[:, -self.feat_len_s:], historicalPrompt_2], dim=0), 
            None, return_topk_boxes=False)
        
        out_2.update(aux_dict_2)
        out_2['backbone_feat'] = x_2

        mask_2 = self.generateMask(aux_dict_2['removed_indexes_s'], 
                                 out_2['pred_boxes'].squeeze(1), 
                                 search[1], x_2, visualizeMask=False)

        ref_v_2 = self.HIP('encode', 
                            search[1], 
                            searchRegionFeature_2, 
                            mask_2.unsqueeze(1)) 
        
        x_3, aux_dict_3 = self.backbone(z=template, x=search[2],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        # ===== 第三帧：记忆由前两帧叠加 =====
        k16_3 = self.key_proj(x_3[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_3 = x_3[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_3_thin = self.key_comp(searchRegionFeature_3)

        historicalPrompt_3 = self.HIP('train_decode', 
                            k16_3, # queryFrame_key
                            searchRegionFeature_3_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2], dim=2)) # memoryValue

        historicalPrompt_3 = historicalPrompt_3.view(B, C, H*W).permute(0, 2, 1)

        feat_x3_last = x_3
        if isinstance(x_3, list):
            feat_x3_last = x_3[-1]

        out_3 = self.forward_head(
            torch.stack([feat_x3_last[:, -self.feat_len_s:], historicalPrompt_3], dim=0), 
            None, return_topk_boxes=False) 


        out_3.update(aux_dict_3)
        out_3['backbone_feat'] = x_3

        mask_3 = self.generateMask(aux_dict_3['removed_indexes_s'], 
                                 out_3['pred_boxes'].squeeze(1), 
                                 search[2], x_3, visualizeMask=False)

        ref_v_3 = self.HIP('encode', 
                            search[2], 
                            searchRegionFeature_3, 
                            mask_3.unsqueeze(1)) 

        x_4, aux_dict_4 = self.backbone(z=template, x=search[3],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        # ===== 第四帧：记忆累积前三帧 =====
        k16_4 = self.key_proj(x_4[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_4 = x_4[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_4_thin = self.key_comp(searchRegionFeature_4)

        historicalPrompt_4 = self.HIP('train_decode', 
                            k16_4, # queryFrame_key
                            searchRegionFeature_4_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2), k16_3.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2, ref_v_3], dim=2)) # memoryValue

        historicalPrompt_4 = historicalPrompt_4.view(B, C, H*W).permute(0, 2, 1)

        feat_x4_last = x_4
        if isinstance(x_4, list):
            feat_x4_last = x_4[-1]

        out_4 = self.forward_head(
            torch.stack([feat_x4_last[:, -self.feat_len_s:], historicalPrompt_4], dim=0), 
            None, return_topk_boxes=False) 

        out_4.update(aux_dict_4)
        out_4['backbone_feat'] = x_4

        mask_4 = self.generateMask(aux_dict_4['removed_indexes_s'], 
                                 out_4['pred_boxes'].squeeze(1), 
                                 search[3], x_4, visualizeMask=False)

        ref_v_4 = self.HIP('encode', 
                            search[3], 
                            searchRegionFeature_4, 
                            mask_4.unsqueeze(1))

        x_5, aux_dict_5 = self.backbone(z=template, x=search[4],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, previous_frames=previous, previous_anno=previousBoxes)
        
        # ===== 第五帧：记忆累积前四帧 =====
        k16_5 = self.key_proj(x_5[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16))

        searchRegionFeature_5 = x_5[:, (Ht // 16)**2:, :].permute(0, 2, 1).view(B, C, Hs // 16, Ws // 16)
        searchRegionFeature_5_thin = self.key_comp(searchRegionFeature_5)

        historicalPrompt_5 = self.HIP('train_decode', 
                            k16_5, # queryFrame_key
                            searchRegionFeature_5_thin, # queryFrame value
                            torch.cat([k16.unsqueeze(2), k16_2.unsqueeze(2), k16_3.unsqueeze(2), k16_4.unsqueeze(2)], dim=2), # memoryKey
                            torch.cat([ref_v, ref_v_2, ref_v_3, ref_v_4], dim=2)) # memoryValue

        historicalPrompt_5 = historicalPrompt_5.view(B, C, H*W).permute(0, 2, 1)

        feat_x5_last = x_5
        if isinstance(x_5, list):
            feat_x5_last = x_5[-1]

        out_5 = self.forward_head(
            torch.stack([feat_x5_last[:, -self.feat_len_s:], historicalPrompt_5], dim=0), 
            None, return_topk_boxes=False) 

        out_5.update(aux_dict_5)
        out_5['backbone_feat'] = x_5

        return [out, out_2, out_3, out_4, out_5]

    def _get_app_feature(self, feat: torch.Tensor):
        if feat is None or not torch.is_tensor(feat):
            return None
        if feat.dim() != 4:
            return None
        vec = feat.flatten(2).mean(dim=2)
        vec = F.normalize(vec, dim=1)
        return vec

    def _memory_should_write(self, score_map, pred_boxes, motion_bbox, search_app_feat, force=False):
        if force:
            self.memory_skip_count = 0
            return True
        if score_map is None or pred_boxes is None:
            self.memory_skip_count = 0
            return True
        allow_write = True
        score_val = torch.max(score_map).item()
        if self.memory_score_thresh > 0 and score_val < self.memory_score_thresh:
            allow_write = False
        if allow_write and self.memory_motion_thresh > 0 and motion_bbox is not None:
            motion_iou = self._compute_motion_iou(pred_boxes, motion_bbox)
            if motion_iou is not None and motion_iou < self.memory_motion_thresh:
                allow_write = False
        if allow_write and self._has_competing_peak(score_map):
            allow_write = False
        if allow_write:
            self.memory_skip_count = 0
            return True
        if self.memory_force_interval and self.memory_force_interval > 0:
            self.memory_skip_count += 1
            if self.memory_skip_count >= self.memory_force_interval:
                self.memory_skip_count = 0
                return True
        return False

    def _has_competing_peak(self, score_map: torch.Tensor):
        if score_map is None:
            return False
        if score_map.dim() == 4:
            score = score_map[:, 0]
        elif score_map.dim() == 3:
            score = score_map
        else:
            return False

        pooled = F.max_pool2d(score.unsqueeze(1), kernel_size=5, stride=1, padding=2)
        is_peak = (score.unsqueeze(1) == pooled).squeeze(1)

        for b in range(score.shape[0]):
            coords = torch.nonzero(is_peak[b], as_tuple=False)
            if coords.numel() == 0:
                continue
            vals = score[b, coords[:, 0], coords[:, 1]]
            if vals.numel() < 2:
                continue
            top1_val, top1_idx = torch.max(vals, dim=0)
            top1_coord = coords[top1_idx].float()
            dists = torch.norm(coords.float() - top1_coord, p=2, dim=1)
            valid = dists >= self.second_peak_min_dist
            if not valid.any():
                continue
            top2_val = vals[valid].max()
            if (top2_val >= self.second_peak_ratio * top1_val and
                    top2_val >= self.second_peak_min_score):
                return True
        return False

    def _compute_motion_iou(self, pred_boxes, motion_bbox):
        if motion_bbox is None:
            return None
        if isinstance(motion_bbox, torch.Tensor):
            motion_tensor = motion_bbox
        else:
            motion_tensor = torch.tensor(motion_bbox, dtype=pred_boxes.dtype, device=pred_boxes.device)
        if motion_tensor.dim() == 1:
            motion_tensor = motion_tensor.unsqueeze(0)
        pred = pred_boxes[:, 0, :]
        pred_xyxy = box_cxcywh_to_xyxy(pred)
        motion_xyxy = box_cxcywh_to_xyxy(motion_tensor)
        iou = ops.box_iou(pred_xyxy, motion_xyxy).mean().item()
        return iou

    def _memory_is_empty(self):
        decoder = getattr(self.HIP, "decoder", None)
        if decoder is None:
            return True
        return getattr(decoder, "mem_k", None) is None

    def deNorm(self, image):
        # 将归一化图像还原到 0-255，方便调试与可视化
        img = image.cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img[0] = (img[0] * std[0] + mean[0]) * 255
        img[1] = (img[1] * std[1] + mean[1]) * 255
        img[2] = (img[2] * std[2] + mean[2]) * 255
        img = img.transpose(1, 2, 0).astype(np.uint8).copy()
        return img

    def generateMask(self, ceMasks, predBoxes, img_normed, img_feat, visualizeMask=False, cxcywh=True, frame=None, seqName=None):
        B, _, H_origin, W_origin = img_normed.shape
        masks = torch.zeros((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        pure_ce_masks = torch.ones((B, H_origin, W_origin), device=img_feat.device, dtype=torch.uint8)
        for i in range(B):
            if cxcywh:
                # predBoxes 默认为中心点 + 宽高，需要转换成左上-右下坐标
                box = (box_cxcywh_to_xyxy((predBoxes[i])) * H_origin).int()
            else:
                box = (predBoxes[i] * H_origin).int()
                box[2] += box[0]
                box[3] += box[1]

            box[0] = 0 if box[0] < 0 else box[0]
            box[1] = H_origin if box[1] > H_origin else box[1]
            box[2] = W_origin if box[2] > W_origin else box[2]
            box[3] = 0 if box[3] < 0 else box[3]
            
            if visualizeMask:
                if not os.path.exists(f"./masks_vis/{seqName}/{frame}"):
                    os.makedirs(f"./masks_vis/{seqName}/{frame}")
                img = self.deNorm(img_normed[i])
            #masks[i] = torch.zeros((H_origin, W_origin), dtype=np.uint8)
            masks[i][box[1].item():box[3].item(), box[0].item():box[2].item()] = 1
            if ceMasks[0] is not None and ceMasks[1] is not None and ceMasks[2] is not None:
                ce1 = ceMasks[0][i]
                ce2 = ceMasks[1][i]
                ce3 = ceMasks[2][i]
                ce = torch.cat([ce1, ce2, ce3], axis=0)
                for num in ce:
                    x = int(num) // 24
                    y = int(num) % 24
                    masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0
                    pure_ce_masks[i][x*16 : (x+1)*16, y*16 : (y+1)*16] = 0
            
            if visualizeMask:
                mask = masks[i].cpu().detach().numpy().astype(np.uint8)
                mask = np.stack([mask, mask, mask], axis=2) * 255
                pure_ce_mask = pure_ce_masks[i].cpu().detach().numpy().astype(np.uint8)
                pure_ce_mask = np.stack([pure_ce_mask, pure_ce_mask, pure_ce_mask], axis=2) * 255
                #cv2.rectangle(mask, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/mask.jpg", mask[:,:,::-1])
                #cv2.imwrite(f"./masks_vis/{seqName}/{frame}/ce_mask.jpg", pure_ce_mask[:,:,::-1])
                #import pdb; pdb.set_trace()
                img2 = img.copy()
                img3 = img.copy()
                img2[mask == 0] = 255
                img3[pure_ce_mask == 0] = 255
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_with_mask.jpg", img2[:,:,::-1])
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img_CE_mask.jpg", img3[:,:,::-1])
                cv2.rectangle(img, (box[0].item(), box[1].item()), (box[2].item(), box[3].item()), (0, 0, 255), 2)
                cv2.imwrite(f"./masks_vis/{seqName}/{frame}/img.jpg", img[:,:,::-1])
        return masks

    def forward_head(self, cat_feature, gt_score_map=None, return_topk_boxes=False):
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # 将 backbone 输出以及历史提示拼接，输入检测头预测得分与边界框
        _, B, HW, C = cat_feature.shape
        H = int(HW ** 0.5)
        W = H
        originSearch = cat_feature[0].view(B, H, W, C).permute(0, 3, 1, 2)
        dynamicSearch = cat_feature[1].view(B, H, W, C).permute(0, 3, 1, 2)
        enc_opt = self.searchRegionFusion(originSearch + dynamicSearch).view(B, C, HW).permute(0, 2, 1)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map, topkBbox = self.box_head(opt_feat, gt_score_map, return_topk_boxes)
            outputs_coord = bbox 
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            if return_topk_boxes:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                       'topk_pred_boxes': topkBbox,
                    }
            else:
                out = {'pred_boxes': outputs_coord_new,
                       'score_map': score_map_ctr,
                       'size_map': size_map,
                       'offset_map': offset_map,
                    }
            return out
        else:
            raise NotImplementedError


def build_hiptrack(cfg, training=True):
    # 构建 HIPTrack 主干：选择 ViT 版本，加载预训练权重，绑定检测头
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('HIPTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           enable_ce=cfg.MODEL.BACKBONE.ENABLE_CE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            enable_ce=cfg.MODEL.BACKBONE.ENABLE_CE
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = HIPTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        new_hip=cfg.MODEL.NEW_HIP,
        memory_max=cfg.MODEL.MAX_MEM,
        update_interval=cfg.TEST.UPDATE_INTERVAL
    )
    model.memory_score_thresh = getattr(cfg.TEST, "MEMORY_SCORE_THRESH", 0.0)
    model.memory_motion_thresh = getattr(cfg.TEST, "MEMORY_MOTION_IOU", 0.0)
    model.second_peak_ratio = getattr(cfg.TEST, "SECOND_PEAK_RATIO", 0.85)
    model.second_peak_min_score = getattr(cfg.TEST, "SECOND_PEAK_MIN_SCORE", 0.25)
    model.second_peak_min_dist = getattr(cfg.TEST, "SECOND_PEAK_MIN_DIST", 2.0)
    model.memory_force_interval = getattr(cfg.TEST, "MEMORY_FORCE_INTERVAL", 3)
    if ('HIPTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models', cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        # 仅在训练阶段载入 DropTrack/HIPTrack 预训练权重，以加速收敛
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
