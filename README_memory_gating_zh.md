# HIPTrack 记忆写入门控说明（中文）

本文说明 HIPTrack 推理阶段的记忆写入门控（memory write gating）。写入并非每帧发生，而是满足触发时机后，再经过门控条件判定。

## 触发时机
- 初始化阶段：第 5 帧和第 10 帧尝试写入（index==5 或 10）。
- 稳定阶段：之后每隔 UPDATE_INTERVAL 帧尝试写入。

## 三个门控条件（同时满足才写入）
1. 置信度门控（Score Threshold）
   - S_max = max(score_map)
   - 若 MEMORY_SCORE_THRESH > 0 且 S_max < MEMORY_SCORE_THRESH → 不写入。

2. 运动一致性门控（Motion Consistency）
   - 当提供 motion_bbox 且 MEMORY_MOTION_IOU > 0 时：
   - 计算 IoU(pred_bbox, motion_bbox)。
   - 若 IoU < MEMORY_MOTION_IOU → 不写入。

3. 多峰干扰门控（Competing Peak）
   - 对 score_map 做 5x5 MaxPool（stride=1, padding=2），筛出局部极大值。
   - 取主峰 S_max 与满足最小空间距离 SECOND_PEAK_MIN_DIST 的次峰 S_t。
   - 若 S_t >= SECOND_PEAK_RATIO * S_max 且 S_t >= SECOND_PEAK_MIN_SCORE → 不写入。
   - 该条件用于抑制多目标/干扰峰造成的错误记忆写入。

## 兜底：强制写入
- 记录连续被门控拦截的次数 memory_skip_count。
- 若连续拦截达到 MEMORY_FORCE_INTERVAL，则下一次强制写入。
- 当记忆为空时也会强制写入。

## 参数与默认值
- TEST.MEMORY_SCORE_THRESH = 0.5
- TEST.MEMORY_MOTION_IOU = 0.3
- TEST.SECOND_PEAK_RATIO = 0.85
- TEST.SECOND_PEAK_MIN_SCORE = 0.25
- TEST.SECOND_PEAK_MIN_DIST = 2.0
- TEST.MEMORY_FORCE_INTERVAL = 3
- TEST.UPDATE_INTERVAL = 20

## 调参建议
- 目标易被干扰：提高 SECOND_PEAK_RATIO 或 SECOND_PEAK_MIN_SCORE。
- 目标易遮挡：降低 MEMORY_SCORE_THRESH 或增大 MEMORY_FORCE_INTERVAL。
- 运动模型稳定：适当提高 MEMORY_MOTION_IOU。
- SECOND_PEAK_MIN_DIST 为特征图单位（score_map 分辨率），2.0 约等于 2 个 cell。

## 修改文件
- lib/models/hiptrack/hiptrack.py
- lib/config/hiptrack/config.py
