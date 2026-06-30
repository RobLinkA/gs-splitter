# GS-Splitter

> 3D Gaussian Splatting PLY 大半径异常高斯点后处理拆分工具。  
> Post-process 3DGS PLY files by splitting abnormal large-radius Gaussians into smaller, density-matched Gaussians.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-green)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

---

## 1. 项目简介

**GS-Splitter** 是一个面向 **3D Gaussian Splatting（3DGS）PLY 成品模型** 的后处理工具。

它不依赖原始照片、不依赖相机参数、不需要重新训练模型，只读取最终导出的 `.ply` 文件，自动检测模型中的 **大半径异常高斯点**，并将这些异常点自适应拆分为多个更小、更致密的高斯点，从而改善：

- 街区、建筑群、道路边缘裁切后的毛刺和外凸小球；
- 文物、玻璃、反光、透光区域中的大半径填洞瑕疵；
- 局部区域“粗糙发虚”、其他区域“细腻平整”的质感割裂；
- 存量 3DGS 模型无法重新训练、但需要统一规格和展示质量的问题。

简单来说：

```text
把 3DGS PLY 里异常大的高斯点，拆成多个更小、更均匀、更适合裁切和展示的小高斯点。
```

---

## 2. 为什么需要这个工具

在 3DGS 实际落地中，即使训练时设置了半径约束、密度控制或剪枝策略，最终导出的 PLY 模型仍然可能存在一些异常的大半径高斯点。

这些点在训练视角下可能可以“遮住洞”或“补齐颜色”，但在实际展示、裁切、近距离查看、文物复刻或下游加工时，会表现为明显问题。

### 2.1 街区 / 建筑群场景

常见问题包括：

- 道路边缘裁切后出现半圆形外凸点；
- 建筑轮廓边缘发毛、不锐利；
- 场景边界切完后出现一圈“毛边”；
- 大半径高斯横跨裁切线，导致裁切线不听话。

### 2.2 文物 / 小物体高精度场景

常见问题包括：

- 玻璃展柜反光造成局部半透明；
- 局部过曝、欠曝导致表面空洞；
- 训练结果用一个或少数几个大高斯点填洞；
- 高质量区域很细腻，瑕疵区域却粗糙、发虚、像糊了一层。

### 2.3 存量模型批量修复

很多模型已经训练完成，甚至已经进入交付、展示或加工环节。此时重新拍摄、重新训练成本很高。

GS-Splitter 的目标是提供一种：

- 后置；
- 通用；
- 可批量；
- 不重训；
- 不依赖原始工程；
- 可通过参数控制效果和体积；

的 PLY 优化方案。

---

## 3. 解决方案思路

3DGS 的 PLY 文件通常会在 `vertex` 元素中保存每个高斯点的独立属性，例如：

- `x`, `y`, `z`：空间位置；
- `scale_0`, `scale_1`, `scale_2`：高斯三轴尺度，通常为 log scale；
- `rot_0`, `rot_1`, `rot_2`, `rot_3`：旋转四元数；
- `opacity`：不透明度，通常为 logit opacity；
- `f_dc_*`, `f_rest_*`：颜色和球谐系数；
- `normal_*` 或其他扩展字段。

GS-Splitter 的核心逻辑是：

```text
读取 PLY
  ↓
还原每个高斯点的真实半径
  ↓
通过全局分位数 + MAD 鲁棒统计 + KNN 局部邻域判断异常大点
  ↓
根据局部正常点云密度和半径，计算每个异常点应拆成多少个子点
  ↓
在母高斯椭球范围内生成子高斯
  ↓
缩小子高斯 scale，继承颜色 / SH / 旋转 / 其他属性
  ↓
按 alpha 合成关系近似守恒 opacity
  ↓
删除原异常大点，写出新的 PLY
```

输出文件默认命名为：

```text
原文件名_split.ply
```

---

## 4. 项目适合什么，不适合什么

### 4.1 适合

- 已经导出的 3DGS PLY 成品模型；
- 街区、建筑群、道路、场景边界裁切；
- 文物、小物体、高精度展示模型；
- 玻璃、反光、透光、局部空洞造成的大半径填洞；
- 不方便重新训练的存量模型；
- 需要统一点云颗粒度和视觉质感的批量项目。

### 4.2 不适合

- 非 3DGS 结构的普通点云 PLY；
- 缺少 `scale_0/1/2` 或 `rot_0/1/2/3` 字段的模型；
- 想从根本上恢复拍摄缺失区域的真实细节；
- 想重新学习光照、反射、视角相关颜色的情况；
- 已经严重训练失败、几何结构整体错误的模型。

### 4.3 重要说明

本工具是 **视觉修复和结构优化工具**，不是重新训练工具。

它会尽量保留原始颜色、透明度、旋转、球谐和其他字段，但因为它改变了高斯数量、位置和尺度，所以不应宣传为严格数学意义上的“完全无损”。更准确的说法是：

```text
几何与外观属性尽量继承，透明度近似守恒，视觉效果尽量无损。
```

---

## 5. 安装

### 5.1 环境要求

建议使用：

```text
Python 3.9+
```

核心依赖：

```bash
pip install numpy scipy plyfile
```

如果需要 GUI 拖拽功能：

```bash
pip install tkinterdnd2
```

### 5.2 下载项目

```bash
git clone https://github.com/RobLinkA/gs-splitter.git
cd gs-splitter
```

### 5.3 推荐的项目结构

```text
GS-Splitter/
├── README.md
├── LICENSE
└── gs_splitter.py
```

---

## 6. 快速开始

### 6.1 GUI 模式

直接运行：

```bash
python gs_splitter.py
```

然后：

1. 选择或拖入 `.ply` 文件；
2. 选择参数预设：`balanced`、`artifact` 或 `city`；
3. 点击开始处理；
4. 等待输出 `_split.ply` 文件。

### 6.2 CLI 模式

通用模型：

```bash
python gs_splitter.py input.ply --profile balanced
```

街区 / 建筑群：

```bash
python gs_splitter.py input.ply --profile city
```

文物 / 小物体 / 透光瑕疵：

```bash
python gs_splitter.py input.ply --profile artifact
```

覆盖单个参数：

```bash
python gs_splitter.py input.ply --profile city --set max_split=64 --set global_percentile=97.8
```

覆盖多个参数：

```bash
python gs_splitter.py input.ply \
  --profile artifact \
  --set local_ratio=1.75 \
  --set max_split=128 \
  --set max_output_multiplier=2.2
```

---

## 7. 参数预设怎么选

| 预设 | 推荐场景 | 特点 |
|---|---|---|
| `balanced` | 通用存量模型 | 效果和文件体积比较均衡 |
| `city` | 街区、建筑群、道路边界裁切 | 更保守，重点去除明显大球和边缘毛刺，控制体积 |
| `artifact` | 文物、小物体、玻璃、反光、透光区域 | 更积极，允许更多拆分，提高局部细腻度 |

推荐顺序：

```text
不确定 → balanced
街区裁切 → city
文物细节 → artifact
文件太大 → 降低 max_split 或 max_output_multiplier
效果不够 → 降低 global_percentile / local_ratio，或提高 max_split
```

---

## 8. 分场景使用流程

下面所有流程都使用统一格式：

```text
第一步：准备或粗处理
第二步：脚本裂解 / 优化
第三步：裁切 / 定型 / 输出
第四步：局部检查 / 可选细化
```

---

### 8.1 街区 / 建筑群 / 道路边界裁切

这是最推荐使用 GS-Splitter 的场景之一。

对于街区裁切，推荐主流程是：

```text
1. 粗剪 Rough Crop
2. 脚本裂解 Split / Optimize
3. 精细裁切 Fine Trim
4. 局部检查 / 细化 Optional Refine
```

#### 第一步：粗剪 Rough Crop

**目的：减负。**

操作：

- 在 CloudCompare、SuperSplat、Postshot、渲染器或自研工具中，先把场景外围完全不相关的远景切掉；
- 删除远处天空、杂乱背景、无关地面、明显不需要的建筑外圈；
- 不需要切得非常准，只要把明显不会进入最终成果的部分删掉即可。

意义：

- 减少 PLY 点数；
- 降低 KDTree 邻域搜索内存；
- 避免脚本把最终会被删除的远景区域也拆分；
- 提高后续处理速度。

建议：

```text
粗剪边界要比最终边界大一圈，给第二步裂解和第三步精裁留出缓冲区。
```

#### 第二步：脚本裂解 Split / Optimize

**目的：提升空间分辨率。**

操作：

```bash
python gs_splitter.py rough_crop.ply --profile city
```

效果：

- 脚本会检测粗剪后模型中的异常大半径高斯点；
- 包括未来精细裁切线附近的大高斯；
- 将这些大高斯拆成更小的子高斯；
- 模型会变成一块颗粒度更小、边界更容易切齐的“材料”。

推荐参数：

```bash
python gs_splitter.py rough_crop.ply \
  --profile city \
  --set max_split=32 \
  --set max_output_multiplier=1.45
```

如果边界仍然发毛，可以更积极一些：

```bash
python gs_splitter.py rough_crop.ply \
  --profile city \
  --set global_percentile=97.5 \
  --set local_ratio=2.0 \
  --set max_split=64 \
  --set max_output_multiplier=1.7
```

注意：

- 街区模型通常点数大，不建议一上来使用 `artifact`；
- 不建议把 `max_output_multiplier` 设得过高；
- 先用一小块区域试参数，再跑完整街区。

#### 第三步：精细裁切 Fine Trim

**目的：定型。**

操作：

- 使用第二步输出的 `_split.ply`；
- 沿道路牙子、建筑围墙、项目红线、地块边界进行精细裁切；
- 这一步决定最终交付轮廓。

视觉效果：

- 因为边界附近的大高斯已经被拆成更小的子点，裁切线会更“听话”；
- 切口不容易出现外凸半圆球；
- 道路和建筑轮廓会更锐利；
- 场景边缘整体更规整。

#### 第四步：局部检查 / 细化 Optional Refine

**目的：修补极端瑕疵。**

操作：

- 检查道路边缘、建筑外轮廓、地块红线、立面边缘；
- 如果仍有少量顽固大点，可以对精裁后的 PLY 再跑一次更局部、更轻量的处理。

保守二次处理：

```bash
python gs_splitter.py fine_trim.ply \
  --profile city \
  --set global_percentile=98.0 \
  --set local_ratio=2.2 \
  --set max_split=24 \
  --set max_output_multiplier=1.25
```

更积极二次处理：

```bash
python gs_splitter.py fine_trim.ply \
  --profile city \
  --set global_percentile=96.8 \
  --set local_ratio=1.8 \
  --set max_split=48 \
  --set max_output_multiplier=1.5
```

判断标准：

```text
如果主要问题是最终边界不锐利：优先使用“粗剪 → 裂解 → 精裁”。
如果模型已经精裁完成，只是边缘还有毛刺：直接对精裁结果跑 city 二次处理。
```

---

### 8.2 文物 / 小物体 / 高精度数字化

文物场景的主要问题通常不是边界裁切，而是局部反光、透光、过曝、欠曝导致的表面空洞或大半径填洞。

推荐流程：

```text
1. 基础清理 Base Clean
2. 脚本裂解 Split / Optimize
3. 表面检查 Surface Review
4. 局部细化 Optional Refine
```

#### 第一步：基础清理 Base Clean

**目的：去掉明显无关区域。**

操作：

- 去掉展柜外部、地面、背景墙、支架、扫描台等无关点；
- 保留文物主体周围少量缓冲区域；
- 不要过早裁掉半透明区域，因为这些区域可能正是需要修复的对象。

意义：

- 减少无关区域对局部密度判断的影响；
- 降低点数和处理耗时；
- 保留需要修复的瑕疵区域。

#### 第二步：脚本裂解 Split / Optimize

**目的：修复大半径填洞和局部粗糙质感。**

推荐使用：

```bash
python gs_splitter.py artifact_input.ply --profile artifact
```

文物高精度参数示例：

```bash
python gs_splitter.py artifact_input.ply \
  --profile artifact \
  --set max_split=128 \
  --set max_output_multiplier=2.3
```

如果文件体积压力较大：

```bash
python gs_splitter.py artifact_input.ply \
  --profile artifact \
  --set max_split=64 \
  --set max_output_multiplier=1.8
```

效果：

- 将局部大半径填洞点拆成多个小点；
- 让透光区域和正常区域颗粒度更接近；
- 降低“局部糊成一片”的观感；
- 提升文物表面细腻统一程度。

#### 第三步：表面检查 Surface Review

**目的：确认修复是否自然。**

重点检查：

- 玻璃反光区域；
- 金属高光区域；
- 釉面、瓷器、玉石等半反射材质；
- 局部过曝或欠曝形成的空洞；
- 高精度展示时会被近距离观察的区域。

判断标准：

```text
修复后的区域应该更细腻，但不能明显变厚、变亮或发灰。
```

如果出现变厚、变亮：

```bash
python gs_splitter.py artifact_input.ply \
  --profile artifact \
  --set opacity_gain=0.95 \
  --set max_split=64
```

#### 第四步：局部细化 Optional Refine

**目的：进一步修补顽固区域。**

如果还有明显大点：

```bash
python gs_splitter.py artifact_split.ply \
  --profile artifact \
  --set global_percentile=95.5 \
  --set local_ratio=1.6 \
  --set max_split=96 \
  --set max_output_multiplier=1.6
```

如果出现轻微噪点：

```bash
python gs_splitter.py artifact_split.ply \
  --profile balanced \
  --set global_percentile=98.0 \
  --set local_ratio=2.3 \
  --set max_split=32
```

注意：

```text
文物场景不建议盲目追求极低阈值。过度拆分可能让局部变成细碎噪点。
```

---

### 8.3 已经完成精细裁切的模型

如果你的模型已经裁切完成，只是发现边缘有毛刺或局部有大球，可以直接对成品 PLY 做后处理。

推荐流程：

```text
1. 备份 Backup
2. 脚本裂解 Split / Optimize
3. 成品复查 Final Review
4. 局部二次处理 Optional Refine
```

#### 第一步：备份 Backup

**目的：保留原始交付版本。**

操作：

```text
model.ply
model_backup.ply
```

虽然脚本会生成一个新文件，但仍建议不要直接覆盖原文件。

#### 第二步：脚本裂解 Split / Optimize

**目的：修复已暴露的边缘毛刺。**

街区成品：

```bash
python gs_splitter.py model.ply --profile city
```

普通模型：

```bash
python gs_splitter.py model.ply --profile balanced
```

文物模型：

```bash
python gs_splitter.py model.ply --profile artifact
```

#### 第三步：成品复查 Final Review

**目的：确认最终效果。**

检查：

- 裁切线是否更锐利；
- 是否还有外凸球；
- 透明度是否变厚；
- 颜色是否保持一致；
- 文件体积是否可接受；
- 渲染器是否能正常加载。

#### 第四步：局部二次处理 Optional Refine

**目的：只针对少量极端区域补刀。**

如果还有毛刺：

```bash
python gs_splitter.py model_split.ply \
  --profile city \
  --set global_percentile=97.0 \
  --set local_ratio=1.9 \
  --set max_split=32
```

如果点数增长过多：

```bash
python gs_splitter.py model.ply \
  --profile city \
  --set max_output_multiplier=1.25 \
  --set max_split=24
```

---

### 8.4 存量 3DGS 模型批量优化

适合已有大量 PLY，需要统一做标准化修复的情况。

推荐流程：

```text
1. 抽样测试 Sample Test
2. 批量裂解 Batch Split
3. 批量质检 Batch Review
4. 参数回调 Optional Refine
```

#### 第一步：抽样测试 Sample Test

**目的：先确定参数，不要直接全量跑。**

操作：

- 从模型库中挑选 3 到 5 个代表样本；
- 包括一个街区模型、一个文物模型、一个问题严重模型、一个正常模型；
- 分别测试 `balanced`、`city`、`artifact`。

建议：

```bash
python gs_splitter.py sample_city.ply --profile city
python gs_splitter.py sample_artifact.ply --profile artifact
python gs_splitter.py sample_normal.ply --profile balanced
```

#### 第二步：批量裂解 Batch Split

**目的：统一处理。**

Linux / macOS 示例：

```bash
for f in *.ply; do
  python gs_splitter.py "$f" --profile balanced
done
```

街区批量：

```bash
for f in *.ply; do
  python gs_splitter.py "$f" --profile city --set max_output_multiplier=1.45
done
```

文物批量：

```bash
for f in *.ply; do
  python gs_splitter.py "$f" --profile artifact --set max_output_multiplier=2.2
done
```

#### 第三步：批量质检 Batch Review

**目的：确认没有过拆或欠拆。**

建议记录：

- 原始点数；
- 输出点数；
- 输出倍率；
- 异常点数量；
- 平均拆分数；
- 是否出现透明度变厚；
- 是否出现噪点。

#### 第四步：参数回调 Optional Refine

**目的：按类别建立项目参数。**

如果某一类模型体积过大：

```bash
--set max_output_multiplier=1.25 --set max_split=24
```

如果某一类模型修复不足：

```bash
--set global_percentile=96.5 --set local_ratio=1.8 --set max_split=64
```

如果透明区域变厚：

```bash
--set opacity_gain=0.95
```

---

### 8.5 高精度展示 / 下游加工 / 烘焙前处理

如果模型后续要进入 Web 展示、三维展示、纹理烘焙、模型转换、激光内雕或其他加工流程，可以把 GS-Splitter 作为预处理工具。

推荐流程：

```text
1. 成果整理 Prepare
2. 脚本裂解 Split / Optimize
3. 下游验证 Downstream Test
4. 轻量化 Optional Refine
```

#### 第一步：成果整理 Prepare

**目的：准备一个干净的输入模型。**

操作：

- 删除明显无关点；
- 确保 PLY 能被常用查看器正常加载；
- 备份原始文件；
- 确定下游平台对文件大小和点数的限制。

#### 第二步：脚本裂解 Split / Optimize

**目的：提高局部均匀性。**

通用展示：

```bash
python gs_splitter.py model.ply --profile balanced
```

高精度展示：

```bash
python gs_splitter.py model.ply --profile artifact --set max_output_multiplier=2.0
```

大场景展示：

```bash
python gs_splitter.py model.ply --profile city --set max_output_multiplier=1.35
```

#### 第三步：下游验证 Downstream Test

**目的：确保优化后的 PLY 适配后续流程。**

检查：

- 渲染器能否正常加载；
- 帧率是否可接受；
- 文件体积是否超限；
- 透明度是否自然；
- 边界和表面是否更稳定。

#### 第四步：轻量化 Optional Refine

**目的：在视觉效果和体积之间平衡。**

如果效果好但文件太大：

```bash
python gs_splitter.py model.ply \
  --profile balanced \
  --set max_output_multiplier=1.25 \
  --set max_split=24
```

如果下游更看重边缘锐度：

```bash
python gs_splitter.py model.ply \
  --profile city \
  --set shrink_extra=1.08 \
  --set max_split=48
```

---

## 9. 参数说明

### 9.1 异常检测相关参数

| 参数 | 含义 | 调大效果 | 调小效果 |
|---|---|---|---|
| `global_percentile` | 全局大半径候选分位 | 更保守，拆得更少 | 更积极，拆得更多 |
| `normal_percentile` | 判断局部正常邻居时使用的半径分位 | 局部参考半径更大，拆得更少 | 局部参考半径更小，拆得更多 |
| `local_ratio` | 点半径超过局部参考半径多少倍才确认异常 | 更保守 | 更积极 |
| `local_ratio_hard` | 即使没过全局阈值，也会被强制候选的局部倍数 | 更保守 | 更容易捕捉局部大点 |
| `robust_z` | 基于 MAD 的全局异常强度 | 更保守 | 更积极 |
| `knn` | 局部邻域点数量 | 局部判断更稳定，但更耗时 | 更敏感，但可能不稳定 |
| `min_alpha` | opacity 可见性过滤阈值 | 忽略更多透明噪点 | 会处理更多低透明点 |

### 9.2 拆分相关参数

| 参数 | 含义 | 调大效果 | 调小效果 |
|---|---|---|---|
| `min_split` | 单个异常点最少拆分数 | 最低拆分更强 | 最低拆分更弱 |
| `max_split` | 单个异常点最多拆分数 | 修复更充分，点数更大 | 更轻量，可能欠拆 |
| `target_radius_factor` | 子点目标半径相对局部正常半径的倍数 | 子点更大，点数更少 | 子点更小，细腻度更高 |
| `split_multiplier` | 拆分数量整体倍率 | 更多子点 | 更少子点 |
| `max_output_multiplier` | 输出点数上限倍率 | 允许更大文件 | 更严格控体积 |

### 9.3 空间分布和外观相关参数

| 参数 | 含义 | 建议 |
|---|---|---|
| `spread` | 子点在母高斯椭球中的散布强度 | 街区 0.60-0.70，文物 0.70-0.75 |
| `shrink_extra` | 子点 scale 额外收缩系数 | 1.00-1.10，越大边缘越锐，但覆盖越少 |
| `boundary_margin` | 子点中心离母椭球边界的安全余量 | 1.10-1.25 |
| `opacity_conserve` | 是否按 alpha 合成关系近似守恒 opacity | 默认 1，通常不要关闭 |
| `opacity_gain` | 子点透明度增益 | 默认 1.0，变厚可降到 0.95 |
| `rot_noise` | 子点旋转随机扰动 | 默认 0，不建议随便开启 |
| `color_noise` | f_dc 颜色扰动 | 默认 0，不建议随便开启 |
| `min_scale_log` | scale 的 log 下限 | 默认 -12 |
| `random_seed` | 随机种子 | 固定可复现 |
| `write_when_no_change` | 没有异常时是否仍写出文件 | 默认 1 |

---

## 10. 参数调优建议

### 10.1 边缘仍然毛

可以尝试：

```bash
--set global_percentile=97.0 --set local_ratio=1.8 --set max_split=64
```

### 10.2 文件变得太大

可以尝试：

```bash
--set max_output_multiplier=1.25 --set max_split=24
```

### 10.3 透明区域变厚或变亮

可以尝试：

```bash
--set opacity_gain=0.95
```

或者更保守：

```bash
--set max_split=48 --set opacity_gain=0.92
```

### 10.4 修复不足

可以尝试：

```bash
--set global_percentile=96.0 --set local_ratio=1.7 --set max_split=96
```

### 10.5 出现细碎噪点

可以尝试：

```bash
--set global_percentile=98.5 --set local_ratio=2.4 --set max_split=24
```

### 10.6 街区最终推荐起步参数

```bash
python gs_splitter.py rough_crop.ply \
  --profile city \
  --set max_split=32 \
  --set max_output_multiplier=1.45
```

### 10.7 文物最终推荐起步参数

```bash
python gs_splitter.py artifact.ply \
  --profile artifact \
  --set max_split=96 \
  --set max_output_multiplier=2.0
```

---

## 11. 为什么街区推荐“粗剪 → 裂解 → 精裁”

街区裁切有一个特殊问题：最终边界通常不是一开始就确定的，而是在 CloudCompare 或渲染器中沿道路牙子、围墙、红线一点点修出来的。

如果直接在未处理的大高斯模型上精裁，裁切线可能会穿过一些大半径高斯。因为这些高斯本身覆盖范围很大，即便中心点在边界内，它们的可见半径也可能跨出边界，于是形成：

- 外凸半圆球；
- 毛刺；
- 边界晕影；
- 裁切线不干净。

所以更稳的街区流程是：

```text
先粗剪，去掉绝对无关区域；
再裂解，让待裁切区域的大高斯变小；
再精裁，让最终边界由更小的点组成；
最后局部复查和二次处理。
```

这个流程的优势是：

- 粗剪降低计算量；
- 裂解提升边界附近的空间分辨率；
- 精裁决定最终轮廓；
- 局部 refine 解决极端残留问题。

但也要注意：

```text
如果粗剪后的模型仍然非常大，第二步一定要用 city 或保守参数。
不要在完整超大街区上直接使用 artifact 预设。
```

---

## 12. 算法细节

### 12.1 半径读取

标准 3DGS PLY 中的 `scale_0/1/2` 通常是 log scale。

工具会先做：

```text
scale_linear = exp(scale_log)
```

然后使用三个轴中的最大值作为异常判断的主要半径：

```text
size = max(scale_0, scale_1, scale_2)
```

这样更容易捕捉会“刺出来”的长轴高斯。

### 12.2 局部正常半径

工具会为每个高斯点建立 KNN 邻域，计算周围正常邻居的参考半径。

大致逻辑：

```text
当前点半径 / 周围正常点半径 = 局部异常倍数
```

如果一个点在全局上偏大，且相对周围邻居也明显偏大，就会被确认为异常点。

### 12.3 异常判断

工具综合三类信号：

1. 全局分位数异常；
2. MAD 鲁棒统计异常；
3. 局部邻域倍数异常。

这样可以避免只靠单一阈值造成误判。

### 12.4 拆分数量

拆分数量近似基于体积关系：

```text
n_split ≈ (父点半径 / 目标子点半径)^3
```

并受以下参数限制：

```text
min_split <= n_split <= max_split
```

同时受输出点数预算限制：

```text
输出点数 <= 原始点数 × max_output_multiplier
```

### 12.5 子点位置

子点不会无限随机散开，而是在母高斯椭球内部进行有界采样。

这可以降低新生成子点飞出原始覆盖范围、制造新毛刺的风险。

### 12.6 子点属性继承

子点会继承父点的大部分属性，包括：

- 颜色；
- 球谐系数；
- 旋转；
- 法线；
- 其他 PLY 字段。

### 12.7 opacity 近似守恒

如果父点透明度为 `alpha_parent`，拆成 `n` 个子点后，工具默认使用近似合成关系计算子点 alpha：

```text
alpha_child = 1 - (1 - alpha_parent)^(1 / n)
```

这样多个子点叠加后的总体透明度更接近父点，而不是简单复制父点 opacity。

简单复制 opacity 会导致局部变厚、变亮、发糊。

---

## 13. 常见问题 FAQ

### Q1：这个工具需要原始图片或相机参数吗？

不需要。

它只处理最终导出的 3DGS `.ply` 文件。

### Q2：它会不会改变模型颜色？

默认不会主动改变颜色字段。

子点会继承父点的颜色和球谐字段。除非你手动设置 `color_noise > 0`，否则不会添加颜色扰动。

### Q3：它会不会改变透明度？

会改变子点的 opacity，但目的是让多个子点叠加后的总体透明度接近父点。

默认不建议关闭 `opacity_conserve`。

### Q4：为什么输出文件变大了？

因为异常大点被替换成多个子点，点数会增加。

可以通过以下参数控制：

```bash
--set max_output_multiplier=1.25 --set max_split=24
```

### Q5：为什么我处理后看不出变化？

可能原因：

- 模型中没有明显大半径异常点；
- 参数太保守；
- 你的查看器没有明显展示边缘差异；
- 问题不是大半径点造成，而是训练几何本身错误。

可以尝试：

```bash
--set global_percentile=96.5 --set local_ratio=1.8
```

### Q6：为什么处理后局部变厚？

可能是拆分太多或透明度增益偏高。

可以尝试：

```bash
--set opacity_gain=0.95 --set max_split=48
```

### Q7：可以连续处理多次吗？

可以，但不建议盲目多次全局处理。

推荐方式：

```text
第一次：正常参数全局处理
第二次：只在成品检查后，用更保守参数处理残留问题
```

### Q8：支持普通点云 PLY 吗？

不支持普通点云。

输入 PLY 必须包含 3DGS 所需字段，至少包括：

```text
x, y, z, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
```

### Q9：支持 binary PLY 吗？

支持常见 binary PLY 读取和写出。

输出默认是 binary PLY。

### Q10：能不能只处理某个局部区域？

当前推荐做法是先在外部工具里裁出局部区域，再运行脚本。

未来可以增加空间包围盒、mask 或交互式局部选择功能。

---

## 14. 故障排查

### 14.1 缺少依赖

如果看到：

```text
ModuleNotFoundError: No module named 'plyfile'
```

运行：

```bash
pip install plyfile
```

如果看到：

```text
ModuleNotFoundError: No module named 'scipy'
```

运行：

```bash
pip install scipy
```

### 14.2 GUI 拖拽不可用

拖拽功能依赖：

```bash
pip install tkinterdnd2
```

即使没有拖拽，也可以点击选择文件。

### 14.3 内存不足

可以尝试：

- 先 Rough Crop；
- 使用 `city` 预设；
- 降低 `knn`；
- 降低 `max_output_multiplier`；
- 降低 `max_split`；
- 分块处理模型。

示例：

```bash
python gs_splitter.py input.ply \
  --profile city \
  --set knn=24 \
  --set max_split=24 \
  --set max_output_multiplier=1.25
```

### 14.4 渲染器加载异常

可能原因：

- 输入 PLY 字段不是标准 3DGS 格式；
- 下游查看器对字段顺序或字段类型敏感；
- 输出点数过大；
- 原始文件中存在 NaN 或 Inf。

建议：

- 先用小文件测试；
- 确认输出 PLY 仍包含原始字段；
- 降低输出倍率；
- 检查原始 PLY 是否损坏。

---

## 15. 推荐工作流速查表

| 场景 | 推荐流程 | 推荐 profile |
|---|---|---|
| 街区最终裁切 | 粗剪 → 裂解 → 精裁 → 局部检查 | `city` |
| 已裁切街区修毛刺 | 备份 → 裂解 → 复查 → 二次处理 | `city` |
| 文物透光 / 反光瑕疵 | 基础清理 → 裂解 → 表面检查 → 局部细化 | `artifact` |
| 普通存量模型 | 备份 → 裂解 → 复查 → 参数回调 | `balanced` |
| 批量模型标准化 | 抽样测试 → 批量裂解 → 批量质检 → 参数回调 | 按类别选择 |
| 下游展示 / 加工 | 成果整理 → 裂解 → 下游验证 → 轻量化 | `balanced` / `city` |

---

## 16. 开源协议

本项目建议使用：

```text
Apache License 2.0
```

开源时请在仓库根目录添加：

```text
LICENSE
```

并放入完整的 Apache License 2.0 正文。

如果项目中后续包含第三方代码、示例模型或外部资源，也建议增加：

```text
NOTICE
```

用于说明额外版权和引用信息。

---

## 17. Roadmap

计划或可考虑的后续功能：

- [ ] 局部区域处理：通过 bounding box 或 mask 只处理指定区域；
- [ ] 边界感知处理：针对裁切边界附近更积极拆分；
- [ ] 可视化报告：输出异常点数量、半径分布、拆分前后统计图；
- [ ] 批处理 GUI：一次拖入多个 PLY；
- [ ] 参数模板保存：保存项目级参数预设；
- [ ] 更强的内存控制：超大模型分块 KDTree；
- [ ] Before / After 平行对比。

---

## 18. 贡献

欢迎提交：

- 不同 3DGS 导出格式的兼容性反馈；
- 街区 / 文物 / 小物体场景的参数建议；
- Before / After 对比截图；
- Bug report；
- 性能优化；
- GUI 改进；
- 文档翻译。

建议 issue 中包含：

```text
1. 使用的 profile
2. 修改过的参数
3. 原始点数和输出点数
4. 问题截图
5. 是否为街区 / 文物 / 普通模型
6. 使用的查看器或渲染器
```

---

## 19. 免责声明

GS-Splitter 是一个后处理工具，不能保证对所有 3DGS 模型都产生理想结果。

请在正式交付前：

- 备份原始 PLY；
- 使用小区域测试参数；
- 检查输出文件体积；
- 在目标渲染器中复查；
- 对重要项目保留人工质检流程。
