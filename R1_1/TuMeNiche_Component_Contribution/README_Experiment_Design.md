# TuMeNiche：审稿人第 1 条意见的组成部分贡献实验

## 运行顺序

1. 在学校服务器原 NicheCompass 环境解压本包。
2. 逐份 Restart Kernel → Run All，或者打开 `00_Run_All.ipynb` 顺序运行；控制文件会为每个样本启动独立 kernel。
3. 最后运行 `notebooks/11_Summary_All_10_Datasets.ipynb`。

10 份样本 notebook 都自包含。`component_engine.py` 和 `sample_configs.json` 是同一实现的可读副本，不是单份 notebook 运行的外部依赖。

输出主目录：
```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/case/TuMeNiche_Component_Contribution/
```
启动任意样本后自动创建 10 个样本子目录和 `_summary`。本交付并未连接学校服务器，因此目录由 notebook 在服务器运行时创建。

## 样本及 GPU

| Notebook | 样本 | 原 GPU |
|---|---|---|
| `01_Human_breast_cancer_ViHBC_Component_Experiment.ipynb` | Human_breast_cancer_ViHBC | 1 |
| `02_Human_Prostate_Cancer_Component_Experiment.ipynb` | Human_Prostate_Cancer | 1 |
| `03_B1_Component_Experiment.ipynb` | B1 | 0 |
| `04_LC1_Component_Experiment.ipynb` | LC1 | 0 |
| `05_LC5_Component_Experiment.ipynb` | LC5 | 0 |
| `06_SN048_A121573_Rep1_Component_Experiment.ipynb` | SN048_A121573_Rep1 | 0 |
| `07_SN048_A121573_Rep2_Component_Experiment.ipynb` | SN048_A121573_Rep2 | 0 |
| `08_SN84_A120838_Rep1_Component_Experiment.ipynb` | SN84_A120838_Rep1 | 0 |
| `09_SN123_A595688_Rep1_Component_Experiment.ipynb` | SN123_A595688_Rep1 | 0 |
| `10_SN124_A595688_Rep2_Component_Experiment.ipynb` | SN124_A595688_Rep2 | 1 |


## 预先确定的实验设计

本实验检验组成部分的贡献，允许收益为正、为零或为负，不按病理 ARI 选择参数。

| 比较 | 回答的问题 | 保持固定的内容 |
|---|---|---|
| 初始 SpaCET–GMM vs 保存的最终 Level 1 | 宏观并集输出改变了多少？ | 同一病理映射、同一有效 spot 集 |
| FullCoupling vs SenderOnly | 加入邻域接收端有什么增益？ | 发送端分数、微簇、空间图、恶性分数、门控及生态位筛选规则 |
| FullCoupling vs AUCellOnly | 相比单 spot 源端基因活性，完整评分增加了什么？ | 原 Source_Genes 集合、完整评分基因背景、微簇、恶性信息和后续规则 |

定义（P、E、R 均沿用原评分的定义）：

\[
S_{i,k}=P_{i,k}E_{i,k},\quad
F_{i,k}=\log(1+S_{i,k}\sum_j A_{ij}R_{j,k}),\quad
M^{sender}_{i,k}=\log(1+S_{i,k}),\quad M^{AUCell}_{i,k}=E_{i,k}.
\]

SenderOnly 是针对接收端空间耦合的消融。AUCellOnly 同时去掉了 PROGENy 调制及接收端空间耦合，不能仅将其差异归因于接收端。

### 评分尺度与固定阈值

- `original_rules`：保留原始评分值和原 `T_high < 0.15` 安全锁，用于复现保存的完整方案，并作为尺度敏感性分析。
- `common_scale`：各方案在同一样本、每条轴上分别除以该评分的最大值，零轴保留为零；双方都重跑相同 GMM/兜底及筛选规则。该模式的 `0.15` 是相对阈值，**不是原始耦合尺度上的 0.15**。
- 主要评分对照报告 common_scale 的配对差值，同时报告 original_rules，不能隐藏两种设置的差异。缩放本身也是实验设置，不把 common_scale 的 FullCoupling 称为未改动的原始输出。
- 初始 vs 最终 Level 1 的比较始终使用保存的最终标签；不依赖上述缩放。

原筛选：FDR < 0.05、活跃比例 FC ≥ 1.30、活跃 spot ≥ 10、均值强度比 ≥ 1.20；恶性门控后保留 spot ≥ 10 且保留比例 ≥ 0.50。各样本沿用其原 gate offset。

### Level 2 的评价边界

固定 ROI 为初始 SpaCET 肿瘤 spot，避免各方案改变评价区域。报告活跃覆盖、各轴重叠、标签转移、空间邻域一致性，以及非恶性 SpaCET 细胞组成的关联。

细胞组成关联先汇总到**固定微簇**，再计算调整后 eta² 和逐微簇留一的 LOCO R²；标签使用该微簇在固定 ROI 内占比最多的 Level 2 类别。LOCO R² 对未在其他微簇出现的类别回退到训练微簇的总体均值，允许为负；这是转导标签的关联诊断，不是有监督模型的独立泛化验证。

空间一致性较高或重叠率较低均不自动意味着更准确；SpaCET 细胞组成也来自同一转录组，不能作为独立代谢流证据。若指定验证 GMT，将排除全部 Source/Target 基因和 PROGENy top=500 footprint 基因，且保留足够的验证基因。

### 已有方法参照

AUCellOnly 使用自定义的 10 组 Source_Genes，需命名为“AUCell-based source activity baseline”，**不冒充原版 scMetabolism**。

另有实际 R 包 `scMetabolism::sc.metabolism(method="AUCell", imputation=FALSE, metabolism.type="KEGG")` 参照分支。它保留包内 KEGG 集合，输出 spot 通路评分及预定义四通路均值，不强行映射成 10 条通讯轴，也不重新聚类替换 NicheCompass。R 或包未安装时明确记录 skipped，运行错误记录 failed，均不会制造结果。

参考：[scMetabolism 官方实现](https://github.com/wu-yc/scMetabolism/blob/main/R/compute_metabolism.R)；[2024 年口腔鳞癌空间代谢分区应用](https://doi.org/10.1038/s41368-023-00267-8)；[2025 年胰腺癌应用](https://doi.org/10.1016/j.isci.2025.112681)。


## 输入与参数

- 完整表达来自各原 notebook Cell 3；最终结果来自 Cell 11。全部路径已经填好。
- 固定 `latent_leiden_0.6`、原空间图、原恶性分数、原 Sender/Receiver。不会重新训练 NicheCompass。
- 前列腺 gate offset=0.25，其他样本=0.10，与本轮上传源码一致；不按 ARI 调参。
- 结直肠癌病理标签采用各样本 Cell 11；前列腺采用 Invasive carcinoma vs 其他有效类型；不直接使用可能过时的 Pathology_Binary_ARI。
- B1 等存在 spot 过滤时，用最终对象的 obs_names 对齐所有方案，避免不同有效集合的指标相减。
- LC1 指向 AllGenes 结果，前列腺指向 Cell 11 使用的 test 目录。不自动换成其他目录的模型；若服务器文件位置变动，仅在 CONFIG 中修改实际路径。
- 若原始 FullCoupling 未复现既有 Level 1/2，严格模式会输出差异并停止评分归因。初始 vs 保存最终的配对表仍可汇总。源代码和旧执行输出有差异时，此检查用于避免混用版本。
- `validation_gmt=None`：默认不需要额外 GMT。若启用，使用非评分输入基因的保留集合，并记录排除基因。

## 环境

沿用原 Python 环境（Python 3.9 及以上，numpy/pandas/scipy/scikit-learn/anndata/matplotlib；需要重算 AUCell 时使用原 decoupler 1.x 的 run_aucell API）。
若 AUCell 已保存且完整 spot 集相同，会直接复用，不更换评分实现。依赖版本在每次运行输出中记录。

原版 scMetabolism 分支需要 `Rscript` 及 R 包 Matrix、scMetabolism、AUCell、GSEABase。该分支默认尝试运行，缺少依赖会记录 skipped；不自动安装或声称已有结果。
如需安装原版方法，在自己的 R 环境中按官方说明执行：
```r
install.packages("remotes")
remotes::install_github("wu-yc/scMetabolism")
```
AUCell、GSEABase 等 Bioconductor 依赖按官方说明安装；不影响固定流程的三方案主对照运行。

## 每个样本的主要输出

- `input_audit.json`、`run_configuration.json`、`frozen_spots.csv`：输入来源、有效 spot、参数和环境。
- `full_reconstruction_check.json`、`full_reconstruction_mismatches.csv`：完整方案复现核对。
- `level1_metrics.csv`：SpaCET、保存的最终输出和六个评分条件的 ARI/F1/IoU/Precision/Recall 等。
- `level1_added_spots.csv`：Level 1 并集新增 spot 数、病理对应及是否意外删除初始肿瘤 spot。
- `original_rules__*`、`common_scale__*`：各条件阈值、微簇统计和恶性门控记录。
- `level2_summary.csv`、`level2_validation_features.csv`、`level2_axis_overlap.csv`：固定 ROI 的 Level 2 描述、细胞组成关联与轴重叠。
- `figures/`：宏观和微观空间图、预设轴评分图、细胞组成关联图。PNG 300 dpi，同时保存部分 PDF。
- `component_comparison_results.h5ad`：轻量结果；含原坐标、图、评分、全部标签，表达通过源文件路径获取。
- `scMetabolism_reference/`：实际原版 R 方法的状态、评分、sessionInfo 和图，未运行不会伪造。
- `freeze_integrity_check.json`：上游输入在评分比较前后的哈希一致性。

## 统计汇总

`_summary/paired_effect_size_summary.csv` 给出逐切片配对差值的均值、中位数、SD、IQR、范围及变化方向计数。
默认不把 10 张切片当成 10 个独立患者。需要独立单位推断时，用真实患者/重复关系填写自动生成的模板，并保存为 `_summary/independent_units.csv`，显式确认后重跑汇总。
重复切片先在独立单位内平均差值，之后才做 bootstrap CI 和配对 Wilcoxon，并输出多重检验 q 值。

## 验证范围

交付前执行语法、notebook 格式、合成 AnnData 端到端、原 Cell 7 逻辑对照、输入冻结、配对统计与汇总测试。
没有访问学校服务器的真实 h5ad，也没有执行其 R scMetabolism 分支，因此真实 10 样本结果需在服务器运行后产生。
本包不填入模拟实验结果，不预设方法一定优于简化评分。
