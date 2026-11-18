# TuMeNiche：10 个数据集 notebook 输入与结果核对

依据：本轮上传的 10 份 `(1).ipynb` 的代码及已保存执行输出。日期：2026-11-11。

已找到全部 10 个样本的完整表达 h5ad、最终结果 h5ad、SpaCET 反卷积 CSV、初始预测 CSV、GMM 阈值 CSV，以及病理注释字段。可以直接据此编写实验 notebook，无需用户重新整理这些路径。

本次是对 notebook 的静态核对；执行输出是此前运行记录。没有打开服务器上的实际 h5ad，因此不把文件存在性、当前维度或当前字段状态视为已经验证。

## 1. 可直接复用的字段与读取流程

| 信息 | 代码位置/字段 | 实验中的使用方式 |
|---|---|---|
| 完整表达及原始空间对象 | 各文件 Cell 3 的 `file_path`，`sc.read_h5ad(file_path)` | 读取进入训练基因筛选之前的全部可用基因 |
| 原生微簇 | Cell 6，`obs["latent_leiden_0.6"]` | 从最终保存对象复用，固定两组划分 |
| 潜在表征 | `obsm["nichecompass_latent"]` | 已有表征的记录与核对 |
| 空间坐标 | `obsm["spatial"]` | spot 对齐及绘图 |
| 空间图 | `obsp["spatial_connectivities"]` | 复用同一张图 |
| 恶性分数 | 原始 SpaCET 合并及 GMM 使用 `obs["Malignant"]`；Cell 7 另有候选列检测 | 固定恶性信息 |
| 源端/接收端分数 | `obs[f"{axis}_Sender_Score"]`、`obs[f"{axis}_Receiver_Score"]` | 完整耦合、Sender-only |
| AUCell 原始输出 | Cell 4，`obsm["aucell_estimate"]` | 优先检查和复用；缺失时按原评分逻辑重算 |
| PROGENy 原始输出 | Cell 4，`obsm["mlm_estimate"]` | 记录或复用通路评分 |
| 初始预测列 | CSV 中以 `Unsupervised_Pred_` 开头的列 | 重算相同有效 spot 上的 SpaCET 基线 |
| GMM 阈值 | 阈值 CSV 的 `Metric == "Optimal_GMM_Threshold"` 对应 `Value` | 固定门控阈值 |
| 最终层级 | `obs["Level1_Macro_Region"]`、`obs["Level2_Micro_Niche"]` | 最终配对评价 |

Cell 4 用完整对象 `adata.X` 进行 PROGENy、AUCell 和接收端评分，调用采用 `use_raw=False`。Cell 5 的 `adata_model = adata.copy()` 及基因筛选会保留 obs 和 obsm，所以已有 AUCell/PROGENy 输出按代码流应被继承；实际文件仍应在运行时检查。

### 表达处理的结论

- 10 个文件的 Cell 3 都重新从磁盘加载表达 h5ad。因此，先前 SpaCET 基准评估中对内存对象执行的 `normalize_total`/`log1p`，不能直接认定会延续到 Cell 4。Cell 3 的重新读取是明确的对象分界。
- Cell 4 直接使用重新加载对象的 `.X`，这两个单元之间未发现新增的 normalize_total/log1p 调用。
- Cell 5 用 `"counts" if "counts" in adata_model.layers else None` 选择训练矩阵。它是运行时判断，不能仅凭这行源码断言每个磁盘文件都存在 counts 层。B1 的已保存输出明确显示该次运行 `counts_key = None`。
- 对 `.X` 的原始计数属性，notebook 没有完整记录上游 h5ad 的生成过程；新代码应自动记录 layers、raw、log1p 元数据以及数值特征。数值看起来像整数也不能单独证明表达数据从未处理。无需为此要求用户重复提供表达文件路径。
- 新评分按最终结果的 obs_names 对齐，固定同一批 spot；完整表达文件用于评分基因背景，最终文件用于冻结上游划分，两者用途不同。

## 2. 样本清单

| 样本 | 病理注释列 | 原生微簇列 | 恶性门控偏移 |
|---|---|---|---|
| Human_Prostate_Cancer | `Pathology_Annotation` | `latent_leiden_0.6` | tumor_threshold − 0.25，下限 0 |
| Human_breast_cancer_ViHBC | `annot_type` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| SN048_A121573_Rep1 | `pathology_annotation` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| SN048_A121573_Rep2 | `pathology_annotation` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| SN123_A595688_Rep1 | `pathology_annotation` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| SN124_A595688_Rep2 | `pathology_annotation` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| SN84_A120838_Rep1 | `pathology_annotation` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| lung_cancer_and_kidney_cancer_LC1 | `ground_truth` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| lung_cancer_and_kidney_cancer_LC5 | `ground_truth` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |
| pancreatic_cancer_B1 | `label` | `latent_leiden_0.6` | tumor_threshold − 0.10，下限 0 |

## 3. 每个样本的完整路径与评估规则

以下路径均原样来自代码的活动赋值；“最终结果”优先采用 Cell 11 实际评估读取的路径，并核对其与 Cell 10 一致。

### Human_Prostate_Cancer

来源文件：`myAndSpaCET_Human_Prostate_Cancer(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_Prostate_Cancer/Human_Prostate_Cancer_annotated.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/test/Human_Prostate_Cancer/SpaCET_And_My/Human_Prostate_Cancer_annotated.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/Human_Prostate_Cancer/Human_Prostate_Cancer_CellFractions.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/test/Human_Prostate_Cancer/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/test/Human_Prostate_Cancer/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_Prostate_Cancer/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/test/Human_Prostate_Cancer/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_Prostate_Cancer/SpaCET_And_My
```

病理注释列：`Pathology_Annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.25)`。

Cell 11 评估映射：

```json
"Invasive carcinoma -> 1; other nonmissing observed labels -> 0 (source dict comprehension)"
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "Tumor": 1,
  "Invasive": 1,
  "Surrounding tumor": 0,
  "Healthy": 0
}
```

映射差异：

```json
{
  "note": "Cell10 applies breast-style Tumor/Invasive/Surrounding tumor/Healthy map to Pathology_Annotation; Cell11 uses Invasive carcinoma."
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 4371 spots, 17943 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 4371 spots, 3047 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### Human_breast_cancer_ViHBC

来源文件：`myAndSpaCET_Human_breast_cancer_ViHBC(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET_And_My/Human_breast_cancer_ViHBC_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/Human_breast_cancer_ViHBC/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET_And_My
```

病理注释列：`annot_type`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "Tumor": 1,
  "Invasive": 1,
  "Surrounding tumor": 0,
  "Healthy": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "Tumor": 1,
  "Invasive": 1,
  "Surrounding tumor": 0,
  "Healthy": 0
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 3798 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 3798 spots, 36601 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 12, "load": 13, "score": 15, "final": 21, "evaluation": 23}`。

### SN048_A121573_Rep1

来源文件：`myAndSpaCET_SN048_A121573_Rep1(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/12_colorectal_cancer_CRC/12_colorectal_cancer_CRC_h5ad/SN048_A121573_Rep1.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET_And_My/SN048_A121573_Rep1_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/12_colorectal_cancer_CRC/SN048_A121573_Rep1/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET_And_My
```

病理注释列：`pathology_annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 0,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 1,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

映射差异：

```json
{
  "stroma_fibroblastic_IC high": {
    "Cell10": 1,
    "Cell11": 0
  }
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 2052 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 2052 spots, 36601 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 9, "train": 12, "load": 13, "score": 15, "final": 21, "evaluation": 23}`。

### SN048_A121573_Rep2

来源文件：`myAndSpaCET_SN048_A121573_Rep2(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/12_colorectal_cancer_CRC/12_colorectal_cancer_CRC_h5ad/SN048_A121573_Rep2.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET_And_My/SN048_A121573_Rep2_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/12_colorectal_cancer_CRC/SN048_A121573_Rep2/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep2/SpaCET_And_My
```

病理注释列：`pathology_annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 0,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 1,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

映射差异：

```json
{
  "stroma_fibroblastic_IC high": {
    "Cell10": 1,
    "Cell11": 0
  }
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 2291 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 2291 spots, 36601 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 9, "train": 12, "load": 13, "score": 15, "final": 21, "evaluation": 23}`。

### SN123_A595688_Rep1

来源文件：`myAndSpaCET_SN123_A595688_Rep1(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/12_colorectal_cancer_CRC/12_colorectal_cancer_CRC_h5ad/SN123_A595688_Rep1.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET_And_My/SN123_A595688_Rep1_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/12_colorectal_cancer_CRC/SN123_A595688_Rep1/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN123_A595688_Rep1/SpaCET_And_My
```

病理注释列：`pathology_annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "tumor": 1,
  "tumor&stroma IC med to high": 1,
  "stroma_fibroblastic_IC high": 0,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregate_submucosa": 0,
  "stroma_fibroblastic_IC med": 0,
  "muscularis_IC med to high": 0,
  "IC aggregate_muscularis or stroma": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 0,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 1111 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 1111 spots, 3037 genes

当前 Cell 5 活动代码中的基因筛选调用：`未发现`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### SN124_A595688_Rep2

来源文件：`myAndSpaCET_SN124_A595688_Rep2(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/12_colorectal_cancer_CRC/12_colorectal_cancer_CRC_h5ad/SN124_A595688_Rep2.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET_And_My/SN124_A595688_Rep2_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/12_colorectal_cancer_CRC/SN124_A595688_Rep2/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN124_A595688_Rep2/SpaCET_And_My
```

病理注释列：`pathology_annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 0,
  "stroma_fibroblastic_IC med": 0,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "muscularis_IC med to high": 0,
  "IC aggreates_stroma or muscularis": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 1,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

映射差异：

```json
{
  "stroma_fibroblastic_IC high": {
    "Cell10": 1,
    "Cell11": 0
  }
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 356 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 356 spots, 36601 genes

当前 Cell 5 活动代码中的基因筛选调用：`未发现`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### SN84_A120838_Rep1

来源文件：`myAndSpaCET_SN84_A120838_Rep1(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/12_colorectal_cancer_CRC/12_colorectal_cancer_CRC_h5ad/SN84_A120838_Rep1.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET_And_My/SN84_A120838_Rep1_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/12_colorectal_cancer_CRC/SN84_A120838_Rep1/SpaCET_CellFractions_Result.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN84_A120838_Rep1/SpaCET_And_My
```

病理注释列：`pathology_annotation`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "tumor&stroma_IC med to high": 1,
  "tumor": 1,
  "tumor&stroma_IC low": 1,
  "stroma_fibroblastic_IC high": 0,
  "stroma_fibroblastic_IC med": 0,
  "stroma_fibroblastic_IC low": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "tumor": 1,
  "tumor&stroma_IC med to high": 1,
  "stroma_fibroblastic_IC high": 1,
  "epithelium&submucosa": 0,
  "non neo epithelium": 0,
  "submucosa": 0,
  "IC aggregregate_submucosa": 0,
  "IC aggregate_submucosa": 0,
  "IC aggregate connective tissue": 0
}
```

映射差异：

```json
{
  "stroma_fibroblastic_IC high": {
    "Cell10": 1,
    "Cell11": 0
  }
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 240 spots, 36601 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 240 spots, 36601 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### lung_cancer_and_kidney_cancer_LC1

来源文件：`myAndSpaCET_lung_cancer_and_kidney_cancer_LC1(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/lung_cancer_and_kidney_cancer/LC1.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET_And_My_AllGenes/lung_cancer_and_kidney_cancer_LC1_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/lung_cancer_and_kidney_cancer/LC1/LC1_CellFractions.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET_And_My_AllGenes
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC1/SpaCET_And_My_AllGenes
```

病理注释列：`ground_truth`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "TUM": 1,
  "NOR": 0,
  "TLS": 0,
  "INFL": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "TUM": 1,
  "NOR": 0,
  "TLS": 0,
  "INFL": 0
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 2475 spots, 17845 genes

已保存执行输出中的最终维度：未提供；不能从代码路径推断当前磁盘结果已生成。

当前 Cell 5 活动代码中的基因筛选调用：`未发现`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### lung_cancer_and_kidney_cancer_LC5

来源文件：`myAndSpaCET_lung_cancer_and_kidney_cancer_LC5(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/lung_cancer_and_kidney_cancer/LC5.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET_And_My/lung_cancer_and_kidney_cancer_LC5_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/lung_cancer_and_kidney_cancer/LC5/LC5_CellFractions.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/lung_cancer_and_kidney_cancer/LC5/SpaCET_And_My
```

病理注释列：`ground_truth`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "TUM": 1,
  "NOR": 0,
  "TLS": 0,
  "INFL": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "TUM": 1,
  "NOR": 0,
  "TLS": 0,
  "INFL": 0
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 622 spots, 17845 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 622 spots, 3032 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

### pancreatic_cancer_B1

来源文件：`myAndSpaCET_pancreatic_cancer_B1(1).ipynb`。以下单元编号采用代码中写出的 Cell 标签，物理 notebook 第几个单元另在括号内记录。

**完整表达 h5ad（Cell 3）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/pancreatic_cancer/B1.h5ad
```

**最终评估 h5ad（Cell 11）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET_And_My/pancreatic_cancer_B1_NicheCompassAndmy_Final.h5ad
```

**SpaCET 细胞组成 CSV**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/datasets/SpaCET_R_Result/pancreatic_cancer/B1/B1_CellFractions.csv
```

**SpaCET 初始二分类预测 CSV（Cell 10 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv
```

**SpaCET GMM 阈值 CSV（Cell 7 读取）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET/GMM_Optimal_Threshold_Result.csv
```

**GMM 生成目录（Cell 0-1）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET
```

**模型训练保存目录（Cell 5）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET_And_My
```

**模型读取目录（Cell 6）**

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/pancreatic_cancer/B1/SpaCET_And_My
```

病理注释列：`label`。恶性门控：`max(0.0, hard_tumor_threshold - 0.1)`。

Cell 11 评估映射：

```json
{
  "invasive cancer": 1,
  "connective tissue": 0,
  "adipose tissue": 0,
  "breast glands": 0
}
```

Cell 10 绘图/辅助病理字段映射：

```json
{
  "invasive cancer": 1,
  "connective tissue": 0,
  "adipose tissue": 0,
  "breast glands": 0
}
```

已保存执行输出中的输入维度：✅ 数据加载成功: 269 spots, 15109 genes

已保存执行输出中的最终维度：✅ 数据加载成功: 268 spots, 3027 genes

当前 Cell 5 活动代码中的基因筛选调用：`sc.pp.filter_genes, sc.pp.highly_variable_genes`。

来源物理单元（从 1 开始）：`{"input": 8, "train": 11, "load": 12, "score": 14, "final": 20, "evaluation": 22}`。

## 4. 会影响配对实验的具体不一致

### 4.1 前列腺的目录不一致

Cell 0-1 的 GMM 生成目录与 Cell 6 的模型加载目录位于 `outputs/Human_Prostate_Cancer/`；Cell 5 训练保存、Cell 7 门控阈值读取、Cell 9–11 最终结果位于 `outputs/test/Human_Prostate_Cancer/`。

仅凭目录不同不能证明文件内容不同。新实验应优先固定 Cell 11 使用的最终对象，并检查相应预测和阈值的 spot、数值及来源；不通过重新训练来消除路径差异。

前列腺 Cell 10 对 `Pathology_Annotation` 使用 Tumor/Invasive/Surrounding tumor/Healthy 映射；Cell 11 则以 Invasive carcinoma 为肿瘤。新的配对评价应共用 Cell 11 的一份映射并重新生成辅助病理标签。

### 4.2 结直肠癌 Cell 10 / Cell 11 映射不一致

SN048 Rep1、SN048 Rep2、SN84 Rep1、SN124 Rep2 的 `stroma_fibroblastic_IC high` 在 Cell 10 为 1，在 Cell 11 为 0。SN123 的该条映射未发现同样差异。

因此不能直接依赖保存的 `Pathology_Binary_ARI` 作为统一 ground truth。建议从原病理字符串列出发，按每个样本 Cell 11 的映射，对初始预测和最终预测使用同一标签和有效 spot 集。这里的 0/1 是代码现状记录，不是重新判定病理归属。

### 4.3 原始 spot 与最终 spot 数量可能不同

B1 的 Cell 3 已保存输出为 269 spots、15109 genes；Cell 10 已保存输出为 268 spots、3027 genes。Cell 5 包含总计数无效 spot 的过滤，因而应对齐最终 spot 集后重算初始及最终指标，不能直接将旧基准 CSV 与新结果 CSV 相减。

### 4.4 当前源代码与保存输出并非总是同一轮运行

- 多个 notebook 当前 Cell 5 存在 HVG 筛选，但其保存输出写着“输入基因数 = 训练基因数 = 36601”；这些输出不能单独证明当前代码训练的实际基因数。
- SN123/SN124 仅旧版基因筛选段被注释，同一 Cell 后面仍有活动的全基因复制与训练代码；后续 Cell 6 从保存模型加载。两者已保存最终输出分别为 3037 和 36601 个基因，SN123 的全基因训练输出与最终加载输出仍不一致。
- 前列腺 Cell 5 输出训练 17943 基因，但最终 Cell 10 输出 3047 基因，结合 train/load 目录不一致，需固定实际最终对象而非仅凭最近修改的训练代码判断版本。
- LC1 源代码中训练、加载和最终结果路径均为 `SpaCET_And_My_AllGenes`，相应训练/后续单元未见保存执行输出；新代码应自动检查该文件是否已存在。

这些都不妨碍编写评分对照代码；它们要求在 notebook 开头打印并冻结本次实际加载的文件、维度、spot ID、微簇标签和图。

## 5. 可直接采用的新实验读取顺序

1. 从上述 Cell 11 路径读取最终对象，冻结 obs_names、微簇、空间图、恶性分数及既有完整方案标签。
2. 从 Cell 3 路径读取完整表达对象，并按冻结的 spot ID 对齐；保留全部可用基因作为评分背景。
3. 从已有预测 CSV、阈值 CSV 读取 SpaCET 初始预测及阈值；所有方案沿用同一份输入。
4. 优先复用保存的 Sender/Receiver/AUCell/PROGENy；缺少的评分严格按原 Cell 4 逻辑从完整表达对象重算。
5. 病理标签以各样本 Cell 11 为统一来源；初始与最终结果在同一有效 spot 集上重新评价。
6. 评分方案的阈值尺度处理作为明确的实验配置；保持完整方案原始输出与用于可比尺度分析的输出可区分。
7. 将新实验结果保存到独立实验目录，保留来源记录。

## 6. 仍需运行时读取、但不要求用户重新整理的信息

- 服务器文件实际存在性及当前内容；h5ad 的 layers、raw、obsm/obsp 键。
- 最终文件是否含已保存微簇、图及全部评分；缺失时按已有路径定位中间产物。
- .X 的数值特征与明确的 counts 元数据；已有代码直接使用 .X 的行为可复现，但上游表达预处理不能仅凭路径名认定。
- 输入和最终 spot 的一对一对应、未映射病理标签、缺失基因与全零评分轴。

患者/切片重复关系未在这些代码中形成明确的对应表。它不阻碍逐切片运行与描述性配对汇总；独立患者层面的推断需有相应分组依据，不能从文件名擅自确定。
