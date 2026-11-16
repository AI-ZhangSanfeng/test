你这个修正是对的。**Result 10 的实验设计应该改成“病理肿瘤内部功能异质性验证”**，而不是用整体 ARI 或全切片生态位对比作为主线。

我已经按你这个逻辑重新生成了 notebook：

[下载 LC3_TMCN_S1P_Axis.ipynb](sandbox:/mnt/data/LC3_TMCN_S1P_Axis.ipynb)

这个 notebook 的核心比较已经改为：

```text
TMCN_S1P_Axis in Pathology_Tumor
vs
Pathology_Tumor 内部除 TMCN_S1P_Axis 之外的区域
```

也就是说，分析逻辑是：

```text
TuMeNiche 在病理肿瘤区域内部识别出 TMCN_S1P_Axis
        ↓
将该区域与病理肿瘤区域内的非 S1P 区域对比
        ↓
做 S1P 轴基因/score、DEG、ORA/GSEA、细胞组成分析
        ↓
根据实际富集结果判断它是否对应血管生成、内皮迁移、免疫趋化、炎症信号、NFkB/PI3K/MAPK 等医学机制
```

我在 notebook 里避免了直接说“ARI 低”这件事，而是把 Result 10 定位成：

> **在病理注释肿瘤区域内部，TuMeNiche 进一步识别出一个 S1P 脂质信号相关的功能性亚生态位。**

主要模块包括：

1. **Figure 10A**
   左图：LC3 病理 Tumor / Non-Tumor 注释
   右图：只高亮 `TMCN_S1P_Axis`，其他 spots 灰色
   并且沿用你原 LC3 notebook 的 Squidpy 画法，不使用 `palette=dict`。

2. **核心比较组定义**
   `S1P_in_Pathology_Tumor` vs `Tumor_non_S1P`

3. **S1P 轴验证**
   检查 `SPHK1, SPHK2, SPNS2, S1PR1, S1PR2, S1PR3` 以及 `S1P_Axis_Gene_Score`。

4. **DEG 分析**
   使用原始 LC3.h5ad 的全基因表达矩阵，比较：

```text
S1P_in_Pathology_Tumor vs Tumor_non_S1P
```

5. **ORA / 可选 GSEA**
   使用本地 GMT：

```text
/home/zhangjunyi/xiangmu/nichecompass-main/data/pre_data/GMT/MSigDB
```

重点筛选与以下医学机制相关的通路：

```text
angiogenesis
endothelial
vascular development
immune chemotaxis
leukocyte migration
inflammatory response
cytokine signaling
NFkB
PI3K-AKT
MAPK
```

6. **SpaCET 细胞组成异质性分析**
   重点比较病理肿瘤内部 S1P 区域和非 S1P 区域的：

```text
Endothelial
CAF
Macrophage
Macrophage M1/M2
T CD8
T CD8 exhausted
Treg
NK
B cell
Plasma
Malignant
```

7. **空间聚集性 permutation**
   检查 S1P 生态位在病理肿瘤区域内部是否形成空间局部聚集。

8. **自动 summary 表**
   输出：

```text
/home/zhangjunyi/xiangmu/nichecompass-main/outputs/case/LC3_TMCN_S1P_Axis/Result10_LC3_TMCN_S1P_Axis_summary.csv
```

这个版本的设计更适合你论文里的表达：

> TuMeNiche 不只是做肿瘤/非肿瘤划分，而是在病理肿瘤区域内部进一步识别出具有特定代谢轴和功能异质性的 S1P 生态位。
