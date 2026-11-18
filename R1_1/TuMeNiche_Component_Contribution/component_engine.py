"""TuMeNiche reviewer component experiment. Python >=3.9; no model training.

This exact source is embedded in each sample notebook so they are standalone.
"""
import os
import sys
import re
import json
import math
import hashlib
import itertools
import warnings
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom, spearmanr, wilcoxon
from scipy.io import mmwrite
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             jaccard_score, confusion_matrix)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

PROTOCOL_VERSION = "TuMeNiche_component_v1.0"
IMPLEMENTATION_SHA256 = globals().get("EMBEDDED_ENGINE_SHA256")
if IMPLEMENTATION_SHA256 is None:
    IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if "__file__" in globals() else "unrecorded"
AXIS_ORDER = ["Lactate", "Adenosine", "PGE2", "Glutamine", "Succinate",
              "Kynurenine", "ATP", "S1P", "LPA", "Glutamate"]
METHODS = ["FullCoupling", "SenderOnly", "AUCellOnly"]
SCORE_MODES = ["original_rules", "common_scale"]
TUMOR_COLOR = "#D62728"
NON_TUMOR_COLOR = "#1F77B4"


def serializable(x):
    if isinstance(x, dict):
        return {str(k): serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [serializable(v) for v in x]
    if isinstance(x, (np.ndarray, pd.Index)):
        return serializable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(serializable(data), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(frame, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    result = np.full(p.shape, np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    if not len(ix):
        return result
    order = np.argsort(p[ix], kind="mergesort")
    ranked = p[ix][order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) /
                                     np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result[ix[order]] = np.minimum(adjusted, 1.0)
    return result


def hash_values(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def hash_graph(a):
    a = sp.csr_matrix(a, copy=True)
    a.sort_indices()
    h = hashlib.sha256()
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    for v in (a.indptr, a.indices):
        h.update(np.asarray(v, dtype=np.int64).tobytes())
    h.update(np.asarray(a.data, dtype=np.float64).tobytes())
    return h.hexdigest()


def file_record(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path.resolve()), "exists": True,
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def matrix_profile(x):
    values = x.data if sp.issparse(x) else np.asarray(x).reshape(-1)
    step = max(1, int(len(values) // 100000))
    v = np.asarray(values[::step][:100000], dtype=float)
    finite = v[np.isfinite(v)]
    return {"shape": list(x.shape), "type": type(x).__name__,
            "sparse": sp.issparse(x), "sample_size": len(v),
            "sample_all_finite": bool(np.isfinite(v).all()),
            "sample_min": float(finite.min()) if len(finite) else None,
            "sample_max": float(finite.max()) if len(finite) else None,
            "sample_integer_like": bool(np.allclose(finite, np.rint(finite)))
            if len(finite) else None,
            "provenance_note": "数值抽查不能单独证明是未经处理的原始 counts。"}


def initialize_experiment(config, settings, all_sample_ids):
    root = Path(settings["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    for sample_id in all_sample_ids:
        (root / sample_id).mkdir(exist_ok=True)
    (root / "_summary").mkdir(exist_ok=True)
    out = root / config["sample_id"]
    (out / "figures").mkdir(exist_ok=True)
    for key in ("input_h5ad", "final_h5ad", "spacet_prediction_csv",
                "spacet_threshold_csv", "spacet_fraction_csv"):
        # All inputs stay outside the new output tree.
        source = Path(config[key]).resolve()
        if root.resolve() == source or root.resolve() in source.parents:
            raise ValueError("输入文件不能位于新输出目录内部: " + str(source))
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "anndata",
                    "decoupler", "matplotlib", "torch", "nichecompass"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    write_json(out / "run_status.json", {"status": "running",
               "protocol": PROTOCOL_VERSION, "sample_id": config["sample_id"]})
    write_json(out / "run_configuration.json", {"protocol": PROTOCOL_VERSION,
               "engine_sha256": IMPLEMENTATION_SHA256,
               "sample": config, "settings": settings, "versions": versions,
               "python": sys.version,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Arial" if "Arial" in available_fonts
                         else "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300})
    print("输出目录:", out)
    print("GPU 环境沿用原 notebook；本实验复用微簇，无 NicheCompass 训练。")
    return out


def run_stage(out, name, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        path = Path(out) / "run_status.json"
        state = json.loads(path.read_text()) if path.is_file() else {}
        if state.get("status") != "reconstruction_mismatch":
            state.update(status="failed", protocol=PROTOCOL_VERSION, failed_stage=name,
                         error_type=type(error).__name__, error=str(error))
            write_json(path, state)
        raise


def read_axis_table(config):
    path = Path(config["axis_table_path"])
    if path.is_file():
        table = pd.read_csv(path, encoding="utf-8-sig")
        source = str(path)
    else:
        raise FileNotFoundError("四元组文件未找到，检查已配置路径: " + str(path))
    for key in ("TMCN_Name", "Source_Pathways", "Source_Genes", "Target_Genes"):
        if key not in table:
            raise KeyError("四元组缺少列: " + key)
        table[key] = table[key].fillna("").astype(str).str.strip()
    table["axis"] = table.TMCN_Name.str.replace("TMCN_", "", regex=False).str.replace(
        "_Axis", "", regex=False)
    if table.axis.duplicated().any() or set(table.axis) != set(AXIS_ORDER):
        raise ValueError("要求与原实验相同的 10 条轴；实际为 " + repr(table.axis.tolist()))
    table = table.set_index("axis").loc[AXIS_ORDER].reset_index()
    return table, source


def split_genes(value):
    return list(dict.fromkeys(x.strip() for x in str(value).split(",") if x.strip()))


def label_to_axes(label):
    value = str(label)
    inactive = {"Healthy_Region", "TMCN_Transitional", "TMCN_Quiescent",
                "Transitional", "Quiescent (Background)"}
    if value in inactive or value.startswith("Quiescent"):
        return ()
    found = tuple(a for a in AXIS_ORDER if re.search(
        r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", value))
    if not found:
        raise ValueError("无法识别既有生态位标签，不能将其静默当作非活跃: " + value)
    return found


def canonical_labels(axis_sets, spacet_tumor):
    level1, level2 = [], []
    for axes, tumor in zip(axis_sets, spacet_tumor):
        active = len(axes) > 0
        level1.append("Tumor_Region" if tumor or active else "Healthy_Region")
        if len(axes) == 1:
            level2.append("TMCN_" + axes[0] + "_Axis")
        elif len(axes) > 1:
            level2.append("Multi_Axis_" + "_".join(axes))
        else:
            level2.append("TMCN_Transitional" if tumor else "Healthy_Region")
    return np.asarray(level1), np.asarray(level2)


def load_frozen_inputs(config, settings, out):
    import anndata as ad
    needed = ["input_h5ad", "final_h5ad", "spacet_prediction_csv",
              "spacet_threshold_csv", "spacet_fraction_csv"]
    records = {key: file_record(config[key]) for key in needed}
    write_json(out / "input_files.json", records)
    missing = [v["path"] for v in records.values() if not v["exists"]]
    if missing:
        raise FileNotFoundError("输入文件缺失；不自动替换不同版本:\n" + "\n".join(missing))
    frozen = ad.read_h5ad(config["final_h5ad"])
    full = ad.read_h5ad(config["input_h5ad"])
    if not frozen.obs_names.is_unique or not full.obs_names.is_unique:
        raise ValueError("存在重复 spot ID，无法做严格一对一配对。")
    if not full.var_names.is_unique:
        raise ValueError("完整表达的基因名重复；需沿用原数据的明确合并规则。")
    ids = frozen.obs_names.copy()
    missing_spots = ids.difference(full.obs_names)
    if len(missing_spots):
        raise ValueError("最终结果中有 spot 不在完整表达文件中: " + repr(missing_spots[:10].tolist()))
    cluster_key = config["native_cluster_key"]
    if cluster_key not in frozen.obs or frozen.obs[cluster_key].isna().any():
        raise KeyError("最终结果未保存完整微簇列 " + cluster_key + "；需定位已有的正确结果文件。")
    if config["graph_key"] not in frozen.obsp:
        raise KeyError("未保存原空间图；实验不自行重建不同的图。")
    graph = sp.csr_matrix(frozen.obsp[config["graph_key"]], copy=True)
    if graph.shape != (len(ids), len(ids)) or not np.isfinite(graph.data).all() or (graph.data < 0).any():
        raise ValueError("空间图形状、有限性或非负性不符合要求。")
    if "spatial" not in frozen.obsm or np.asarray(frozen.obsm["spatial"]).shape[1] < 2:
        raise KeyError("缺少 obsm['spatial'] 坐标。")
    if not np.isfinite(np.asarray(frozen.obsm["spatial"])[:, :2]).all():
        raise ValueError("空间坐标包含非有限值。")
    if config["malignant_column"] not in frozen.obs:
        raise KeyError("最终结果缺少恶性分数 " + config["malignant_column"])
    malignant = pd.to_numeric(frozen.obs[config["malignant_column"]], errors="coerce").to_numpy(float)
    if not np.isfinite(malignant).all():
        raise ValueError("冻结的恶性分数存在缺失或非有限值。")
    pred = pd.read_csv(config["spacet_prediction_csv"], index_col=0)
    pred.index = pred.index.astype(str)
    if not pred.index.is_unique or len(ids.difference(pred.index)):
        raise ValueError("SpaCET 预测 CSV 的 spot ID 不唯一或未覆盖最终对象。")
    columns = [c for c in pred if c.startswith("Unsupervised_Pred_")]
    if len(columns) != 1:
        raise ValueError("需唯一 SpaCET 预测列；找到 " + repr(columns))
    pred = pred.reindex(ids)
    labels = pred[columns[0]].astype(str).str.strip()
    if not set(labels).issubset({"Predicted_Tumor", "Predicted_Non_Tumor"}):
        raise ValueError("初始预测包含未知标签: " + repr(set(labels)))
    spacet_tumor = labels.eq("Predicted_Tumor").to_numpy()
    if "Malignant" in pred and not np.allclose(pd.to_numeric(pred.Malignant).to_numpy(),
                                             malignant, atol=1e-6, rtol=1e-5):
        raise ValueError("预测 CSV 的恶性分数与最终对象不一致；避免混用版本。")
    threshold_table = pd.read_csv(config["spacet_threshold_csv"])
    threshold_values = pd.to_numeric(threshold_table.loc[
        threshold_table.Metric.eq("Optimal_GMM_Threshold"), "Value"], errors="coerce")
    if len(threshold_values) != 1 or not np.isfinite(threshold_values.iloc[0]):
        raise ValueError("GMM 阈值 CSV 未包含唯一有效 Optimal_GMM_Threshold。")
    tumor_threshold = float(threshold_values.iloc[0])
    if not 0 <= tumor_threshold <= 1:
        raise ValueError("SpaCET 阈值不在 [0,1]。")
    # The original code uses >= at the GMM split.
    threshold_match = np.mean((malignant >= tumor_threshold) == spacet_tumor)
    gt_key = config["pathology_column"]
    if gt_key not in frozen.obs:
        raise KeyError("缺少原病理字符串列 " + gt_key)
    original_gt = frozen.obs[gt_key]
    gt_text = original_gt.astype(str).str.strip()
    if config.get("gt_rule") == "prostate_invasive_vs_other":
        invalid = original_gt.isna() | gt_text.isin(["", "nan", "None", "NA", "exclude", "UNASSIGNED"])
        gt = pd.Series(np.where(gt_text.eq("Invasive carcinoma"), 1.0, 0.0), index=ids)
        gt.loc[invalid] = np.nan
    else:
        gt = gt_text.map(config["gt_map"]).astype(float)
    valid = gt.notna().to_numpy()
    if not valid.any():
        raise ValueError("病理映射无有效 spot，停止评价。")
    axis_table, prior_source = read_axis_table(config)
    score_cols = [a + suffix for a in AXIS_ORDER for suffix in ("_Sender_Score", "_Receiver_Score")]
    missing_cols = [c for c in score_cols if c not in frozen.obs]
    if missing_cols:
        raise KeyError("最终对象缺少既有评分，优先检查结果版本: " + repr(missing_cols))
    # Keep original numeric dtypes: the uploaded code uses each stored column's
    # values directly, and float32 -> float64 can slightly change sparse sums.
    sender = frozen.obs[[a + "_Sender_Score" for a in AXIS_ORDER]].to_numpy()
    receiver = frozen.obs[[a + "_Receiver_Score" for a in AXIS_ORDER]].to_numpy()
    if not np.isfinite(sender).all() or not np.isfinite(receiver).all() or min(sender.min(), receiver.min()) < 0:
        raise ValueError("既有发送端/接收端评分非有限或为负，不能静默修正冻结输入。")
    original_key = "Level2_Micro_Niche" if "Level2_Micro_Niche" in frozen.obs else "Final_Niche_Type"
    if original_key not in frozen.obs:
        raise KeyError("缺少既有生态位标签，无法复核完整方案。")
    sets = [label_to_axes(x) for x in frozen.obs[original_key].astype(str)]
    reconstructed_l1, original_l2 = canonical_labels(sets, spacet_tumor)
    if "Level1_Macro_Region" in frozen.obs:
        original_l1 = frozen.obs.Level1_Macro_Region.astype(str).to_numpy()
        if not set(original_l1).issubset({"Tumor_Region", "Healthy_Region"}):
            raise ValueError("未知既有 Level 1 标签。")
    else:
        original_l1 = reconstructed_l1
    freeze = {"spots": hash_values(ids),
              "clusters": hash_values(frozen.obs[cluster_key].astype(str)),
              "graph": hash_graph(graph),
              "malignant": hashlib.sha256(malignant.astype(np.float64).tobytes()).hexdigest()}
    aligned_full = full[ids].copy()
    ctx = dict(config=config, settings=settings, out=out, frozen=frozen, full=full,
               aligned_full=aligned_full, ids=ids, clusters=frozen.obs[cluster_key].astype(str).to_numpy(),
               graph=graph, malignant=malignant, sender=sender, receiver=receiver,
               spacet_tumor=spacet_tumor, tumor_threshold=tumor_threshold,
               low_gate=max(0.0, tumor_threshold - config["gate_offset"]),
               gt=gt.to_numpy(float), valid_gt=valid, gt_text=gt_text.to_numpy(),
               original_l1=original_l1, original_l2=original_l2,
               axis_table=axis_table, freeze=freeze)
    audit = {"files": records, "complete_shape": full.shape, "frozen_shape": frozen.shape,
             "full_spots_excluded": len(full.obs_names.difference(ids)),
             "valid_gt_spots": int(valid.sum()), "unmapped_gt_counts": gt_text[~valid].value_counts().to_dict(),
             "pathology_column": gt_key, "evaluation_map": config.get("gt_map", config.get("gt_rule")),
             "X_profile": matrix_profile(full.X), "layers": list(full.layers),
             "raw_available": full.raw is not None, "log1p_metadata": full.uns.get("log1p"),
             "freeze": freeze, "prior_source": prior_source,
             "prior_sha256": hashlib.sha256(Path(prior_source).read_bytes()).hexdigest(),
             "spacet_prediction_column": columns[0], "tumor_threshold": tumor_threshold,
             "threshold_prediction_agreement": float(threshold_match),
             "low_malignancy_gate": ctx["low_gate"],
             "original_union_agreement": float(np.mean(original_l1 == reconstructed_l1)),
             "fixed_level2_roi": "initial SpaCET Predicted_Tumor, independent of score method"}
    write_json(out / "input_audit.json", audit)
    write_csv(pd.DataFrame({"spot_id": ids, "native_cluster": ctx["clusters"],
                           "malignant": malignant, "spacet_tumor": spacet_tumor,
                           "pathology": gt_text.to_numpy(), "pathology_binary": ctx["gt"],
                           "valid_for_paired_metrics": valid}), out / "frozen_spots.csv")
    write_csv(axis_table, out / "axis_prior_used.csv")
    if audit["original_union_agreement"] < 1:
        raise ValueError("既有 Level 1 不满足 SpaCET 肿瘤 OR 活跃生态位；见 input_audit.json。")
    print("冻结:", len(ids), "spots;", len(np.unique(ctx["clusters"])), "microclusters; low gate", ctx["low_gate"])
    return ctx


def recover_enzyme_scores(ctx):
    full, frozen = ctx["full"], ctx["frozen"]
    records = []
    for row in ctx["axis_table"].itertuples():
        source, target = split_genes(row.Source_Genes), split_genes(row.Target_Genes)
        source_present = [g for g in source if g in full.var_names]
        target_present = [g for g in target if g in full.var_names]
        records.append(dict(axis=row.axis, source_genes=";".join(source),
                            source_present=";".join(source_present), source_n=len(source_present),
                            source_missing=";".join(g for g in source if g not in full.var_names),
                            target_present=";".join(target_present), target_n=len(target_present)))
    write_csv(pd.DataFrame(records), ctx["out"] / "gene_coverage.csv")
    key = "aucell_estimate"
    holder = None
    if key in full.obsm:
        holder, reason = full, "complete_h5ad.obsm.aucell_estimate"
    elif key in frozen.obsm and len(full) == len(frozen) and set(full.obs_names) == set(frozen.obs_names):
        holder, reason = frozen, "frozen_h5ad.obsm.aucell_estimate; full spot set preserved"
    if holder is not None and not ctx["settings"].get("force_recompute_aucell", False):
        raw = holder.obsm[key]
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("aucell_estimate 缺少轴名称；不能猜测矩阵列序。启用 force_recompute_aucell。")
        raw = raw.reindex(index=holder.obs_names, columns=AXIS_ORDER)
    else:
        import decoupler as dc
        if not hasattr(dc, "run_aucell"):
            raise RuntimeError("需与原代码兼容的 decoupler 1.x run_aucell API；不自动替换评分实现。")
        net = pd.DataFrame([{"source": row.axis, "target": gene}
                            for row in ctx["axis_table"].itertuples()
                            for gene in split_genes(row.Source_Genes) if gene in full.var_names])
        # Same full gene universe, original X and original min_n=1; no HVG restriction.
        dc.run_aucell(mat=full, net=net, source="source", target="target",
                      min_n=1, verbose=False, use_raw=False)
        raw = full.obsm[key].reindex(columns=AXIS_ORDER)
        holder, reason = full, "recomputed_original_decoupler_AUCell_on_complete_X"
    absent_axes = raw.columns[raw.isna().all()].tolist()
    if raw.isna().any().any() and set(raw.columns[raw.isna().any()]) != set(absent_axes):
        raise ValueError("AUCell 输出存在部分缺失值。")
    raw = raw.fillna(0.0)
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("AUCell 输出包含非有限值。")
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(raw), index=raw.index, columns=AXIS_ORDER)
    aligned = scaled.reindex(ctx["ids"])
    if aligned.isna().any().any():
        raise ValueError("AUCell 评分与冻结 spot 无法对应。")
    write_json(ctx["out"] / "aucell_provenance.json", {"source": reason,
               "scaling_spot_count": len(raw), "unavailable_axes": absent_axes,
               "source_set": "same 10 Source_Genes sets; not original scMetabolism gene sets",
               "parameters": {"min_n": 1, "use_raw": False, "normalization": "per-axis MinMax before spot alignment"}})
    return aligned.to_numpy(float)


def construct_scores(ctx, enzyme):
    neighbor_receiver = ctx["graph"].dot(ctx["receiver"])
    scores = {"FullCoupling": np.log1p(ctx["sender"] * neighbor_receiver),
              "SenderOnly": np.log1p(ctx["sender"]), "AUCellOnly": np.asarray(enzyme, float)}
    for key, x in scores.items():
        if x.shape != (len(ctx["ids"]), len(AXIS_ORDER)) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("非法评分矩阵: " + key)
    raw_checks = []
    for j, axis in enumerate(AXIS_ORDER):
        key = "Coupling_" + axis
        if key in ctx["frozen"].obs:
            old = ctx["frozen"].obs[key].to_numpy(float)
            raw_checks.append({"axis": axis, "max_abs_error_vs_saved_coupling": float(np.max(np.abs(old - scores["FullCoupling"][:, j])))})
    write_csv(pd.DataFrame(raw_checks, columns=["axis", "max_abs_error_vs_saved_coupling"]),
              ctx["out"] / "coupling_numeric_reconstruction.csv")
    return scores


def common_scale(scores):
    """Positive multiplicative scaling preserves zeros and within-method mean ratios."""
    maxima = np.max(scores, axis=0)
    return np.divide(scores, maxima, out=np.zeros_like(scores), where=maxima > 0), maxima


def learn_threshold(scores, seed=42, absolute_floor=0.15):
    scores = np.asarray(scores, dtype=float)
    positive = scores[np.isfinite(scores) & (scores > 0)].reshape(-1, 1)
    high, quiet, method = None, None, "Robust Dynamic Fallback"
    if len(positive) > 50:
        try:
            gmm = GaussianMixture(n_components=2, random_state=seed)
            labels = gmm.fit_predict(positive)
            m0, m1 = gmm.means_[0, 0], gmm.means_[1, 0]
            s0, s1 = np.sqrt(gmm.covariances_[0, 0, 0]), np.sqrt(gmm.covariances_[1, 0, 0])
            if abs(m0 - m1) > 0.8 * np.std(positive):
                upper = 0 if m0 > m1 else 1
                high = float(np.min(positive[labels == upper]))
                quiet = float((m1 + s1) if upper == 0 else (m0 + s0))
                method = "GMM (2-Components)"
        except (ValueError, FloatingPointError):
            pass
    if high is None:
        high = float(max(np.mean(scores) + 1.5 * np.std(scores), np.quantile(scores, 0.95)))
        quiet = float(np.mean(scores))
    suppressed = high < absolute_floor
    if suppressed:
        high = np.inf
    return {"T_high": high, "T_quiet": quiet, "Method": method,
            "floor_suppressed": suppressed}


def annotate_scores(scores, clusters, malignant, low_gate, spacet_tumor,
                    seed=42, absolute_floor=0.15):
    """Exact active/quiescent/gate logic from the uploaded Cell 7, no retraining."""
    n, n_axes = scores.shape
    thresholds = [learn_threshold(scores[:, j], seed, absolute_floor) for j in range(n_axes)]
    active = np.column_stack([scores[:, j] >= t["T_high"] for j, t in enumerate(thresholds)])
    quiet = np.column_stack([scores[:, j] <= t["T_quiet"] for j, t in enumerate(thresholds)])
    groups = sorted(np.unique(clusters))
    bg = active.sum(0)
    bg_quiet = quiet.sum(0)
    group_masks = [np.asarray(clusters) == group for group in groups]
    sizes = np.array([x.sum() for x in group_masks])
    counts = np.vstack([active[mask].sum(0) for mask in group_masks])
    quiet_counts = np.vstack([quiet[mask].sum(0) for mask in group_masks])
    p = hypergeom.sf(counts - 1, n, bg, sizes[:, None])
    p[counts == 0] = 1.0
    q = np.column_stack([bh_adjust(p[:, j]) for j in range(n_axes)])
    final_axes = [() for _ in range(n)]
    pre_axes = [() for _ in range(n)]
    cluster_rows, filter_rows = [], []
    for i, (group, mask) in enumerate(zip(groups, group_masks)):
        frac = counts[i] / sizes[i]
        fc = frac / (bg / n + 1e-9)
        means = scores[mask].mean(0)
        intensity = means / (scores.mean(0) + 1e-9)
        passed = (fc >= 1.30) & (counts[i] >= 10) & (intensity >= 1.20) & (q[i] < 0.05)
        cold = (quiet_counts[i] / sizes[i] > bg_quiet / n) | (fc < 0.5)
        axes = tuple(AXIS_ORDER[j] for j in range(n_axes) if passed[j])
        for j in range(n_axes):
            cluster_rows.append(dict(cluster=group, axis=AXIS_ORDER[j], n_spots=int(sizes[i]),
                active_n=int(counts[i, j]), active_fraction=frac[j], fc=fc[j],
                intensity_ratio=intensity[j], p=p[i, j], fdr=q[i, j], passed=bool(passed[j]),
                cold=bool(cold[j])))
        if axes:
            original_ix = np.flatnonzero(mask)
            keep_ix = original_ix[np.asarray(malignant)[mask] >= low_gate]
            for ix in original_ix:
                pre_axes[ix] = axes
            remain_fraction = len(keep_ix) / len(original_ix)
            cancelled = len(keep_ix) < 10 or remain_fraction < 0.50
            if not cancelled:
                for ix in keep_ix:
                    final_axes[ix] = axes
            filter_rows.append(dict(cluster=group, axes=";".join(axes), original_n=len(original_ix),
                 remain_n=len(keep_ix), remain_fraction=remain_fraction, cancelled=cancelled))
    level1, level2 = canonical_labels(final_axes, spacet_tumor)
    return dict(scores=scores, active=active, quiet=quiet, final_axes=final_axes,
                pre_axes=pre_axes, level1=level1, level2=level2,
                thresholds=pd.DataFrame([dict(axis=AXIS_ORDER[j], **t) for j, t in enumerate(thresholds)]),
                cluster_stats=pd.DataFrame(cluster_rows),
                filter_stats=pd.DataFrame(filter_rows, columns=["cluster", "axes", "original_n", "remain_n", "remain_fraction", "cancelled"]))


def binary_metrics(gt, prediction, valid):
    y = np.asarray(gt)[valid].astype(int)
    p = np.asarray(prediction)[valid].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(n_valid=len(y), gt_classes=len(np.unique(y)),
                ARI=adjusted_rand_score(y, p), NMI=normalized_mutual_info_score(y, p),
                F1=f1_score(y, p, zero_division=0), IoU=jaccard_score(y, p, zero_division=0),
                Precision=precision_score(y, p, zero_division=0),
                Recall=recall_score(y, p, zero_division=0), Accuracy=accuracy_score(y, p),
                BalancedAccuracy=balanced_accuracy_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def run_comparisons(ctx, raw_scores):
    rows, results, scales = [], {}, []
    baseline = ctx["spacet_tumor"]
    base_metrics = binary_metrics(ctx["gt"], baseline, ctx["valid_gt"])
    rows.append(dict(method="SpaCET_GMM", score_mode="reference", **base_metrics))
    rows.append(dict(method="SavedFull", score_mode="reference", **binary_metrics(
        ctx["gt"], ctx["original_l1"] == "Tumor_Region", ctx["valid_gt"])))
    for mode in SCORE_MODES:
        for method in METHODS:
            values = raw_scores[method]
            if mode == "common_scale":
                values, maxima = common_scale(values)
                scales.extend(dict(method=method, axis=axis, divisor=maxima[j])
                              for j, axis in enumerate(AXIS_ORDER))
            result = annotate_scores(values, ctx["clusters"], ctx["malignant"], ctx["low_gate"],
                                     baseline, seed=ctx["settings"]["seed"])
            result["method"], result["score_mode"] = method, mode
            results[(mode, method)] = result
            rows.append(dict(method=method, score_mode=mode, **binary_metrics(
                ctx["gt"], result["level1"] == "Tumor_Region", ctx["valid_gt"])))
            for field, filename in [("thresholds", "thresholds"), ("cluster_stats", "cluster_statistics"),
                                    ("filter_stats", "malignancy_gate")]:
                write_csv(result[field], ctx["out"] / (mode + "__" + method + "__" + filename + ".csv"))
    raw_full = results[("original_rules", "FullCoupling")]
    mismatch1 = raw_full["level1"] != ctx["original_l1"]
    mismatch2 = raw_full["level2"] != ctx["original_l2"]
    reconstruction = {"level1_mismatch_n": int(mismatch1.sum()), "level2_mismatch_n": int(mismatch2.sum()),
                      "level1_agreement": float(np.mean(~mismatch1)),
                      "level2_agreement": float(np.mean(~mismatch2)),
                      "status": "matched" if not (mismatch1.any() or mismatch2.any()) else "mismatch",
                      "note": "比较规范化后的活跃轴集合与 Level1/2；多轴排列次序不构成差异。"}
    write_json(ctx["out"] / "full_reconstruction_check.json", reconstruction)
    write_csv(pd.DataFrame(dict(spot_id=ctx["ids"][mismatch1 | mismatch2],
        saved_level2=ctx["original_l2"][mismatch1 | mismatch2],
        reconstructed_level2=raw_full["level2"][mismatch1 | mismatch2])), ctx["out"] / "full_reconstruction_mismatches.csv")
    metrics = pd.DataFrame(rows)
    metrics.insert(0, "sample_id", ctx["config"]["sample_id"])
    metrics.insert(1, "cancer_type", ctx["config"]["cancer_type"])
    for metric in ["ARI", "F1", "IoU", "Recall", "Precision"]:
        metrics["delta_" + metric + "_vs_SpaCET"] = metrics[metric] - base_metrics[metric]
    write_csv(metrics, ctx["out"] / "level1_metrics.csv")
    write_csv(pd.DataFrame(scales), ctx["out"] / "common_scale_divisors.csv")
    transitions = []
    for key, result in results.items():
        p = result["level1"] == "Tumor_Region"
        transitions.append(dict(score_mode=key[0], method=key[1],
             additions_outside_initial_tumor=int((p & ~baseline).sum()),
             removed_initial_tumor=int((~p & baseline).sum()),
             added_pathology_tumor=int((p & ~baseline & (ctx["gt"] == 1)).sum()),
             added_pathology_nontumor=int((p & ~baseline & (ctx["gt"] == 0)).sum()),
             added_unmapped_pathology=int((p & ~baseline & ~ctx["valid_gt"]).sum())))
        if np.any(~p & baseline):
            raise AssertionError("Level 1 并集逻辑错误：删除了初始肿瘤 spot。")
    write_csv(pd.DataFrame(transitions), ctx["out"] / "level1_added_spots.csv")
    if reconstruction["status"] != "matched" and ctx["settings"]["strict_reconstruction"]:
        write_json(ctx["out"] / "run_status.json", {"status": "reconstruction_mismatch",
            "engine_sha256": IMPLEMENTATION_SHA256,
            "protocol": PROTOCOL_VERSION, "sample_id": ctx["config"]["sample_id"],
            "reconstruction": reconstruction,
            "note": "原始预测与保存的最终预测配对表仍可汇总；评分组成部分推断暂停。"})
        raise RuntimeError("原始规则未复现保存的完整方案；已输出差异表。先核对版本/门控/四元组，不能将差异归因于评分。")
    return results, metrics, reconstruction


def mode_label(values):
    values, counts = np.unique(np.asarray(values, dtype=str), return_counts=True)
    return values[np.argmax(counts)]


def association_diagnostics(values, labels):
    """Descriptive association on frozen microcluster means; no spot-level p values."""
    values, labels = np.asarray(values, float), np.asarray(labels, str)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    n, k = len(values), len(np.unique(labels))
    result = {"n_microclusters": n, "n_labels": k, "eta2": np.nan,
              "adjusted_eta2": np.nan, "loco_R2": np.nan}
    if n < 5 or np.var(values) <= 1e-15:
        return result
    total = np.sum((values - values.mean()) ** 2)
    fitted = np.array([values[labels == label].mean() for label in labels])
    residual = np.sum((values - fitted) ** 2)
    result["eta2"] = 1.0 - residual / total
    if n > k:
        result["adjusted_eta2"] = 1.0 - (residual / (n - k)) / (total / (n - 1))
    predictions = []
    for i in range(n):
        train = np.arange(n) != i
        same = train & (labels == labels[i])
        predictions.append(values[same].mean() if same.any() else values[train].mean())
    result["loco_R2"] = 1.0 - np.sum((values - predictions) ** 2) / total
    return result


def get_validation_features(ctx):
    path = ctx["config"]["spacet_fraction_csv"]
    fractions = pd.read_csv(path, index_col=0)
    fractions.index = fractions.index.astype(str)
    if not fractions.index.is_unique or len(ctx["ids"].difference(fractions.index)):
        raise ValueError("SpaCET 细胞组成表无法严格对应冻结 spot。")
    fractions = fractions.reindex(ctx["ids"])
    selected, excluded = {}, []
    # Fractions come from the original SpaCET output, not arbitrary numeric obs columns.
    for column in fractions:
        if re.search(r"malignan|tumou?r|cancer|unidentif|unknown|unassign", column, re.I):
            excluded.append({"feature": column, "reason": "malignancy/gating or unidentified"})
            continue
        values = pd.to_numeric(fractions[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or values.std() < 1e-8:
            excluded.append({"feature": column, "reason": "missing/nonnumeric/constant"})
            continue
        if values.min() < -1e-6 or values.max() > 1 + 1e-6:
            excluded.append({"feature": column, "reason": "not a fraction in [0,1]"})
            continue
        selected["SpaCET::" + column] = values.to_numpy(float)
    write_csv(pd.DataFrame(excluded, columns=["feature", "reason"]), ctx["out"] / "validation_feature_exclusions.csv")
    features = pd.DataFrame(selected, index=ctx["ids"])
    write_json(ctx["out"] / "validation_status.json", {
        "cell_fraction_features": list(features),
        "interpretation": "同一转录组推断的非恶性细胞组成关联；不是独立实验或代谢通量验证。",
        "source_genes_used_as_validation": False,
        "optional_gmt": ctx["settings"].get("validation_gmt")})
    return features


def optional_gmt_features(ctx):
    """Optional validation removes direct prior genes AND PROGENy footprint genes."""
    gmt = ctx["settings"].get("validation_gmt")
    if not gmt:
        return pd.DataFrame(index=ctx["ids"])
    gmt = Path(gmt)
    if not gmt.is_file():
        raise FileNotFoundError("指定的验证 GMT 不存在: " + str(gmt))
    cached = ctx["settings"].get("progeny_net_csv")
    if cached:
        net = pd.read_csv(cached)
    else:
        import decoupler as dc
        if not hasattr(dc, "get_progeny"):
            raise RuntimeError("GMT 验证需要原 PROGENy top=500 footprint，不能仅排除四元组基因。")
        net = dc.get_progeny(organism="human", top=500)
    if "target" not in net:
        raise ValueError("PROGENy 网络缺少 target 列。")
    excluded = set(net.target.astype(str))
    for row in ctx["axis_table"].itertuples():
        excluded.update(split_genes(row.Source_Genes))
        excluded.update(split_genes(row.Target_Genes))
    write_csv(pd.DataFrame({"gene": sorted(excluded)}), ctx["out"] / "validation_excluded_input_genes.csv")
    data = ctx["aligned_full"]
    matrix = data.layers["counts"] if "counts" in data.layers else data.X
    # Rank-derived validation means: use local per-gene standardization; no AUC reuse.
    output, coverage = {}, []
    var_index = {str(g): i for i, g in enumerate(data.var_names)}
    for line in gmt.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        name, genes = fields[0], set(fields[2:])
        keep = sorted((genes - excluded) & set(var_index))
        coverage.append({"gene_set": name, "original_n": len(genes), "retained_n": len(keep)})
        if len(keep) < ctx["settings"]["validation_min_genes"]:
            continue
        sub = matrix[:, [var_index[g] for g in keep]]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        if not np.isfinite(sub).all() or (sub < 0).any():
            raise ValueError("验证表达包含非有限/负值，不能应用非负表达转换。")
        # Descriptive held-out signature, not a reproduction of GSEA or flux.
        sub = sub.astype(float)
        if "counts" in data.layers:
            library = np.asarray(matrix.sum(1)).reshape(-1)
            factors = np.divide(1e4, library, out=np.zeros_like(library, dtype=float), where=library > 0)
            sub = np.log1p(sub * factors[:, None])
        # Without an explicit counts layer, retain the original X transformation
        # instead of guessing that X is raw or applying a second log transform.
        sd = sub.std(0)
        z = np.divide(sub - sub.mean(0), sd, out=np.zeros_like(sub), where=sd > 0)
        output["GMT::" + name] = z.mean(1)
    write_csv(pd.DataFrame(coverage), ctx["out"] / "validation_gmt_coverage.csv")
    return pd.DataFrame(output, index=ctx["ids"])


def spatial_description(labels, graph, roi):
    y = np.asarray(labels)[roi]
    a = graph[roi][:, roi].tocsr(copy=True)
    a.setdiag(0)
    a.eliminate_zeros()
    row, col = a.nonzero()
    if a.nnz:
        weight = a.data
        observed = float(np.sum(weight * (y[row] == y[col])) / weight.sum())
        categories = np.unique(y)
        degree = np.asarray(a.sum(1)).reshape(-1)
        marginal = np.array([degree[y == cat].sum() / degree.sum() for cat in categories])
        expected = float(np.sum(marginal ** 2))
        adjusted = (observed - expected) / (1 - expected) if expected < 1 else np.nan
    else:
        observed = adjusted = np.nan
    return {"n_roi_spots": len(y), "n_labels": len(np.unique(y)),
            "neighbor_agreement": observed, "chance_adjusted_neighbor_agreement": adjusted}


def evaluate_level2(ctx, results):
    # ROI cannot expand/contract with each method: initial SpaCET tumor spots only.
    roi = ctx["spacet_tumor"].copy()
    features = get_validation_features(ctx)
    optional = optional_gmt_features(ctx)
    features = features.join(optional)
    group_ids = sorted(np.unique(ctx["clusters"][roi]))
    cluster_masks = [(ctx["clusters"] == group) & roi for group in group_ids]
    target_means = pd.DataFrame({c: [features.loc[mask, c].mean() for mask in cluster_masks]
                                 for c in features}, index=group_ids)
    target_means.index.name = "native_cluster"
    target_means.to_csv(ctx["out"] / "validation_microcluster_means.csv", encoding="utf-8-sig")
    associations, summary, overlaps, transitions = [], [], [], []
    for (mode, method), result in results.items():
        axes_mask = np.array([[a in axes for a in AXIS_ORDER] for axes in result["final_axes"]])
        description = spatial_description(result["level2"], ctx["graph"], roi)
        description.update(score_mode=mode, method=method,
            active_fraction_in_fixed_roi=float(axes_mask[roi].any(1).mean()) if roi.any() else np.nan,
            n_active_spots_total=int(axes_mask.any(1).sum()))
        categories = [mode_label(result["level2"][mask]) for mask in cluster_masks]
        for column in features:
            diagnostic = association_diagnostics(target_means[column].to_numpy(), categories)
            associations.append(dict(score_mode=mode, method=method, feature=column, **diagnostic))
        summary.append(description)
    for mode in SCORE_MODES:
        full = results[(mode, "FullCoupling")]
        for method in ["SenderOnly", "AUCellOnly"]:
            simple = results[(mode, method)]
            for axis in AXIS_ORDER:
                f = np.array([axis in axes for axes in full["final_axes"]])[roi]
                s = np.array([axis in axes for axes in simple["final_axes"]])[roi]
                union = (f | s).sum()
                overlaps.append(dict(score_mode=mode, comparison="FullCoupling_vs_" + method,
                    axis=axis, n_full=int(f.sum()), n_simple=int(s.sum()),
                    n_intersection=int((f & s).sum()),
                    Jaccard=float((f & s).sum() / union) if union else np.nan,
                    full_only=int((f & ~s).sum()), simple_only=int((s & ~f).sum())))
            table = pd.crosstab(pd.Series(full["level2"][roi], name="FullCoupling"),
                                pd.Series(simple["level2"][roi], name=method))
            table.to_csv(ctx["out"] / (mode + "__Level2_transition__" + method + ".csv"), encoding="utf-8-sig")
    assoc = pd.DataFrame(associations, columns=["score_mode", "method", "feature", "n_microclusters",
                          "n_labels", "eta2", "adjusted_eta2", "loco_R2"])
    summaries = pd.DataFrame(summary)
    for metric in ("adjusted_eta2", "loco_R2"):
        if len(assoc):
            median = assoc[assoc.feature.str.startswith("SpaCET::")].groupby(["score_mode", "method"])[metric].median()
            summaries["median_cell_fraction_" + metric] = [median.get((r.score_mode, r.method), np.nan)
                                                          for r in summaries.itertuples()]
        else:
            summaries["median_cell_fraction_" + metric] = np.nan
    summaries.insert(0, "sample_id", ctx["config"]["sample_id"])
    write_csv(assoc, ctx["out"] / "level2_validation_features.csv")
    write_csv(summaries, ctx["out"] / "level2_summary.csv")
    write_csv(pd.DataFrame(overlaps), ctx["out"] / "level2_axis_overlap.csv")
    return summaries, assoc, features


R_SCMETABOLISM_SCRIPT = r'''
args <- commandArgs(trailingOnly=TRUE)
work <- args[1]
set.seed(as.integer(args[2]))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(scMetabolism))
mat <- Matrix::readMM(file.path(work, "expression.mtx"))
rownames(mat) <- readLines(file.path(work, "genes.txt"), warn=FALSE)
colnames(mat) <- readLines(file.path(work, "spots.txt"), warn=FALSE)
input <- as.data.frame(as.matrix(mat), optional=TRUE)
colnames(input) <- colnames(mat)
score <- scMetabolism::sc.metabolism(countexp=input,
            method="AUCell", imputation=FALSE, ncores=1, metabolism.type="KEGG")
spots <- colnames(mat)
# The official wrapper uses data.frame(), which may apply make.names() to
# Visium barcodes containing '-'. Restore only after verifying the column order.
if (!identical(colnames(score), spots) &&
    !identical(colnames(score), make.names(spots, unique=TRUE))) {
    stop("Unexpected AUCell cell order/names; cannot guess barcode alignment")
}
colnames(score) <- spots
score <- as.data.frame(t(score), check.names=FALSE)
write.csv(score, file.path(work, "scMetabolism_KEGG_AUCell_scores.csv"), quote=TRUE)
sink(file.path(work, "R_sessionInfo.txt")); print(sessionInfo()); sink()
'''


def run_scmetabolism_reference(ctx, results):
    """Actual R package, optional; never relabel a custom gene-set score as scMetabolism."""
    directory = ctx["out"] / "scMetabolism_reference"
    directory.mkdir(exist_ok=True)
    # Avoid presenting results left by a previous run when this run is skipped.
    for stale in ["scMetabolism_KEGG_AUCell_scores.csv", "predefined_four_pathway_mean.csv",
                  "descriptive_score_correlations.csv", "four_pathway_activity.png", "R_sessionInfo.txt"]:
        (directory / stale).unlink(missing_ok=True)
    status = {"status": "not_run", "implementation": "original R scMetabolism::sc.metabolism",
              "method": "AUCell", "gene_sets": "package KEGG", "imputation": False,
              "used_for_10_axis_labels": False}
    def finish(reason, state="skipped"):
        status.update(status=state, reason=reason)
        write_json(directory / "status.json", status)
        print("scMetabolism reference:", state, reason)
        return status
    if not ctx["settings"].get("run_scmetabolism_reference", True):
        return finish("disabled in settings")
    executable = shutil.which(ctx["settings"].get("rscript", "Rscript"))
    if not executable:
        return finish("Rscript not found; core 10-axis ablation remains valid")
    probe = subprocess.run([executable, "-e", 'stopifnot(all(vapply(c("Matrix","scMetabolism","AUCell","GSEABase"),requireNamespace,logical(1),quietly=TRUE)))'],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode:
        (directory / "dependency_check.log").write_text(probe.stdout + probe.stderr)
        return finish("required R packages unavailable; see dependency_check.log")
    data = ctx["aligned_full"]
    key = "counts" if "counts" in data.layers else None
    matrix = data.layers[key] if key else data.X
    estimate = 3.0 * matrix.shape[0] * matrix.shape[1] * 8 / 1024**3
    status.update(input_matrix="layers[counts]" if key else "original complete X",
                  profile=matrix_profile(matrix), estimated_peak_dense_GB=estimate)
    if estimate > ctx["settings"]["scmetabolism_max_dense_gb"]:
        return finish("estimated dense memory exceeds configured budget")
    matrix = sp.csr_matrix(matrix)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        return finish("nonfinite or negative expression; original R input not exported", "failed")
    (directory / "run_scmetabolism.R").write_text(R_SCMETABOLISM_SCRIPT)
    mmwrite(directory / "expression.mtx", matrix.T)
    (directory / "genes.txt").write_text("\n".join(map(str, data.var_names)) + "\n")
    (directory / "spots.txt").write_text("\n".join(map(str, data.obs_names)) + "\n")
    try:
        proc = subprocess.run([executable, str(directory / "run_scmetabolism.R"), str(directory),
                               str(ctx["settings"]["seed"])], capture_output=True, text=True,
                               timeout=ctx["settings"]["scmetabolism_timeout_seconds"])
        (directory / "execution.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            return finish("R package execution failed; see execution.log", "failed")
        scores = pd.read_csv(directory / "scMetabolism_KEGG_AUCell_scores.csv", index_col=0)
        scores.index = scores.index.astype(str)
        scores = scores.reindex(ctx["ids"])
        if scores.isna().any().any():
            return finish("R output IDs or score values not complete", "failed")
        pathways = ["Glycolysis / Gluconeogenesis", "Pentose phosphate pathway",
                    "Oxidative phosphorylation", "Glutathione metabolism"]
        if not all(p in scores for p in pathways):
            return finish("predefined four pathways absent from this package version", "failed")
        composite = scores[pathways].mean(1)
        write_csv(pd.DataFrame({"spot_id": ctx["ids"], "four_pathway_mean": composite}),
                  directory / "predefined_four_pathway_mean.csv")
        correlations = []
        for j, axis in enumerate(AXIS_ORDER):
            for pathway in pathways:
                f = results[("common_scale", "FullCoupling")]["scores"][:, j]
                s = scores[pathway].to_numpy()
                if np.std(f) > 0 and np.std(s) > 0:
                    rho = float(spearmanr(f, s)[0])
                else:
                    rho = np.nan
                correlations.append(dict(axis=axis, pathway=pathway, Spearman_rho=rho))
        write_csv(pd.DataFrame(correlations), directory / "descriptive_score_correlations.csv")
        status["note"] = "四通路均值仅描述整体代谢活性；没有强行生成 10 条通讯轴或原版论文的独立 k-means 区域。"
        fig, ax = plt.subplots(figsize=(6, 5))
        p = ax.scatter(*np.asarray(ctx["frozen"].obsm["spatial"])[:, :2].T,
                       c=composite.to_numpy(), s=ctx["settings"]["spot_size"], cmap="viridis")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        ax.set_title("scMetabolism: predefined 4-pathway mean")
        fig.colorbar(p, ax=ax, label="AUCell score")
        fig.savefig(directory / "four_pathway_activity.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return finish("completed", "completed")
    except subprocess.TimeoutExpired:
        return finish("R execution exceeded configured timeout", "failed")
    finally:
        if not ctx["settings"].get("keep_scmetabolism_matrix", False):
            (directory / "expression.mtx").unlink(missing_ok=True)


def niche_palette(labels):
    palette = {"Healthy_Region": "#d9d9d9", "TMCN_Transitional": "#bcbd22"}
    axis_colors = plt.get_cmap("tab10").colors
    for j, axis in enumerate(AXIS_ORDER):
        palette["TMCN_" + axis + "_Axis"] = axis_colors[j]
    used = {matplotlib.colors.to_hex(value) for value in palette.values()}
    candidates = list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    for label in sorted(set(map(str, labels))):
        if label not in palette:
            index = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % len(candidates)
            pool = candidates[index:] + candidates[:index]
            pool += [plt.get_cmap("hsv")((i + .5) / 256) for i in range(256)]
            value = next(c for c in pool if matplotlib.colors.to_hex(c) not in used)
            palette[label] = value
            used.add(matplotlib.colors.to_hex(value))
    return palette


def spatial_panel(ax, xy, labels, title, palette, size):
    ax.scatter(xy[:, 0], xy[:, 1], c=[palette[str(x)] for x in labels], s=size, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")


def make_figures(ctx, results, metrics, level2_summary, features):
    folder = ctx["out"] / "figures"
    xy = np.asarray(ctx["frozen"].obsm["spatial"])[:, :2]
    size = ctx["settings"]["spot_size"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    truth_labels = np.where(ctx["gt"] == 1, "Tumor_Region",
                           np.where(ctx["gt"] == 0, "Healthy_Region", "Unmapped"))
    palette = {"Tumor_Region": TUMOR_COLOR, "Healthy_Region": NON_TUMOR_COLOR,
               "Unmapped": "#d9d9d9"}
    plots = [("Pathology", truth_labels), ("Initial SpaCET-GMM", np.where(ctx["spacet_tumor"], "Tumor_Region", "Healthy_Region")),
             ("Saved final Level 1", ctx["original_l1"])]
    plots += [(m + " (common scale)", results[("common_scale", m)]["level1"]) for m in METHODS]
    for ax, (title, labels) in zip(axs.flat, plots):
        spatial_panel(ax, xy, labels, title, palette, size)
    fig.legend(handles=[Patch(color=c, label=l) for l, c in palette.items()],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(ctx["config"]["sample_id"] + ": paired Level 1 comparison")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(folder / "Level1_paired_maps.png", dpi=300)
    fig.savefig(folder / "Level1_paired_maps.pdf")
    plt.close(fig)
    all_labels = np.concatenate([results[("common_scale", m)]["level2"] for m in METHODS])
    palette = niche_palette(all_labels)
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ax, method in zip(axs, METHODS):
        spatial_panel(ax, xy, results[("common_scale", method)]["level2"], method, palette, size)
    legend_labels = sorted(set(all_labels))
    fig.legend(handles=[Patch(color=palette[l], label=l) for l in legend_labels],
               loc="lower center", ncol=min(3, max(1, len(legend_labels))), fontsize=7, frameon=False)
    fig.suptitle(ctx["config"]["sample_id"] + ": Level 2 (same microclusters and gate)")
    fig.tight_layout(rect=(0, min(0.40, 0.055 * math.ceil(len(legend_labels) / 3)), 1, 0.93))
    fig.savefig(folder / "Level2_common_scale_maps.png", dpi=300)
    fig.savefig(folder / "Level2_common_scale_maps.pdf")
    plt.close(fig)
    axes_to_plot = ctx["settings"]["axes_to_plot"]
    fig, axs = plt.subplots(3, len(axes_to_plot), figsize=(4 * len(axes_to_plot), 10), squeeze=False)
    for i, method in enumerate(METHODS):
        for j, axis in enumerate(axes_to_plot):
            ax = axs[i, j]
            z = results[("common_scale", method)]["scores"][:, AXIS_ORDER.index(axis)]
            points = ax.scatter(xy[:, 0], xy[:, 1], c=z, s=size, cmap="viridis", vmin=0, vmax=1, linewidths=0)
            ax.set_title(method + ": " + axis, fontsize=10)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    fig.colorbar(points, ax=axs.ravel().tolist(), shrink=0.5, label="Relative score (0-1)")
    fig.savefig(folder / "predefined_axis_score_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    valid_features = [c for c in features if c.startswith("SpaCET::")]
    if valid_features and ctx["spacet_tumor"].any():
        fig, axs = plt.subplots(1, 3, figsize=(17, max(4, 0.25 * len(valid_features) + 2)))
        for ax, method in zip(axs, METHODS):
            frame = features.loc[ctx["spacet_tumor"], valid_features].copy()
            frame["niche"] = results[("common_scale", method)]["level2"][ctx["spacet_tumor"]]
            means = frame.groupby("niche")[valid_features].mean().T
            im = ax.imshow(means.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(means.shape[1])); ax.set_xticklabels(means.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(means.shape[0])); ax.set_yticklabels([x.replace("SpaCET::", "") for x in means.index], fontsize=8)
            ax.set_title(method)
        fig.tight_layout()
        fig.savefig(folder / "cell_fraction_association_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_result_object(ctx, results, metrics, level2_summary, reconstruction, reference_status):
    import anndata as ad
    obs = pd.DataFrame(index=ctx["ids"])
    obs["native_cluster"] = ctx["clusters"]
    obs["Malignant"] = ctx["malignant"]
    obs["pathology_original"] = ctx["gt_text"]
    obs["pathology_binary"] = ctx["gt"]
    obs["valid_for_paired_metrics"] = ctx["valid_gt"]
    obs["SpaCET_GMM_tumor"] = ctx["spacet_tumor"]
    obs["saved_Level1"] = ctx["original_l1"]
    obs["saved_Level2"] = ctx["original_l2"]
    for (mode, method), result in results.items():
        prefix = mode + "__" + method
        obs[prefix + "__Level1"] = result["level1"]
        obs[prefix + "__Level2"] = result["level2"]
    obj = ad.AnnData(X=sp.csr_matrix((len(obs), 0), dtype=np.float32), obs=obs)
    obj.obsm["spatial"] = np.asarray(ctx["frozen"].obsm["spatial"])
    obj.obsp["spatial_connectivities"] = ctx["graph"].copy()
    for (mode, method), result in results.items():
        obj.obsm[mode + "__" + method + "__scores"] = result["scores"]
    obj.uns["axis_order"] = AXIS_ORDER
    obj.uns["protocol"] = PROTOCOL_VERSION
    obj.uns["source_input_h5ad"] = ctx["config"]["input_h5ad"]
    obj.uns["source_final_h5ad"] = ctx["config"]["final_h5ad"]
    obj.uns["note"] = "轻量结果对象不复制表达；从 source_input_h5ad 读取完整表达。"
    obj.write_h5ad(ctx["out"] / "component_comparison_results.h5ad", compression="gzip")
    write_csv(obs.reset_index().rename(columns={obs.index.name or "index": "spot_id"}), ctx["out"] / "all_spot_labels.csv")
    freeze_after = {"spots": hash_values(ctx["frozen"].obs_names),
                    "clusters": hash_values(ctx["frozen"].obs[ctx["config"]["native_cluster_key"]].astype(str)),
                    "graph": hash_graph(ctx["graph"]),
                    "malignant": hashlib.sha256(ctx["malignant"].astype(np.float64).tobytes()).hexdigest()}
    if freeze_after != ctx["freeze"]:
        raise AssertionError("实验中冻结输入发生变化。")
    write_json(ctx["out"] / "freeze_integrity_check.json", {"before": ctx["freeze"], "after": freeze_after, "passed": True})
    write_json(ctx["out"] / "run_status.json", {"status": "completed", "protocol": PROTOCOL_VERSION,
        "engine_sha256": IMPLEMENTATION_SHA256,
        "sample_id": ctx["config"]["sample_id"], "cancer_type": ctx["config"]["cancer_type"],
        "reconstruction": reconstruction, "scmetabolism_reference": reference_status,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "strict_reconstruction": ctx["settings"]["strict_reconstruction"]})
    print("完成:", ctx["config"]["sample_id"], "保存至", ctx["out"])
    return obj


def paired_descriptions(frame, metrics, key_columns, contrasts):
    """All differences are paired within sample, no pooled-spot inference."""
    records = []
    for contrast_name, left_mode, left_method, right_mode, right_method in contrasts:
        left = frame[(frame.score_mode == left_mode) & (frame.method == left_method)].set_index("sample_id")
        right = frame[(frame.score_mode == right_mode) & (frame.method == right_method)].set_index("sample_id")
        common = left.index.intersection(right.index)
        for sample in common:
            for metric in metrics:
                if metric not in left or metric not in right:
                    continue
                lv, rv = left.loc[sample, metric], right.loc[sample, metric]
                records.append(dict(sample_id=sample, contrast=contrast_name, metric=metric,
                                    left=lv, right=rv, delta=lv - rv))
    return pd.DataFrame(records, columns=["sample_id", "contrast", "metric", "left", "right", "delta"])


def describe_differences(pairs):
    rows = []
    for (contrast, metric), group in pairs.groupby(["contrast", "metric"]):
        d = group.delta.dropna().to_numpy()
        if not len(d):
            continue
        rows.append(dict(contrast=contrast, metric=metric, n_paired_slices=len(d),
            mean_delta=d.mean(), median_delta=np.median(d), SD_delta=d.std(ddof=1) if len(d)>1 else np.nan,
            Q25_delta=np.quantile(d, .25), Q75_delta=np.quantile(d, .75),
            min_delta=d.min(), max_delta=d.max(),
            improved=int((d > 1e-10).sum()), decreased=int((d < -1e-10).sum()),
            unchanged=int((np.abs(d) <= 1e-10).sum())))
    return pd.DataFrame(rows)


def independent_unit_statistics(pairs, grouping_path, seed=42):
    """Inference is enabled only by an explicitly completed independent-unit table."""
    if not grouping_path or not Path(grouping_path).is_file():
        return pd.DataFrame(), "not_run: no verified independent-unit grouping"
    grouping = pd.read_csv(grouping_path, dtype=str).fillna("")
    required = {"sample_id", "independent_unit", "confirmed"}
    if not required.issubset(grouping):
        raise ValueError("分组表需包含 sample_id, independent_unit, confirmed。")
    if grouping.sample_id.duplicated().any():
        raise ValueError("分组表存在重复 sample_id。")
    needed = set(pairs.sample_id)
    g = grouping.set_index("sample_id").reindex(sorted(needed)).fillna("")
    if (g.independent_unit.str.strip() == "").any() or not g.confirmed.str.lower().isin(["true", "1", "yes"]).all():
        return pd.DataFrame(), "not_run: independent_unit missing or not explicitly confirmed"
    merged = pairs.merge(grouping[["sample_id", "independent_unit"]], on="sample_id", validate="many_to_one")
    unit = merged.groupby(["contrast", "metric", "independent_unit"], as_index=False).delta.mean()
    rng = np.random.default_rng(seed)
    output = []
    for (contrast, metric), group in unit.groupby(["contrast", "metric"]):
        d = group.delta.dropna().to_numpy()
        if len(d) < 2:
            continue
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
        boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
        output.append(dict(contrast=contrast, metric=metric, n_independent_units=len(d),
             mean_unit_delta=d.mean(), bootstrap_CI95_low=np.quantile(boot, .025),
             bootstrap_CI95_high=np.quantile(boot, .975), paired_wilcoxon_p=p))
    frame = pd.DataFrame(output)
    if len(frame):
        frame["BH_q_all_reported_tests"] = bh_adjust(frame.paired_wilcoxon_p)
    return frame, "computed at independent-unit level; repeated slices averaged within unit"


def aggregate_experiment(root, samples, grouping_path=None, seed=42):
    root = Path(root)
    output = root / "_summary"
    output.mkdir(parents=True, exist_ok=True)
    status_rows, metric_frames, level2_frames = [], [], []
    eligible_engine_hashes = set()
    for config in samples:
        folder = root / config["sample_id"]
        path = folder / "run_status.json"
        status = json.loads(path.read_text()) if path.is_file() else {"status": "not_run"}
        matched = status.get("reconstruction", {}).get("status") == "matched"
        eligible = (status.get("status") == "completed" and status.get("protocol") == PROTOCOL_VERSION and matched)
        reference_eligible = (status.get("status") in ["completed", "reconstruction_mismatch"]
                              and status.get("protocol") == PROTOCOL_VERSION)
        status_rows.append(dict(sample_id=config["sample_id"], status=status.get("status"),
                               protocol=status.get("protocol"), reconstruction_matched=matched,
                               eligible_for_initial_final_summary=reference_eligible,
                               eligible_for_component_summary=eligible))
        if reference_eligible:
            frame = pd.read_csv(folder / "level1_metrics.csv")
            metric_frames.append(frame if eligible else frame[frame.score_mode.eq("reference")])
        if eligible:
            eligible_engine_hashes.add(status.get("engine_sha256", "missing"))
            level2_frames.append(pd.read_csv(folder / "level2_summary.csv"))
    status = pd.DataFrame(status_rows)
    write_csv(status, output / "sample_completion_status.csv")
    if len(eligible_engine_hashes) > 1 or "missing" in eligible_engine_hashes:
        raise RuntimeError("完成样本使用的实现版本不一致；请用同一版 notebook 生成结果后汇总。")
    if not metric_frames:
        raise RuntimeError("尚无完成且完整方案复现一致的样本；查看 sample_completion_status.csv。")
    metrics = pd.concat(metric_frames, ignore_index=True)
    level2 = pd.concat(level2_frames, ignore_index=True) if level2_frames else pd.DataFrame(
        columns=["sample_id", "score_mode", "method"])
    contrasts = [("SavedFull_minus_SpaCET", "reference", "SavedFull", "reference", "SpaCET_GMM"),
                 ("CommonFull_minus_SenderOnly", "common_scale", "FullCoupling", "common_scale", "SenderOnly"),
                 ("CommonFull_minus_AUCellOnly", "common_scale", "FullCoupling", "common_scale", "AUCellOnly"),
                 ("RawFull_minus_SenderOnly", "original_rules", "FullCoupling", "original_rules", "SenderOnly"),
                 ("RawFull_minus_AUCellOnly", "original_rules", "FullCoupling", "original_rules", "AUCellOnly")]
    macro_pairs = paired_descriptions(metrics, ["ARI", "F1", "IoU", "Precision", "Recall"], [], contrasts)
    l2_pairs = paired_descriptions(level2, ["active_fraction_in_fixed_roi", "neighbor_agreement",
        "chance_adjusted_neighbor_agreement", "median_cell_fraction_adjusted_eta2",
        "median_cell_fraction_loco_R2"], [], contrasts[1:])
    pairs = pd.concat([macro_pairs.assign(level="Level1"), l2_pairs.assign(level="Level2")], ignore_index=True)
    descriptions = describe_differences(pairs)
    write_csv(metrics, output / "all_level1_metrics.csv")
    write_csv(level2, output / "all_level2_summary.csv")
    write_csv(pairs, output / "paired_sample_differences.csv")
    write_csv(descriptions, output / "paired_effect_size_summary.csv")
    infer, inference_status = independent_unit_statistics(pairs, grouping_path, seed)
    if len(infer):
        write_csv(infer, output / "independent_unit_statistics.csv")
    else:
        (output / "independent_unit_statistics.csv").unlink(missing_ok=True)
    write_json(output / "summary_status.json", {"protocol": PROTOCOL_VERSION,
        "n_expected": len(samples), "n_eligible": int(status.eligible_for_component_summary.sum()),
        "n_initial_final_eligible": int(status.eligible_for_initial_final_summary.sum()),
        "complete_10_samples": bool(status.eligible_for_component_summary.all()),
        "inference": inference_status,
        "level2_caution": "空间一致性与标签重叠是描述指标；细胞组成来自同一转录组，不能证明代谢流或跨边界通讯。"})
    if not grouping_path or not Path(grouping_path).is_file():
        write_csv(pd.DataFrame({"sample_id": [c["sample_id"] for c in samples],
                               "independent_unit": "", "confirmed": "false"}),
                  output / "independent_units_TEMPLATE.csv")
    for metric in ["ARI", "F1", "IoU"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, contrast in zip(axes, [x[0] for x in contrasts[:3]]):
            frame = macro_pairs[(macro_pairs.metric == metric) & (macro_pairs.contrast == contrast)]
            for row in frame.itertuples():
                ax.plot([0, 1], [row.right, row.left], "o-", alpha=.7, linewidth=1, markersize=4)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Reference / simple", "Final / full"], rotation=10)
            ax.set_title(contrast.replace("_minus_", "\nminus "), fontsize=10)
            ax.set_ylabel(metric)
        fig.tight_layout(); fig.savefig(output / ("paired_" + metric + ".png"), dpi=300)
        fig.savefig(output / ("paired_" + metric + ".pdf")); plt.close(fig)
    for contrast in [x[0] for x in contrasts[:3]]:
        frame = macro_pairs[(macro_pairs.metric == "ARI") & (macro_pairs.contrast == contrast)]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(frame.sample_id, frame.delta, color=np.where(frame.delta >= 0, "#2ca02c", "#d62728"))
        ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Paired delta ARI")
        ax.tick_params(axis="x", rotation=75); ax.set_title(contrast)
        fig.tight_layout(); fig.savefig(output / (contrast + "__delta_ARI.png"), dpi=300); plt.close(fig)
    print("汇总完成:", output, "; eligible", int(status.eligible_for_component_summary.sum()), "/", len(samples))
    print("推断统计:", inference_status)
    return metrics, level2, pairs, descriptions, status
