"""Meaningful synthetic-data checks; never substitute results for real samples."""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import json
import ast
import tempfile
import warnings
import copy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import nbformat
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import component_engine as e

REPORT = []


def check(name, fn):
    fn()
    REPORT.append({'check': name, 'status': 'passed'})
    print('PASS', name, flush=True)


def test_notebook_format_and_python39():
    paths = list((BASE / 'notebooks').glob('*.ipynb')) + [BASE / '00_Run_All.ipynb']
    assert len(paths) == 12
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                ast.parse(cell.source, feature_version=(3, 9))
                compile(cell.source, str(path) + ':' + str(i), 'exec')
        if path.name[0:2].isdigit() and path.parent.name == 'notebooks' and not path.name.startswith('11_'):
            assert not any('model.train(' in c.source for c in nb.cells if c.cell_type == 'code')
            namespace = {}
            # Definitions are all self-contained before the first execution cell.
            start = next(i for i,c in enumerate(nb.cells) if c.cell_type == 'code' and 'out = initialize_experiment' in c.source)
            for cell in nb.cells[5:start]:
                if cell.cell_type == 'code':
                    exec(compile(cell.source, '<notebook definitions>', 'exec'), namespace)
            assert callable(namespace['annotate_scores']) and callable(namespace['save_result_object'])


def test_bh():
    p = np.array([0.001, .05, .3, .8, 1, 0, .05])
    assert np.allclose(e.bh_adjust(p), multipletests(p, method='fdr_bh')[1])


def make_synthetic(directory, tag='synthetic'):
    rng = np.random.default_rng(18)
    n, groups_n = 240, 8
    ids = pd.Index([f'spot{i:03d}' for i in range(n)], name='spot')
    clusters = np.repeat(np.arange(groups_n).astype(str), n // groups_n)
    malignant = np.full(n, .75)
    malignant[clusters == '1'] = .45
    malignant[clusters == '5'] = .30
    malignant[np.flatnonzero(clusters == '0')[:18]] = .35
    sender = np.empty((n, 10))
    for j in range(10):
        hot = np.isin(clusters, [str(j % 8), str((j + 2) % 8)])
        sender[:, j] = np.where(hot, .85, .04) + rng.uniform(0, .02, n)
    receiver = rng.uniform(.4, .9, (n, 10))
    graph = sp.lil_matrix((n, n))
    for group in range(groups_n):
        ix = np.flatnonzero(clusters == str(group))
        for i in range(len(ix)):
            for shift in [1, 2, 3]:
                a, b = ix[i], ix[(i+shift) % len(ix)]
                graph[a, b] = graph[b, a] = 1
    graph = graph.tocsr()
    baseline = malignant >= .5
    gt = np.where(np.isin(clusters, ['0','1','2','4','6']), 'Tumor', 'Healthy')
    counts = sp.csr_matrix(rng.poisson(1, (n+1, 80)).astype(np.float32))
    full_ids = ids.append(pd.Index(['excluded_spot']))
    full = ad.AnnData(counts, obs=pd.DataFrame({'annot_type': np.concatenate([gt, ['Healthy']])}, index=full_ids),
                      var=pd.DataFrame(index=[f'G{i}' for i in range(80)]))
    full.layers['counts'] = full.X.copy()
    raw_auc = pd.DataFrame(rng.uniform(0, 1, (n+1, 10)), index=full_ids, columns=e.AXIS_ORDER)
    raw_auc.iloc[-1] = 0
    full.obsm['aucell_estimate'] = raw_auc
    frozen = full[ids, :25].copy()
    frozen.obs['latent_leiden_0.6'] = clusters
    frozen.obs['Malignant'] = malignant
    frozen.obsp['spatial_connectivities'] = graph
    frozen.obsm['spatial'] = np.column_stack([np.arange(n) % 20, np.arange(n) // 20])
    for j, axis in enumerate(e.AXIS_ORDER):
        frozen.obs[axis + '_Sender_Score'] = sender[:, j]
        frozen.obs[axis + '_Receiver_Score'] = receiver[:, j]
    raw_full = np.log1p(sender * graph.dot(receiver))
    result = e.annotate_scores(raw_full, clusters, malignant, .4, baseline)
    frozen.obs['Level1_Macro_Region'] = result['level1']
    frozen.obs['Level2_Micro_Niche'] = result['level2']
    for j, axis in enumerate(e.AXIS_ORDER):
        frozen.obs['Coupling_' + axis] = raw_full[:, j]
    directory = Path(directory)
    full.write_h5ad(directory / 'complete.h5ad')
    frozen.write_h5ad(directory / 'final.h5ad')
    pd.DataFrame({'Unsupervised_Pred_test': np.where(baseline, 'Predicted_Tumor', 'Predicted_Non_Tumor'),
                  'Malignant': malignant}, index=ids).iloc[::-1].to_csv(directory / 'prediction.csv')
    pd.DataFrame({'Metric':['Optimal_GMM_Threshold'], 'Value':[.5]}).to_csv(directory / 'threshold.csv', index=False)
    pd.DataFrame({'Malignant': malignant, 'CAF': np.where(clusters == '1', .8, .1),
        'CD8 T': rng.uniform(.05, .5, n), 'Endothelial': rng.uniform(.01, .2, n),
        'constant': 0}, index=ids).to_csv(directory / 'fractions.csv')
    pd.DataFrame({'TMCN_Name':['TMCN_' + a + '_Axis' for a in e.AXIS_ORDER],
        'Source_Pathways':'Hypoxia', 'Source_Genes':[f'G{i},G{i+1}' for i in range(10)],
        'Target_Genes':[f'G{i+30}' for i in range(10)]}).to_csv(directory / 'axis.csv', index=False)
    config = dict(sample_id=tag, cancer_type='Synthetic', gpu='0', gate_offset=.10,
        input_h5ad=str(directory / 'complete.h5ad'), final_h5ad=str(directory / 'final.h5ad'),
        spacet_prediction_csv=str(directory / 'prediction.csv'),
        spacet_threshold_csv=str(directory / 'threshold.csv'), spacet_fraction_csv=str(directory / 'fractions.csv'),
        axis_table_path=str(directory / 'axis.csv'), native_cluster_key='latent_leiden_0.6',
        graph_key='spatial_connectivities', malignant_column='Malignant', pathology_column='annot_type',
        gt_map={'Tumor':1,'Healthy':0})
    settings = json.loads((BASE / 'sample_configs.json').read_text())['settings']
    settings.update(output_root=str(directory / 'outputs'), run_scmetabolism_reference=False)
    return config, settings, frozen, result


def original_cell7_reference(frozen, source_path):
    if source_path.suffix == '.py':
        source = source_path.read_text()
    else:
        notebook = json.loads(source_path.read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'Cell 7:' in ''.join(c.get('source',[]))[:500])
    source = source[:source.index('# 可视化: 过滤前 / 过滤后')]
    tree = ast.parse(source)
    class RemoveGraphics(ast.NodeTransformer):
        def visit_Import(self, node):
            return None
        def visit_ImportFrom(self, node):
            return None
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                f = ast.unparse(node.value.func)
                if f in ['print','display'] or f.startswith(('sns.','plt.','sq.','axes[')):
                    return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func).startswith('plt.'):
                return None
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(RemoveGraphics().visit(tree))
    from sklearn.mixture import GaussianMixture
    from scipy.stats import hypergeom
    namespace = dict(np=np, pd=pd, os=os, GaussianMixture=GaussianMixture, hypergeom=hypergeom,
        multipletests=multipletests, adata_model=SimpleNamespace(obs=frozen.obs.copy(), obsp=frozen.obsp, n_obs=frozen.n_obs),
        AXES=e.AXIS_ORDER, native_cluster_key='latent_leiden_0.6', axes=[None]*10)
    with patch('pandas.read_csv', return_value=pd.DataFrame({'Metric':['Optimal_GMM_Threshold'],'Value':[.5]})):
        exec(compile(tree, '<original Cell7 without graphics>', 'exec'), namespace)
    labels = namespace['adata_model'].obs.Niche_Annotation.astype(str)
    axes = [e.label_to_axes(x) if x.startswith(('Single-axis:', 'Multi-axis:')) else () for x in labels]
    return e.canonical_labels(axes, frozen.obs.Malignant.to_numpy() >= .5)


def test_source_parity():
    with tempfile.TemporaryDirectory(prefix='component_parity_') as tmp:
        cfg, settings, frozen, result = make_synthetic(tmp)
        sources = [('SN048_A121573_Rep1',.10), ('lung_cancer_and_kidney_cancer_LC1',.10), ('Human_Prostate_Cancer',.25)]
        for sample, offset in sources:
            path = BASE.parent / 'upload' / ('myAndSpaCET_' + sample + '(1).ipynb')
            if not path.is_file():
                path = BASE / 'tests' / 'reference_cells' / (sample + '_Cell7.py')
            l1,l2=original_cell7_reference(frozen, path)
            scores=np.column_stack([frozen.obs['Coupling_' + a].to_numpy() for a in e.AXIS_ORDER])
            actual=e.annotate_scores(scores, frozen.obs['latent_leiden_0.6'].astype(str).to_numpy(),
                frozen.obs.Malignant.to_numpy(), .5-offset, frozen.obs.Malignant.to_numpy()>=.5)
            assert np.array_equal(l1, actual['level1']), sample
            assert np.array_equal(l2, actual['level2']), sample


def test_end_to_end():
    with tempfile.TemporaryDirectory(prefix='component_e2e_') as tmp:
        cfg, settings, frozen, reference = make_synthetic(tmp)
        out=e.initialize_experiment(cfg, settings, [cfg['sample_id']])
        ctx=e.load_frozen_inputs(cfg, settings, out)
        enzyme=e.recover_enzyme_scores(ctx)
        assert enzyme.shape==(240,10)
        raw=e.construct_scores(ctx,enzyme)
        results, metrics, reconstruction=e.run_comparisons(ctx,raw)
        assert reconstruction['status']=='matched'
        assert len(metrics)==8 and metrics.n_valid.nunique()==1
        assert all(np.all(r['level1'][ctx['spacet_tumor']]=='Tumor_Region') for r in results.values())
        l2, assoc, features=e.evaluate_level2(ctx,results)
        assert len(l2)==6 and not features.columns.str.contains('Malignant').any()
        assert not assoc.empty and assoc.n_microclusters.min()>=5
        e.make_figures(ctx,results,metrics,l2,features)
        status=e.run_scmetabolism_reference(ctx,results)
        assert status['status']=='skipped'
        obj=e.save_result_object(ctx,results,metrics,l2,reconstruction,status)
        reread=ad.read_h5ad(out/'component_comparison_results.h5ad')
        assert reread.shape==(240,0) and e.hash_graph(reread.obsp['spatial_connectivities'])==ctx['freeze']['graph']
        m,l,p,d,s=e.aggregate_experiment(settings['output_root'],[cfg])
        assert s.eligible_for_component_summary.all() and len(d)>0
        assert not (Path(settings['output_root'])/'_summary'/'independent_unit_statistics.csv').exists()
        # Input h5ad has not been overwritten.
        assert ad.read_h5ad(cfg['input_h5ad']).shape==(241,80)
        assert ad.read_h5ad(cfg['final_h5ad']).shape==(240,25)


def test_failure_guards():
    with tempfile.TemporaryDirectory(prefix='component_guards_') as tmp:
        cfg,settings,frozen,_=make_synthetic(tmp)
        out=e.initialize_experiment(cfg,settings,[cfg['sample_id']])
        ctx=e.load_frozen_inputs(cfg,settings,out)
        ctx['original_l2'][0]='TMCN_LPA_Axis'
        try:
            e.run_comparisons(ctx,e.construct_scores(ctx,e.recover_enzyme_scores(ctx)))
        except RuntimeError as error:
            assert '未复现' in str(error)
        else:
            raise AssertionError('reconstruction guard did not stop')
        assert json.loads((out/'run_status.json').read_text())['status']=='reconstruction_mismatch'
        # Initial/final comparison remains separately usable, component comparison excluded.
        _,_,pairs,_,state=e.aggregate_experiment(settings['output_root'],[cfg])
        assert set(pairs.contrast)=={'SavedFull_minus_SpaCET'}
        assert not state.eligible_for_component_summary.any()
        try:
            e.label_to_axes('unexpected_label')
        except ValueError:
            pass
        else:
            raise AssertionError('unknown niche silently mapped')
        pred=pd.read_csv(cfg['spacet_prediction_csv'],index_col=0)
        pred.iloc[0,pred.columns.get_loc('Malignant')]=.9999
        pred.to_csv(cfg['spacet_prediction_csv'])
        try:
            e.load_frozen_inputs(cfg,settings,out)
        except ValueError as error:
            assert '恶性分数' in str(error)
        else:
            raise AssertionError('mixed malignancy provenance accepted')


def test_scale_and_inference():
    x=np.array([[0,0,5],[1,0,5],[2,0,5]],float)
    norm,scale=e.common_scale(x)
    assert np.array_equal(norm[:,1],np.zeros(3))
    assert np.allclose(norm[:,0],[0,.5,1])
    assert np.all(norm[:,2]==1)
    t=e.learn_threshold(np.zeros(80))
    assert np.isinf(t['T_high'])
    pairs=pd.DataFrame({'sample_id':['a1','a2','b','c'], 'contrast':'test', 'metric':'ARI',
                        'delta':[.1,.3,-.1,.2]})
    with tempfile.TemporaryDirectory() as tmp:
        file=Path(tmp)/'groups.csv'
        pd.DataFrame({'sample_id':pairs.sample_id,'independent_unit':['a','a','b','c'],
                      'confirmed':'true'}).to_csv(file,index=False)
        result,status=e.independent_unit_statistics(pairs,file)
        assert result.iloc[0].n_independent_units==3
        assert np.isclose(result.iloc[0].mean_unit_delta,(.2-.1+.2)/3)


def test_plot_colors_are_distinct():
    names=['Multi_Axis_' + a + '_' + b for a,b in zip(e.AXIS_ORDER,e.AXIS_ORDER[::-1])]
    names += ['TMCN_' + a + '_Axis' for a in e.AXIS_ORDER]
    palette=e.niche_palette(names)
    colors=[e.matplotlib.colors.to_hex(palette[name]) for name in sorted(set(names))]
    assert len(colors)==len(set(colors))


if __name__=='__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    for name,fn in [('12_notebook_format_and_python39',test_notebook_format_and_python39),
                    ('BH_matches_statsmodels',test_bh),
                    ('original_Cell7_parity_three_variants',test_source_parity),
                    ('synthetic_AnnData_end_to_end',test_end_to_end),
                    ('failure_guards_and_partial_reference_summary',test_failure_guards),
                    ('scale_degeneracy_and_grouped_inference',test_scale_and_inference),
                    ('distinct_legend_colors',test_plot_colors_are_distinct)]:
        check(name,fn)
    (BASE/'VALIDATION_REPORT.json').write_text(json.dumps({
        'scope':'Synthetic and source parity only; real school-server data and R branch not executed',
        'checks':REPORT},ensure_ascii=False,indent=2))
