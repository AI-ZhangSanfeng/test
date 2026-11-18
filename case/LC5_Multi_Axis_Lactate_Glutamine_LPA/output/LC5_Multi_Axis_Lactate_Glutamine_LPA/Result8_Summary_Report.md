# Result 8 Summary: LC5 Multi_Axis_Lactate_Glutamine_LPA

## 1. Analysis design
This analysis evaluates whether the TuMeNiche-defined `Multi_Axis_Lactate_Glutamine_LPA` niche in the LC5 lung cancer sample is associated with pathological tumor architecture and medically interpretable tumor microenvironment features.

## 2. Key quantitative results
- Total spots analyzed: 622
- Raw genes used for DEG/GSEA: 17845
- Target niche spots: 90
- Target spots in pathology tumor region: 90.0
- Target tumor-overlap ratio: 1.0
- Fisher exact odds ratio: inf
- Fisher exact P value: 8.93066414183515e-31

## 3. Interpretation template
The identification of a `Multi_Axis_Lactate_Glutamine_LPA` niche suggests that TuMeNiche can capture a spatially restricted tumor microenvironment state with simultaneous lactate, glutamine, and LPA-axis activity. Because no patient-level survival, stage, recurrence, or metastasis information is available in this notebook, this result should be interpreted as medically interpretable pathological and functional relevance rather than direct prognostic association.

## 4. Recommended manuscript wording
In the LC5 lung cancer sample, TuMeNiche identified a distinct Multi_Axis_Lactate_Glutamine_LPA niche. Spatial visualization showed that this niche was localized within the tissue architecture in relation to pathological tumor annotations. Further analyses evaluated tumor-region enrichment, tumor-boundary proximity, three-axis co-activation, SpaCET-derived cell composition, and functional enrichment relative to other tumor regions, supporting the biological interpretability of this multi-axis metabolic niche.

## 5. Main output files
- Fig8A_Pathology_vs_Multi_Axis_Lactate_Glutamine_LPA_Highlight.png
- Result8_TargetNiche_Pathology_Overlap_Stats.csv
- Result8_ThreeAxis_Coactivation_Stats.csv
- Result8_CellFraction_Target_vs_OtherTumor_Stats.csv
- DEG_TargetNiche_vs_OtherTumorRegions_AllGenes.csv
- ORA_TargetNiche_UpGenes_vs_OtherTumor.csv
- Result8_Permutation_Summary.csv