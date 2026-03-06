# ICML TODO

## Freeze the protocol (one-page spec)

- [ ] Write "Protocol Spec" (exactly one page):
  - Proposal labels (K) (fish: habitat; ADE: top-50 scenes)
  - Audit labels (H) (fish: 10 parts; ADE: 151 seg classes)
  - Feature summary (X_i = \mathrm{pool}(z_{i,p})) (choose max or mean; fix once)
  - Proposal model (default): L1 logistic (OVR or multinomial); ranking (s_j=\sum_c |W_{cj}|)
  - Budgets (B \in {10,25,50,100})
  - Grounding metric: best-class AP; grounded if AP ≥ τ (fix τ; report τ sensitivity only in appendix)
  - Yield(B) and AUC over B
  - Null: shuffle (K) (fish: within-genus if used; ADE: global)
  - Uncertainty: bootstrap over audit images

## 1) Shared infrastructure (all datasets, all libraries)

### 1.1 Canonical representations

- [ ] Implement canonical patch→image summary:
  - input: per-patch feature activations (z_{i,p,j})
  - output: per-image vector (X_i[j])
- [ ] Implement canonical activation-map export:
  - input: (z_{i,p,j})
  - output: per-feature per-image score maps in a common format (patch grid + upsample rule)

### 1.2 Proposal (ranking) step

- [ ] Implement L1 logistic proposal (multiclass):
  - OVR vs multinomial chosen and fixed
  - feature importance (s_j=\sum_c |W_{cj}|)
- [ ] Implement bootstrap stability (optional but recommended):
  - selection frequency per feature
  - (keep default ranking as weight magnitude; use stability as ablation)

### 1.3 Audit (grounding) step

- [ ] Implement best-class AP over mask classes:
  - fish: best-part AP over 10 parts
  - ADE: best-class AP over 151 classes
- [ ] Implement groundedness + yield curve:
  - grounded(j) = AP_j ≥ τ
  - Yield(B), AUC_B
- [ ] Implement bootstrap CI over audit images
- [ ] Implement permutation null (proposal-stage):
  - rerun proposal → audit → yield

## 2) Feature-library baselines (apples-to-apples)

### 2.1 Baseline libraries to include (minimum)

- [ ] Raw units baseline (no decomposition):

  - pick one: token dimensions / MLP neurons / attention head outputs (define once)
- [ ] PCA on patch activations (m components)
- [ ] NMF on patch activations (m components)
- [x] k-means (existing)
- [ ] Random-feature null matched to activation distribution

(Optional if time)

- [ ] sparse PCA or ICA (pick one)

### 2.2 Baseline exports

- [ ] Record baseline feature activations for audit datasets (same format as SAE)
- [ ] Compute per-image summaries (X_i) for proposal for each baseline

## 3) FishVista (primary scientific-domain)

### 3.1 Data + labels

- [ ] Finalize habitat label mapping (K):
  - choose 1–2 tasks (e.g., reef vs pelagic; deep vs shallow; cruiser vs maneuverer)
- [ ] Decide whether to use genus grouping for the permutation null / CV split (document choice)
- [ ] Verify part masks and class mapping (10 parts) on seg val

### 3.2 Backbone activations (done / pending)

- [x] Record DINOv3 ViT-L/16 activations (256 tokens) on fish-vista-imgfolder/all (2e339319)
- [x] Record DINOv3 ViT-L/16 activations (256 tokens) on fish-vista-segfolder/train (5dcb2f75)
- [x] Record DINOv3 ViT-L/16 activations (256 tokens) on fish-vista-segfolder/val (8692dfa9)
- [x] Record BioCLIP 2 ViT-L/14 activations (256 tokens) on fish-vista-imgfolder/all (643f9f5e)
- [x] Record BioCLIP 2 ViT-L/14 activations (256 tokens) on fish-vista-segfolder/train (c8abf6e8)
- [x] Record BioCLIP 2 ViT-L/14 activations (256 tokens) on fish-vista-segfolder/val (1bc9cc5d)
- [ ] Record SigLIP 2 ViT-L/16 activations (256 tokens) on fish-vista-imgfolder/all
- [ ] Record SigLIP 2 ViT-L/16 activations (256 tokens) on fish-vista-segfolder/train
- [ ] Record SigLIP 2 ViT-L/16 activations (256 tokens) on fish-vista-segfolder/val

### 3.3 SAE training (fish)

- [x] Train Matryoshka SAEs on DINOv3 ViT-L/16 on fish-vista-imgfolder (all)
- [x] Train Matryoshka SAEs on BioCLIP 2 ViT-L/14 on fish-vista-imgfolder (all)
- [ ] Train Matryoshka SAEs on SigLIP 2 ViT-L/16 on fish-vista-imgfolder (all)
- [ ] (Optional) Train TopK+AuxK SAE variant on fish for the best backbone

### 3.4 SAE activations for protocol

- [x] Record SAE activations for Pareto SAEs (BioCLIP2) on fish-vista-segfolder/train
- [x] Record SAE activations for Pareto SAEs (BioCLIP2) on fish-vista-segfolder/val
- [x] Record SAE activations for Pareto SAEs (DINOv3) on fish-vista-segfolder/train
- [ ] Record SAE activations for Pareto SAEs (DINOv3) on fish-vista-segfolder/val
- [ ] Record SAE activations for Pareto SAEs (SigLIP2) on fish-vista-segfolder/train
- [ ] Record SAE activations for Pareto SAEs (SigLIP2) on fish-vista-segfolder/val

### 3.5 Run the canonical protocol (fish)

- [ ] Proposal: fit L1 habitat classifier on classification split summaries; rank features
- [ ] Audit: compute best-part AP for top-B ranked features on seg val
- [ ] Report: Yield(B), AUC_B, bootstrap CI, permutation null
- [ ] Qualitative: visualize top grounded features (per habitat side) for expert review

## 4) ADE20K (general-domain sanity check)

### 4.1 ADE labels

- [ ] Define proposal labels (K): top-50 ADE20K scene classes by train frequency
  - produce fixed list + counts
- [ ] Audit labels (H): all 151 semantic segmentation classes (as in dataset)

### 4.2 Backbone activations (already done for DINOv3 ViT-L/16, 256 tokens)

- [x] Record DINOv3 ViT-L/16 activations (256 tokens) on ADE20K/train (614861a0)
- [x] Record DINOv3 ViT-L/16 activations (256 tokens) on ADE20K/val (3802cb66)

### 4.3 Feature libraries trained on IN1K (already done / verify coverage)

- [x] Train Matryoshka SAEs on DINOv3 ViT-L/16 on IN1K (train/val)
- [x] Train vanilla SAEs on DINOv3 ViT-L/16 on IN1K (train/val)

### 4.4 Export feature activations on ADE for protocol

- [x] Record SAE activations for Pareto Matryoshka SAEs (IN1K-trained) on ADE/train
- [x] Record SAE activations for Pareto Matryoshka SAEs (IN1K-trained) on ADE/val
- [x] Record SAE activations for Pareto vanilla SAEs (IN1K-trained) on ADE/train
- [x] Record SAE activations for Pareto vanilla SAEs (IN1K-trained) on ADE/val
- [ ] Record baseline feature activations on ADE/train+val (PCA/NMF/raw units/k-means)
  - ensure same token grid / upsample conventions

### 4.5 Run the canonical protocol (ADE)

- [ ] Proposal: fit L1 classifier on ADE train using top-50 scene classes; rank features
- [ ] Audit: compute best-class AP over 151 seg classes on ADE val for top-B features
- [ ] Report: Yield(B), AUC_B, bootstrap CI, permutation null (shuffle scene labels)
- [ ] Qualitative: show 10 grounded features with best-matching seg class overlays

## 5) Reporting + writing

### 5.1 Main figures/tables

- [ ] Fig: pipeline schematic (proposal (K) → audit (H) → yield curve)
- [ ] Fig: FishVista Yield(B) curves (SAE vs PCA vs NMF vs raw units) + null band
- [ ] Fig: ADE Yield(B) curves (same methods)
- [ ] Table: AUC_B ± CI for fish + ADE + permutation null
- [ ] Qualitative panels: top grounded features (fish + ADE)

### 5.2 Ablations (appendix unless crucial)

- [ ] τ sensitivity (2–3 values)
- [ ] alternative proposal ranking (univariate AUC; shallow tree)
- [ ] stability selection vs |W| ranking
- [ ] SAE variant (TopK+AuxK) if run
- [ ] additional backbone (SigLIP2) if completed

### 5.3 Text sections

- [ ] Intro + framing: discovery workflow; proposal labels vs audit labels disjointness
- [ ] Method: Algorithm box + definitions (Yield(B), AUC_B, AP_j)
- [ ] Experiments: datasets, backbones, training details, compute
- [ ] Results: fish (primary), ADE (sanity), baselines, nulls
- [ ] Limitations: "grounded ≠ novel," expert validation qualitative, taxonomy confounds

## 6) Triage (if time is tight)

- Must-have: protocol + baselines + FishVista + ADE yield curves + permutation null + bootstrap CIs
- Nice-to-have: SigLIP2, TopK+AuxK, extra decompositions (ICA/sparse PCA), deeper robustness tests

