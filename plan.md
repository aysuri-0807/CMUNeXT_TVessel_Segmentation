# Research Plan: CMUNeXt + Image Processing for Efficient Retinal Thin-Vessel Segmentation

## Summary

Study whether a lightweight CMUNeXt-based pipeline can achieve clinically useful retinal vessel segmentation performance while using substantially fewer resources than heavier state-of-the-art thin-vessel models. The paper's core claim should be an accuracy-efficiency tradeoff: comparable segmentation quality, especially on thin vessels, with lower parameters, FLOPs, memory, and inference time.

Use retinal fundus vessel segmentation as the first scope because it has mature public benchmarks and many comparable baselines: DRIVE, STARE, CHASE_DB1, and HRF.

## Research Questions

- Can vanilla CMUNeXt perform competitively on thin retinal vessel segmentation after careful training?
- Which image-processing refinements most improve thin-vessel continuity and boundary quality?
- Does the hybrid CMUNeXt pipeline provide a better accuracy/resource tradeoff than heavier vessel-specific state-of-the-art models and recent lightweight models?
- Where does the method fail: ultra-thin capillaries, low contrast, lesions, crossings, central thick vessels, or peripheral vessels?

## Method

- Establish baselines:
  - U-Net or U-Net++ as classical baseline.
  - CMUNeXt as lightweight baseline.
  - At least one recent retinal-vessel state-of-the-art or strong reported comparator from literature, such as FSG-Net, DA-U2Net, VasCA-Net, WDM-UNet, or DSAE-Net.
  - Include reported results when reproduction is infeasible, but reproduce at least U-Net and CMUNeXt under identical conditions.

- Build the proposed hybrid pipeline:
  - Pre-processing: green-channel extraction or contrast normalization, CLAHE, gamma/illumination correction, and optional vesselness filtering.
  - Model: CMUNeXt trained for binary vessel segmentation.
  - Losses: Dice + BCE as baseline; test vessel-aware alternatives such as Focal, Tversky, or clDice for thin-structure preservation.
  - Post-processing: threshold calibration, connected-component filtering, skeleton-preserving cleanup, morphological closing/opening, and optional CRF or vessel-continuity refinement.
  - Ablate each image-processing stage so the paper can show which pieces actually matter.

- Evaluate thin-vessel behavior explicitly:
  - Standard metrics: Dice/F1, IoU, sensitivity/recall, specificity, accuracy, AUC.
  - Efficiency metrics: parameters, FLOPs/MACs, GPU memory, inference time per image, training time.
  - Thin-vessel metrics: skeleton F1, clDice, centerline recall, and performance split by vessel width if masks can be skeletonized/dilated into width bands.

## Experimental Design

- Datasets:
  - Primary: DRIVE and CHASE_DB1.
  - Secondary validation: STARE and HRF if time permits.
  - Use official splits where available; otherwise use fixed documented splits and report seeds.

- Training protocol:
  - Same preprocessing resolution, augmentations, optimizer, scheduler, epochs, and early stopping across reproduced models.
  - Use 3 to 5 random seeds for the final comparison.
  - Tune decision threshold on validation data, not test data.
  - Keep all hardware and runtime settings documented.

- Ablations:
  - CMUNeXt only.
  - CMUNeXt + preprocessing.
  - CMUNeXt + vessel-aware loss.
  - CMUNeXt + post-processing.
  - Full hybrid pipeline.
  - Optional: compare small/medium CMUNeXt width variants if available.

## Paper Structure

- Introduction:
  - Motivate thin-vessel segmentation as accuracy-sensitive and resource-intensive.
  - Present the hypothesis: lightweight segmentation plus classical image processing may recover much of state-of-the-art performance.

- Related Work:
  - Retinal vessel segmentation.
  - Lightweight medical segmentation.
  - Thin-structure losses and skeleton-aware metrics.
  - CMUNeXt and efficient U-shaped architectures.

- Methods:
  - Dataset preparation.
  - CMUNeXt architecture summary.
  - Hybrid image-processing pipeline.
  - Training losses and post-processing.
  - Evaluation protocol.

- Results:
  - Main metric table across datasets.
  - Efficiency table.
  - Ablation table.
  - Thin-vessel-specific analysis.
  - Qualitative examples: success cases and failure cases.

- Discussion:
  - Whether comparable performance is achieved.
  - Resource savings and deployment implications.
  - Limits of post-processing.
  - Generalization to other thin-vessel modalities.

## Acceptance Criteria

- The final paper claim is supported if the hybrid CMUNeXt pipeline:
  - Comes within a small margin of strong retinal-vessel methods on Dice/F1/AUC.
  - Shows clearly lower parameters, FLOPs, memory, or inference time.
  - Improves thin-vessel continuity over vanilla CMUNeXt.
  - Demonstrates the improvement through ablations, not only final scores.

## Assumptions

- Target modality is retinal fundus vessel segmentation.
- Main contribution is a hybrid CMUNeXt + image-processing pipeline, not a new architecture.
- Success means a favorable efficiency tradeoff, not necessarily beating absolute state of the art.
- Public datasets are sufficient for the first paper version.
- Sources to ground the literature review include [CMUNeXt](https://arxiv.org/abs/2308.01239), recent retinal vessel work such as [FSG-Net](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-025-02021-4), [DA-U2Net](https://link.springer.com/article/10.1186/s12886-025-03908-0), and lightweight retinal-vessel work such as [DSAE-Net](https://pubmed.ncbi.nlm.nih.gov/41003356/).
