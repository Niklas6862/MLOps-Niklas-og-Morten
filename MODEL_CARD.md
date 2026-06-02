# Model Card — Bean Leaf Disease Classifier

## Model Details

| Field | Value |
|---|---|
| **Model name** | Bean Leaf Disease Classifier |
| **Base model** | `google/vit-base-patch16-224` |
| **Architecture** | Vision Transformer (ViT-Base, patch size 16, input 224×224) |
| **Task** | Image classification (3 classes) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Developed by** | Niklas & Morten |
| **Version** | 1.0 (draft) |
| **License** | For academic/research use only |

---

## Intended Use

**Intended use:** Decision-support tool for automated detection of bean leaf diseases from photographs. Intended to assist farmers or agronomists in early identification of plant disease.

**Out-of-scope uses:**
- Other crop types or plant species
- Non-leaf images
- Autonomous decision-making without human review
- Real-time edge deployment on embedded hardware

---

## Training Data

| Field | Value |
|---|---|
| **Dataset** | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) |
| **Source** | HuggingFace Hub |
| **Classes** | `angular_leaf_spot`, `bean_rust`, `healthy` |
| **Total images** | ~1,295 (train: 1,034 / validation: 133 / test: 128) |
| **Image type** | RGB photographs of bean leaves |
| **Preprocessing** | Resized to 224×224, normalised using ViT processor mean/std |

---

## Training Procedure

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-5 (AdamW) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Train batch size | 16 |
| Eval batch size | 32 |
| Mixed precision (fp16) | False |
| Best model criterion | Highest validation accuracy |

Experiments were tracked with MLflow. All hyperparameters are defined in `configs/training.yaml`.

---

## Evaluation Results

| Metric | Value |
|---|---|
| Overall accuracy (test) | _TBD_ |
| angular\_leaf\_spot — F1 | _TBD_ |
| bean\_rust — F1 | _TBD_ |
| healthy — F1 | _TBD_ |

Minimum accepted accuracy threshold for deployment: **0.80** (enforced in the Jenkins pipeline).

---

## Limitations

- Model was trained on research-grade photographs; performance on low-quality field photos is not validated.
- Softmax outputs can be overconfident under distribution shift — raw confidence scores should not be trusted as calibrated probabilities.
- The label set is fixed at training time; adding a new disease class requires full retraining.
- Dataset is relatively small (~1,000 images per class), which may limit generalisation.

---

## Ethical Considerations

Incorrect predictions may cause farmers to misapply or withhold treatment, with direct economic consequences. The model should be used as a supplementary tool alongside expert agronomic judgement, not as a sole decision-maker.

---

## Infrastructure

| Component | Detail |
|---|---|
| Containerisation | Docker (multi-stage build) |
| Registry | Private Docker registry (`172.24.198.42:5000`) |
| Experiment tracking | MLflow (`http://172.24.198.42:5050`) |
| CI/CD | Jenkins (lint → test → build → train → evaluate → register → deploy) |
| Dependency management | `uv` + `pyproject.toml` |
