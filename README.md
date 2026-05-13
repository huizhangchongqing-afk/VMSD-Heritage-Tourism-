# VMSD: Structured Visual Annotation Framework

This repository contains the implementation for the research paper:

**VMSD: A Structured Visual Annotation Framework for Detecting Hidden Operational Dissatisfaction in High-Rated Heritage Tourism Reviews**

## Author

**Hui Zhang**  
Department of Information Engineering  
Gingko College of Hospitality Management  
Chengdu, China  

Email: zh20250906@163.com


**VMSD: A Structured Visual Annotation Framework for Detecting Hidden Operational Dissatisfaction in High-Rated Heritage Tourism Reviews**

This repository implements a complete project pipeline for **VMSD: Value–Management Satisfaction Divergence** in heritage-tourism reviews.

The core idea is simple: a review can praise the intrinsic value of a monument/site while still reporting operational-management problems such as crowding, queues, toilets, cleanliness, signage, maintenance, staff behavior, accessibility, pricing, heat, shade, food/water, seating, or safety. Such cases are marked as **VMSD-positive**.

The project follows the research setup used in the VMSD manuscript:

- Dataset size target: **2,000 high-rated English reviews**
- Sites: **Taj Mahal, Red Fort, Qutub Minar, Hampi, Konark Sun Temple**
- Target class distribution: **800 VMSD-positive / 1,200 VMSD-negative**
- Label rule: `IV = 1 AND (OM_text = 1 OR OM_image = 1)`
- Text-image fusion rule: `D_F = 1 - (1 - D_T) * (1 - D_I)`

---

## 1. Folder Structure

```text
VMSD_Heritage_Tourism_Project/
│
├── README.md
├── requirements.txt
├── config/
│   ├── config.yaml
│   └── label_taxonomy.yaml
│
├── dataset/
│   ├── sample_vmsd_reviews.csv
│   └── .gitkeep
│
├── image/
│   └── .gitkeep
│
├── code/
│   ├── run_pipeline.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── make_plots.py
│   └── vmsd/
│       ├── core/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── scoring/
│       ├── modeling/
│       ├── evaluation/
│       ├── visualization/
│       └── utils/
│
├── outputs/
├── notebooks/
│   └── 01_run_vmsd_pipeline_colab.ipynb
└── tests/
```

---

## 2. Installation

### Local VS Code / Terminal

```bash
cd VMSD_Heritage_Tourism_Project
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Google Colab

Upload this folder to Colab or Google Drive, then run:

```python
%cd /content/VMSD_Heritage_Tourism_Project
!pip install -r requirements.txt
```

---

## 3. Expected Dataset Columns

Your main dataset should be placed inside the `dataset/` folder.

Recommended name:

```text
dataset/vmsd_heritage_2000.xlsx
```

Expected columns:

```text
review_id
heritage_site
city_state
review_rating
review_text
review_date
source
image_present
image_url
image_description
intrinsic_value_positive
text_operational_issue
image_operational_issue
operational_aspect
image_evidence_type
text_image_contradiction
vmsd_label
vmsd_severity
vmsd_score
evidence_text_span
annotator_1_label
annotator_2_label
final_label
notes
```

The code is defensive: if manual annotation columns are present, it respects them; if they are missing, it generates predictions using review text, image descriptions, rating, and keyword rules.

---

## 4. Run Full VMSD Pipeline

For the sample dataset:

```bash
python code/run_pipeline.py \
  --input dataset/sample_vmsd_reviews.csv \
  --output outputs/vmsd_predictions.csv
```

For the full 2,000-row Excel dataset:

```bash
python code/run_pipeline.py \
  --input dataset/vmsd_heritage_2000.xlsx \
  --output outputs/vmsd_predictions_2000.csv
```

The output file will include:

```text
pred_intrinsic_value
pred_text_om_issue
pred_image_om_issue
pred_text_confidence
pred_image_confidence
pred_fusion_score
pred_vmsd_label
pred_vmsd_severity
pred_operational_aspects
pred_evidence_source
```

---

## 5. Train Text Classifier

The rule-based pipeline is the main explainable version. A lightweight ML classifier is also included for comparison.

```bash
python code/train_model.py \
  --input dataset/vmsd_heritage_2000.xlsx \
  --model-output outputs/vmsd_text_classifier.joblib
```

This trains a TF-IDF + Logistic Regression classifier using `review_text + image_description`.

---

## 6. Evaluate Predictions

```bash
python code/evaluate_model.py \
  --input outputs/vmsd_predictions_2000.csv \
  --label-column final_label \
  --prediction-column pred_vmsd_label \
  --output outputs/evaluation_report.json
```

If `final_label` is missing, use:

```bash
python code/evaluate_model.py \
  --input outputs/vmsd_predictions_2000.csv \
  --label-column vmsd_label \
  --prediction-column pred_vmsd_label \
  --output outputs/evaluation_report.json
```

---

## 7. Generate Plots

```bash
python code/make_plots.py \
  --input outputs/vmsd_predictions_2000.csv \
  --output-dir outputs/plots
```

Generated plots:

- VMSD label distribution
- Severity distribution
- Operational aspect frequency
- Evidence source distribution
- Site-wise VMSD rate

---

## 8. Project Logic

### Intrinsic Value Detection

A review is considered intrinsic-value positive when:

- rating is 4 or 5, OR
- `intrinsic_value_positive` is already marked Yes/1/True, OR
- text contains heritage-value praise such as beautiful, historic, architecture, monument, spiritual, cultural, UNESCO, heritage, etc.

### Operational Management Detection

Operational problems are detected from:

- review text
- image description
- manual columns, if present

The project uses a HISTOQUAL-aligned taxonomy with categories such as:

```text
crowding, queues, cleanliness, toilets, signage, accessibility,
heat_shade_comfort, pricing_ticketing, staff_behavior, maintenance,
safety, water_food, seating, interpretation
```

### VMSD Decision Rule

```text
VMSD = Yes if IV = 1 and (OM_text = 1 or OM_image = 1)
VMSD = No otherwise
```

### Fusion Score

```text
D_F = 1 - (1 - D_T) * (1 - D_I)
```

Where:

- `D_T` = text operational-management confidence
- `D_I` = image operational-management confidence
- `D_F` = fused multimodal divergence score

---

## 9. Recommended Workflow

1. Put your final Excel file inside `dataset/`.
2. Run `code/run_pipeline.py`.
3. Check `outputs/vmsd_predictions_2000.csv`.
4. Run evaluation if your file has ground-truth labels.
5. Run plots for paper figures.
6. Use the generated metrics and plots in the VMSD report/paper.

---

## 10. Notes

This project is designed for academic/research use. The pipeline is intentionally transparent and rule-explainable because VMSD is not only a classification problem; it is also an evidence-alignment problem between rating, text, and image-derived operational cues.
