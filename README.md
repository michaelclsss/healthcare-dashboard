# 🏥 Hospital Readmission Predictor

A machine learning pipeline that predicts 30-day hospital readmissions at the point of patient discharge — helping clinical teams intervene before patients end up back in the emergency room.

---

## 📋 Overview

When a patient is discharged, this system scans their records and outputs a readmission risk score. High-risk patients can then receive proactive follow-ups: a phone call, a medication review, or a scheduled check-in — before something goes wrong.

**The core problem:** ~11% of hospital patients are readmitted within 30 days. These readmissions are costly for hospitals (who face financial penalties from insurers) and harmful for patients. The goal is systematic, scalable early detection.

---

## 📊 Dataset

- ~100,000 patient records
- Features include: age, length of stay, number of medications, prior admissions, diabetes status, and more
- Target variable: readmitted within 30 days (binary: yes / no)

### Data Cleaning Steps

| Issue | Fix |
|---|---|
| Age stored as ranges like `[70-80)` | Converted to midpoint values (e.g., `75`) |
| Unknown values marked as `?` | Treated as `NaN` and handled as missing data |
| Weight column missing for 97% of patients | Dropped entirely |
| No medication change signal | Engineered a new feature: **number of medication changes** during the visit |

---

## ⚖️ Handling Class Imbalance

The dataset is heavily skewed — 89% of patients were *not* readmitted. A naive model that always predicts "not readmitted" would be 89% accurate but completely useless.

Two approaches were tested:

- **SMOTE** — Synthetically generates minority-class samples until both classes are balanced
- **Sample Weights** ✅ *(final approach)* — Assigns higher weight to readmitted patients during training (e.g., weight of 8×), so the model takes them more seriously without fabricating data

---

## 🤖 Model: Gradient Boosting

The model builds **300 sequential decision trees**, where each tree focuses on correcting the errors of the previous one. Think of it as an iterative self-correcting process — by tree 300, the model has systematically addressed nearly every pattern in the training data.

Each tree asks clinical questions like:
- Was the patient hospitalized more than 3 times in the past year?
- Are they on more than 10 medications?
- Did their medication regimen change during this visit?

The final prediction is the combined output of all 300 trees.

---

## 📈 Results

The model was evaluated on a **held-out test set (20% of data)** — patients the model had never seen during training.

| Metric | Score | What It Means |
|---|---|---|
| **AUC** | 0.684 | Given one readmitted and one non-readmitted patient, the model ranks the right one higher 68.4% of the time (50% = random chance) |
| **Recall** | 62% | Of all patients who were actually readmitted, the model flags 62% of them |
| **Precision** | 18% | When the model flags someone as high-risk, it's correct 18% of the time |
