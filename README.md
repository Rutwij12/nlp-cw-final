# nlp-cw-final


# Patronising and Condescending Language Detection

**Author:** Rutwij Patel  
**CID:** rp1222  
**Trained Model:** https://drive.google.com/drive/folders/1JAe0pxz97rFPXLXx5L5ZScM1dvAavR6w?usp=sharing

This repository contains the implementation for detecting **Patronising and Condescending Language (PCL)** in news paragraphs using transformer-based models.

The project fine-tunes **RoBERTa-base** on the *Don't Patronize Me!* dataset and evaluates performance on the official development split.

All experiments, analysis, and model training are implemented in a single Jupyter notebook.

---

# Model Training

The final model is a **RoBERTa-base binary classifier** trained to predict whether a paragraph contains PCL.

### Input Representation

Each input is formatted as: keyword _s paragraph

where:

- **keyword** – contextual keyword provided in the dataset  
- s - RoBERTa separator token  

Including the keyword provides additional context about the vulnerable group referenced in the text.

---

# Training Configuration

| Parameter | Value |
|------|------|
| Model | `roberta-base` |
| Max sequence length | 128 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Optimiser | AdamW |
| Weight decay | 0.01 |
| Warm-up steps | 500 |

To address the strong class imbalance (~9:1), training uses **weighted cross-entropy loss** with class weights computed from the training distribution.

---

# Threshold Calibration

Instead of using the default probability threshold of **0.5**, the decision threshold is tuned on the development set.

Thresholds between **0.10 and 0.89** are evaluated, and the value that maximises **F1-score for the PCL class** is selected.

---

# Results

Final performance on the development set:

| Metric | Score |
|------|------|
| Precision (PCL) | 0.54 |
| Recall (PCL) | 0.68 |
| F1-score (PCL) | **0.604** |

This significantly improves over the baseline F1 (~0.48) reported for the shared task.

---

# Model Weights

The trained model file (`model.safetensors`) exceeds GitHub size limits.

It can be downloaded here:

**Google Drive:**  
https://drive.google.com/drive/folders/1JAe0pxz97rFPXLXx5L5ZScM1dvAavR6w?usp=sharing

---

# Running the Project

1. Clone the repository

```bash
git clone https://github.com/Rutwij12/nlp-cw-final
cd nlp-cw-final
```

Install dependencies

```bash
pip install transformers datasets scikit-learn pandas matplotlib seaborn
```

Open Notebook

```bash
nlp_cw.ipynb
```

Notes

The notebook reconstructs the official train/dev splits using paragraph IDs to avoid data leakage.

Evaluation uses F1-score for the PCL class, which is appropriate for the highly imbalanced dataset.






