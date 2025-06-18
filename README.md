# CV-Final  
**Final Deliverable — Computer Vision Project @ Heidelberg University (2025)**

This repository contains the pipelines and experiments for our Computer Vision project focused on skin lesion segmentation, with special attention to preprocessing techniques aimed at achieving dataset diversity.

For a summary on the achieved results please refer to the PDF report.
---

## 📋 Installation

To set up the environment, make sure you have Python **3.10.7** installed. Then, create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---
## 🧪 Overview

All evaluation results and analysis are provided in `demo.ipynb`, which includes tests on both:

- **ISIC-Merged** test set  
- **Waterloo Dark Skin** test set  

These demonstrate the generalization and robustness of our models across different skin tones and datasets.

For detailed methodology, results, and discussion, please refer to `report.pdf`.

---

## 🏗️ Components

### 🧼 Preprocessing
- `normalize_images.py`: Contains preprocessing functions including top-hat filter for hair removal and resizing of ISIC-Merged.

### 🔧 Augmentation
- `augment_base.py`: Standard augmentations used to build Case 2 and Case 3 datasets.
- `augment_DS.py`: Augmentation pipeline tailored for DS1 and DS2 to enrich datasets with dark skin images (Case 4).

### 🧠 Architectures
- All investigated network architectures (custom and adapted from literature) are located in the `unets/` directory.

### 🏋️ Training
- `train.ipynb`: Script used to train all models across datasets and augmentation configurations.
- Utility functions and metrics used to train the models with the Tensorflow/Keras framework are located in the `tf_utils/` directory.

---

## 📦 Pretrained Models

All trained models are available in the `Models/` directory.  

---

## 📌 Notes

- All tests and figures shown in `demo.ipynb` are reproducible with the code and models provided.
- Results highlight the importance of preprocessing and dataset diversity in skin lesion classification.

---

## 👥 Authors

- **Luca Barda** — [luca.barda@mail.polimi.it](mailto:luca.barda@mail.polimi.it)  
- **Damiano Baschiera** — [damiano.baschiera@mail.polimi.it](mailto:damiano.baschiera@mail.polimi.it)  

For any questions, feel free to reach out via email.

---

Feel free to clone and explore! For any questions or to reproduce results, start with `demo.ipynb`.
