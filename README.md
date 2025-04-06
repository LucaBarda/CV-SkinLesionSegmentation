# CV-Final  
**Final Deliverable — Computer Vision Project @ Heidelberg University (2025)**

This repository contains the full pipeline and experiments for our Computer Vision project focused on skin lesion classification, with special attention to dataset diversity and preprocessing techniques.

## 🧪 Overview

All evaluation results and analysis are provided in `demo.ipynb`, which includes tests on both:

- **ISIC-Merged** test set  
- **Waterloo Dark Skin** test set  

These demonstrate the generalization and robustness of our models across different skin tones and datasets.

For detailed methodology, results, and discussion, please refer to `report.pdf`.

---

## 🏗️ Components

### 🔧 Augmentation
- `augment_DS.py`: Augmentation methods tailored for DS1 and DS2 datasets with dark skin images.
- `augment_base.py`: Standard baseline augmentations used in Case 2 for model comparison.

### 🧼 Preprocessing
- `normalize_images.py`: Contains preprocessing functions including image normalization and a top-hat filter for hair artifact removal.

### 🧠 Architectures
- All network architectures (custom and adapted from literature) are located in the `units/` directory.

### 🏋️ Training
- `train.py`: Script used to train all models across datasets and augmentation configurations.

---

## 📦 Pretrained Models

All trained models are available in our shared drive folder.  
👉 **[Download Models Here](<insert-your-link-here>)**

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
