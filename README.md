# 🦴 NutriBone — Knee Osteoporosis Classifier

**Classify knee X-rays into Normal, Osteopenia and Osteoporosis using a 4-model deep learning ensemble**

## Live Demo
[Coming soon after deployment]

## About
NutriBone uses an ensemble of 4 pretrained deep learning models to classify knee X-ray images into 3 clinical stages of bone density:

- 🟢 **Normal** — Healthy bone density
- 🟠 **Osteopenia** — Early bone density loss (pre-osteoporosis stage)
- 🔴 **Osteoporosis** — Significant bone density loss with high fracture risk

## Model Architecture
| Model | Parameters | Fine-tuned Layers |
|-------|-----------|-------------------|
| VGG-19 | 143M | Block 4 + Block 5 |
| ResNet50 | 25M | Last 40 layers |
| EfficientNetB0 | 5.3M | Top 30% |
| ResNet50V2 | 25M | Last 40 layers |

**Ensemble weights:** VGG-19=0.6, ResNet50=1.0, EfficientNetB0=0.2, ResNet50V2=0.2

## Performance
| Dataset | Accuracy |
|---------|----------|
| Own test set (3-class) | 89.46% |
| Sachinkumar external (binary) | 93.01% |
| Orvile external (3-class) | 61.92% |

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Trained on combined Kaggle datasets:
- mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset (1,947 images)
- stevepython/osteoporosis-knee-xray-dataset (372 images)

Total: 2,319 images balanced to 724 per class using augmentation.

## License
cc-by-nc-4.0 — Free for non-commercial and research use only.

## Disclaimer
⚠️ This application is for **research and educational purposes only**.
It is NOT intended for clinical diagnosis.
Always consult a qualified medical professional.
