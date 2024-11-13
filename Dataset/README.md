
## Adversarial Image Dataset for CIFAR-10, CIFAR-100, and ImageNet

This repository contains adversarial image datasets generated for CIFAR-10, CIFAR-100, and ImageNet using three adversarial attack techniques: **FGSM**, **BIM**, and **PGD**. Each dataset includes images at different perturbation levels (ε = 0.01, 0.1, 0.3). These images are organized to facilitate research on adversarial example classification and detection.

Dataset Drive Link: [Google Drive Dataset](https://drive.google.com/drive/folders/1wf1fZ0X9ti1ztGCpQs2JrmgKTrJ0fZWL?usp=sharing)

## File Structure

Each dataset (`cifar10`, `cifar100`, `imagenet`) is organized into separate folders containing benign and adversarial images. Below is the directory structure for each dataset:

Adversarial_dataset_24/
├── cifar10/
│   ├── Benign_cf10/
│   │   └── Benign_cf10/
│   │       └── images/
│   │           └── 5000 benign images
│   └── Adversarial_cf10/
│       ├── BIM/
│       │   └── images/
│       │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_BIM.png
│       │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
│       ├── FGSM/
│       │   └── images/
│       │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_FGSM.png
│       │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
│       └── PGD/
│           └── images/
│               └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_PGD.png
│               └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
├── cifar100/
│   ├── Benign_cf100/
│   │   └── Benign_cf100/
│   │       └── images/
│   │           └── 5000 benign images
│   └── Adversarial_cf100/
│       ├── BIM/
│       │   └── images/
│       │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_BIM.png
│       │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
│       ├── FGSM/
│       │   └── images/
│       │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_FGSM.png
│       │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
│       └── PGD/
│           └── images/
│               └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_PGD.png
│               └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
└── imagenet/
    ├── Benign_imagenet/
    │   └── Benign_imagenet/
    │       └── images/
    │           └── 5000 benign images
    └── Adversarial_imagenet/
        ├── BIM/
        │   └── images/
        │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_BIM.png
        │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
        ├── FGSM/
        │   └── images/
        │       └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_FGSM.png
        │       └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)
        └── PGD/
            └── images/
                └── image_<imageno>_label_<labelno>_ep_<epsilon>_attack_PGD.png
                └── ... (all adversarial images for ε = 0.01, 0.1, 0.3 in a single folder)




### File Naming Convention

Each adversarial image follows this naming format:
image_<imageno>label<labelno>ep<epsilon>attack<attackname>.png
- **`imageno`**: Image number in the dataset.
- **`labelno`**: Label or class number.
- **`epsilon`**: Perturbation level (0.01, 0.1, 0.3).
- **`attackname`**: Name of the adversarial attack (BIM, FGSM, or PGD).

### Dataset Content

Each dataset (`cifar10`, `cifar100`, `imagenet`) includes:
- **Benign Images**: Located in `Benign_<dataset>/images/`, containing 5000 images.
- **Adversarial Images**: Organized into folders by attack type (`BIM`, `FGSM`, `PGD`), with images for each perturbation level (ε = 0.01, 0.1, 0.3) stored together in one folder per attack.

## Guidelines for Selecting Training and Validation Sets

To ensure balanced representation across benign and adversarial images, follow these steps when creating training and validation sets:

### Training Set
1. **Benign Images**: Select 5000 benign images from the `Benign_<dataset>/images/` folder.
2. **Adversarial Images**: Select a total of 5000 adversarial images as follows:
   - **Equal Distribution by Attack Type**: Include an equal number of images for each attack type (`BIM`, `FGSM`, `PGD`).
   - **Equal Distribution by Perturbation Level**: Within each attack type, select images equally across the three perturbation levels (ε = 0.01, 0.1, 0.3).
3. The resulting training set will contain 5000 benign images and 5000 adversarial images, with balanced representation of each attack and perturbation level.

### Validation Set
1. **Benign Images**: Select 1000 benign images from the `Benign_<dataset>/images/` folder.
2. **Adversarial Images**: Select a total of 1000 adversarial images as follows:
   - **Equal Distribution by Attack Type**: Include an equal number of images for each attack type (`BIM`, `FGSM`, `PGD`).
   - **Equal Distribution by Perturbation Level**: Within each attack type, select images equally across the three perturbation levels (ε = 0.01, 0.1, 0.3).
3. The resulting validation set will contain 1000 benign images and 1000 adversarial images, maintaining balance across attacks and perturbation levels.

## Usage

This dataset is intended for research purposes, particularly in studying adversarial attacks and their impact on image classification models. You are free to use and modify this dataset in accordance with the license provided.

For further inquiries or issues, please contact the repository maintainer.

---
