# Efficient Computer Vision Models for Silkworm Feeding Prediction and Habitat Analysis

## 1. Project Title
**Efficient Computer Vision Models for Silkworm Feeding Prediction and Habitat Analysis**

## 2. Description
The project focuses on binary classification, where the goal is to implement lightweight neural network architectures to determine whether the silkworms need feeding. Another objective involves unsupervised segmentation techniques designed to automatically distinguish and separate the three key elements present in the images:silkworms, mulberry leaves, and background. By applying non-supervised methods, the project seeks to extract more detailed insights from the rearing environment without requiring pixel-wise annotations for training. Both tasks will be complemented by data augmentation strategies, improving the models’ ability to generalize across different conditions and making them more robust to variations in environmental factors.

The primary objectives are:
*   **Rearing Classification**: To implement and evaluate lightweight neural network architectures (such as EfficientNetV2) for binary classification to determine if silkworms require feeding.
*   **Unsupervised Segmentation**: To apply and compare various unsupervised segmentation techniques to automatically identify and isolate silkworms, mulberry leaves, and background elements in images without needing manually annotated masks for training.
*   *   **Performance Evaluation**: Assess classification models and segmentation outputs using quantitative metrics and qualitative analysis to determine their effectiveness in real-world conditions.

The project explores and compares several state-of-the-art and classical computer vision methods:
*   **Binary Classification**: EfficientNetV2-S, MobileVit and RepNext with Grad-CAM for model interpretability.
*   **Supervised Segmentation**: SegFormer trained on a small, manually annotated dataset.
*   **Unsupervised Segmentation**:
    *   DINOv2 Feature Extraction + KMeans Clustering
    *   Segment Anything Model (SAM)
    *   Watershed Algorithm
    *   KMeans Clustering in LAB color space



## 3. How to Install and Run

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Git

### Dependencies
Clone the repository and install the required Python packages.

```bash
git clone https://github.com/AriannaViola/Silkworm-CV.git
cd AriannaViola/Silkworm-CV
pip install -r requirements.txt
```

A `requirements.txt` file should be created with the following content:
```
numpy
pandas
opencv-python-headless
matplotlib
Pillow
torch
torchvision
scikit-learn
seaborn
tqdm
requests
timm
segment-anything-py
transformers
```

### Special Installations

**1. RepNeXt Model (for `project8-cv.ipynb`)**
The `RepNeXt` model is loaded via the `timm` library but requires cloning its source repository first. The main notebook (`project8-cv.ipynb`) includes a script to do this automatically.

**2. Segment Anything Model (SAM) Checkpoint (for `sam-masks.ipynb`)**
The SAM notebook will automatically download the required model checkpoint (`sam_vit_b_01ec64.pth`). Ensure you have an active internet connection on the first run.

### Dataset Structure
Place your dataset in a root folder (e.g., `data/`). The scripts expect the following structure, based on the Kaggle environment paths:

```/data/
└── silk_dataset/
    ├── 0_data.csv
    ├── IMG_xxxx.jpg
    ├── IMG_yyyy.jpg
    └── ...
```
The final comparison notebook (`project8-cv.ipynb`) also requires the output masks from the other segmentation notebooks. These should be placed in their respective folders as specified within that notebook's code.

### Running the Notebooks
The project is organized into several Kaggle Notebooks. It is recommended to run them in the following order to generate all necessary components for the final analysis.

1.  **Generate Segmentation Masks (Unsupervised Methods):**
    *   `watershed.ipynb`: Generates masks using the Watershed algorithm.
    *   `sam-masks.ipynb`: Generates masks using the Segment Anything Model.
    *   `kmeans-lab-5.ipynb`: Generates masks using KMeans clustering.
2.  **Train Supervised Models (Optional but recommended):**
    *   `labelme-segformer.ipynb`: Contains scripts to convert LabelMe JSON annotations to PNG masks and then trains a SegFormer model on them.
3.  **Run Main Analysis:**
    *   `project8-cv.ipynb`: This is the main notebook. It trains the classification model, performs unsupervised segmentation using DINOv2, and runs a final comparison of all generated segmentation masks.

## 4. How to Use the Project

Each notebook is a self-contained script for a specific task. Open them in a Jupyter environment and run the cells sequentially.

*   **`project8-cv.ipynb`**: This is the core notebook.
    *   It trains a lightweight classifier to predict feeding needs.
    *   It uses Grad-CAM to visualize what the classifier focuses on.
    *   It performs unsupervised segmentation using DINOv2 features and KMeans.
    *   It loads the masks generated by all other notebooks to provide a final side-by-side visual comparison of segmentation techniques.
    *   Load (as a dataset) all the masks located in the folder named "Masks" (generated by us, load both "sam_generated_masks-2.zip", "sam_generated_masks.zip", "watershed_masked.zip", "labelme_generated_masks.zip", "kmeans_multiclass_masks.zip", "dinomasks' will be generated after running the code) or generate yours by running all other notebooks to provide a final side-by-side visual comparison of segmentation techniques.

*   **`watershed.ipynb`**: Applies the classical Watershed algorithm for segmentation. The output is a set of masks where different colors represent worms and leaves.

*   **`sam-masks.ipynb`**: Uses the powerful Segment Anything Model (SAM) to generate instance masks for all objects in the image, which are then classified by color into leaf or worm categories.

*   **`kmeans-lab.ipynb`**: Segments images by clustering pixel colors in the perceptually uniform CIELAB color space. This is a fast and effective unsupervised method.

*   **`labelme-segformer.ipynb`**: Demonstrates a supervised approach.
    1.  First, it converts polygon annotations from LabelMe's JSON format into PNG masks.
    2.  Then, it fine-tunes a `SegFormer` transformer model on this small, manually-labeled dataset.
    3.  To run this code is necessary to upload as a dataset all the images located in the folder named "Images".

## 5. Collaborators
*   Arianna Viola
*   Rossella Milici

## 6. Documentation
For a deeper understanding of the models and libraries used in this project, please refer to their official documentation and research papers:
*   **PyTorch**: [Official Documentation](https://pytorch.org/docs/stable/index.html)
*   **OpenCV**: [Official Documentation](https://docs.opencv.org/4.x/)
*   **Hugging Face Transformers (SegFormer, DINOv2)**: [Documentation](https://huggingface.co/docs/transformers/index)
*   **Segment Anything Model (SAM)**: [Research Paper](https://arxiv.org/abs/2304.02643)
*   **EfficientNetV2**: [Research Paper](https://arxiv.org/abs/2104.00298)
*   **DINOv2**: [Research Paper](https://arxiv.org/abs/2304.07193)


## 7. References 
* 1. Zhao, M., Luo, Y., and Ouyang, Y. (2024). RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization. arXiv.

* 2. Tan, M., and Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv.

* 3. Mehta, S., and Rastegari, M. (2022). MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. arXiv.

* 4. Rossetti, S., Sam`a, N., and Pirri, F. (2023). Removing supervision in semantic segmentation with local-global matching and area balancing. arXiv.

* 5. Niu, D., Wang, X., Han, X., Lian, L., Herzig, R., and Darrell, T. (2023). Unsupervised Universal Image Segmentation. arXiv.


