# ELEC475_Lab3

pip install -r requirements.txt











Got it 👍 — based on **Section 2** of your *ELEC 475 Lab 3 (PDF)*, here are ready-to-send **Cursor prompts** to guide you through each of the four steps.
Each is phrased to get high-quality, verifiable code (you can paste them one at a time).

---

### 🧩 **Step 1 – Test Pretrained FCN-ResNet50**

**Prompt for Cursor:**

Write a minimal, clear PyTorch script that:

1. Loads the pretrained `torchvision.models.segmentation.fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")`
2. Downloads and loads the PASCAL VOC 2012 segmentation dataset (`torchvision.datasets.VOCSegmentation`)
3. Applies the proper preprocessing (resize, normalization, ToTensor)
4. Runs the model in evaluation mode on a few validation images
5. Computes mean Intersection-over-Union (mIoU) across the sample batch
6. Displays both the original image and predicted segmentation mask side-by-side

Include comments explaining dataset transforms, tensor shapes, and how the mIoU is computed.
Keep the script self-contained and runnable in `Python 3.11`.

---

### 🧠 **Step 2 – Implement Your Own Model**

**Prompt for Cursor:**

Design a **lightweight semantic segmentation model** for PASCAL VOC 2012 using PyTorch.
Architecture goal: very few parameters but reasonable accuracy.

Use this template description:

* Encoder: `MobileNetV3-Small` pretrained on ImageNet (features up to stride 16)
* Context module: ASPP (Atrous Spatial Pyramid Pooling) with dilation rates {1, 6, 12, 18}
* Decoder: skip connection from low-level features + bilinear upsampling + 1×1 classifier (21 classes)
* Dropout 0.5 before final classifier
* Feature taps (for later KD): low ≈ stride 4, mid ≈ stride 8, high ≈ stride 16

Generate a clean `nn.Module` class with clear docstrings and a `count_parameters(model)` helper.
Ensure it can output both logits and intermediate feature maps.

---

### ⚙️ **Step 3 – Train & Test Your Model**

**Prompt for Cursor:**

Create a training and evaluation script for the custom segmentation model built in Step 2, using the PASCAL VOC 2012 dataset.

Requirements:

* Use `DataLoader`s for train/val with preprocessing (resize to 256×256 or 512×512, normalize to ImageNet stats).
* Define loss = CrossEntropyLoss (ignore_index for background if needed).
* Use Adam optimizer (lr = 1e-4) and a cosine or step LR scheduler.
* Log training loss and validation mIoU every epoch.
* Save best model checkpoint based on val mIoU.
* Provide a `main()` function that can be run as `python train.py`.

Include optional `test_model.py` script to load the checkpoint and evaluate on validation images, saving predicted masks to a folder.

---

### 🔥 **Step 4 – Knowledge Distillation (Teacher–Student)**

**Prompt for Cursor:**

Implement a **knowledge distillation pipeline** where:

* **Teacher:** pretrained FCN-ResNet50 (frozen)
* **Student:** the compact model from Step 2

Implement both:

1. **Response-based distillation**:
   `L_total = α * CE(student_logits, y) + β * KLDivLoss(softmax(student/T), softmax(teacher/T)) * T²`
2. **Feature-based distillation**:
   Cosine similarity loss between low/mid/high feature maps of student and teacher.

Requirements:

* Make α, β, T configurable (α = 1, β = 0.5, T = 4 by default)
* Show full training loop that combines both losses
* Print loss components each epoch (CE, KD, Cosine)
* Save final student model and report mIoU comparison with/without KD.

Include concise comments explaining the mathematics and where gradients flow (teacher frozen).

---

Would you like me to also generate the **exact folder/file structure** (e.g., `models/`, `train.py`, `distill.py`, `utils/metrics.py`) with short placeholder code for each so you can drop it straight into Cursor?
