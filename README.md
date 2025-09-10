# person-suppression-patch
Adversarial patch training for suppressing YOLO person detections.  
This repo contains code and notebooks to generate, optimize, and evaluate printable adversarial patches that reduce YOLO’s ability to detect people.

---

## Features
- **Patch Training (Colab / PyTorch)**  
  - Top-K anchor focus with K-safe selection  
  - Relative loss vs clean (no-increase constraint)  
  - TV loss (smoothness/printability) + L2 regularization  
  - Torso placement with soft-edge mask (or center fallback)  
- **Evaluation**  
  - Logs loss components, gradient norms, and detection counts  
  - Saves the best-performing patch
- **Live Demo**  
  - Run detection in real time from webcam or video input  

---

## Contents
- `notebooks/PersonDetectionAttack.ipynb` — Main training notebook (Colab-ready).
-  `demo/lgs_live_demo.py` — Live detection script (webcam or video).  
- `requirements.txt` — Python dependencies.  

---

## Training the Adversarial Patch (Colab)

Open `notebooks/PersonDetectionAttack.ipynb` in Google Colab.  
The training loop:

- Samples batches from your dataset  
- Places the patch on the **torso** when a bbox is available (soft edges), otherwise centers it  
- Computes losses on **Top-K person-related anchors**:  
  - **Main:** push down patched person log-scores  
  - **Relative:** prevent patched scores from exceeding clean scores  
  - **TV loss:** smoothness / printability (reduces high-frequency noise)  
  - **L2 loss:** keeps patch values bounded  


## Results

During training the notebook prints:

- Step number and loss components  
- Gradient norms  
- Mean number of detected persons on patched images  

At the end, the best patch is saved to:

```python
patch = best_patch.to(device).detach().clamp(0, 1)




