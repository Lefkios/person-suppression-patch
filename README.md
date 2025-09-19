# person-suppression-patch

Adversarial patch training for suppressing YOLO person detections.  
This repo provides code and notebooks to **generate, optimize, and evaluate printable adversarial patches** that reduce YOLO’s ability to detect people, along with a **live demo environment** to test defenses in real time.

---

## ✨ Features

- **Patch Training (Colab / PyTorch)**  
  - Top-K anchor focus with K-safe selection  
  - Relative loss vs clean (no-increase constraint)  
  - Total Variation (TV) loss for smoothness & printability  
  - L2 regularization to bound pixel values  
  - Torso placement with soft-edge mask (or center fallback)  

- **Evaluation**  
  - Logs: loss components, gradient norms, detection counts  
  - Automatically saves the best-performing patch  

- **Live Defense Demo (OpenCV + YOLOv8)**  
  - Real-time person detection from webcam or video input  
  - Toggle adversarial defenses on the fly:
    - **Strong Gaussian Blur**  
    - **JPEG compression**  
    - **LGS stripe smoothing**  
  - Clean **overlay panel** for switching defenses & monitoring confidence  

---

## 📂 Contents

- `notebooks/PersonDetectionAttack.ipynb` — Main adversarial patch training notebook (Colab-ready).  
- `demo/lgs_live_demo.py` — Live YOLO detection + defense demo (webcam/video).  
- `requirements.txt` — Python dependencies.  

---

## 🚀 Training the Adversarial Patch (Colab)

Open `notebooks/PersonDetectionAttack.ipynb` in Google Colab.  

The training loop:  
- Samples batches from your dataset  
- Places the patch on the **torso** when a bbox is available (with soft edges), otherwise centers it  
- Computes losses on **Top-K person-related anchors**:  
  - **Main Loss:** suppress patched person log-scores  
  - **Relative Loss:** prevent patched scores from exceeding clean scores  
  - **TV Loss:** enforce smoothness/printability (reduces high-frequency noise)  
  - **L2 Loss:** keeps patch values bounded  

The notebook prints per step:  
- Step number + loss breakdown  
- Gradient norms  
- Mean number of detected persons on patched images  

At the end, the best patch is saved to:

```python

patch = best_patch.to(device).detach().clamp(0, 1)

```
## 🎥 Running the Live Demo

You can run the real-time demo with webcam

```bash
python demo/lgs_live_demo.py
```

### Controls
- `0` → No defense  
- `1` → Strong Gaussian Blur  
- `2` → JPEG compression  
- `3` → LGS stripe smoothing  
- `q` → Quit  

### The demo displays:
- **Defense Panel** (bottom-left) with active mode highlighted  
- **Smoothed confidence percentage** (bottom-right)  

This is ideal for **live presentations** (e.g. Researchers’ Night) to show how adversarial patches affect YOLO and how simple defenses mitigate attacks.




