import sys, os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

# =================== CONFIG ===================
MODEL_PATH   = "yolov8n.pt"
IMG_SIZE     = 640

CONF_BASE    = 0.65
device = "cpu"

# =================== LGS params ===================
LGS_SIGMA      = 2.5
LGS_THRESH     = 0.03
LGS_MORPH      = 3
LGS_DILATE_IT  = 2
LGS_ITERS      = 2
LGS_MULTISCALE = True
LGS_DEBUG      = False

FEATHER_PX     = 6
BG_SOFTEN      = 0.0
EXACT_BOX_ONLY = False

# ===== LGS stripe =====
STRIPE_ORIENT   = "horizontal"
STRIPE_FRACTION = 1/3

# =================== JPEG defense params ===================
JPEG_QUALITY = 25   

# ===== Smoothing for confidence meter =====
SMOOTH_ALPHA = 0.25   # 0..1 (lower = smoother, less sudden)

# =================== MODEL ====================
model = YOLO(MODEL_PATH)
to_tensor = T.ToTensor()
resize640 = T.Resize((IMG_SIZE, IMG_SIZE))

# ================= CAMERA / VIDEO ==============
VIDEO_SOURCE = sys.argv[1] if len(sys.argv) > 1 else None

def open_camera(preferred_index=0):
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    tried = set()
    for backend in backends:
        for idx in [preferred_index]:
            key = (backend, idx)
            if key in tried:
                continue
            tried.add(key)
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"[cam] opened index={idx} backend={backend}")
                return cap
            cap.release()
    return None

if VIDEO_SOURCE and os.path.exists(VIDEO_SOURCE):
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {VIDEO_SOURCE}")
    print(f"[vid] playing file: {VIDEO_SOURCE}")
else:
    cap = open_camera(preferred_index=0)
    if cap is None:
        raise RuntimeError(
            "No camera opened. Close Teams/Zoom/OBS/Camera app and try again.\n"
            "Or run with a video file:  python phys_patch_lgs_stripe.py path\\to\\video.mp4"
        )

# =================== HELPERS ===================
def tensor_to_uint8(img_t):
    arr = (img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)

# Photometric normalization (subtle)
def gray_world_awb_subtle(img_rgb_u8, strength=0.25):
    mean = img_rgb_u8.mean(axis=(0,1), keepdims=True).astype(np.float32) + 1e-6
    gray = mean.mean(axis=2, keepdims=True)
    gain = np.clip(gray / mean, 0.8, 1.2)
    awb = np.clip(img_rgb_u8.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.addWeighted(img_rgb_u8, 1.0-strength, awb, strength, 0)

def clahe_on_L_subtle(img_rgb_u8, clip=1.5, strength=0.25):
    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2,a,b])
    clahe_rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return cv2.addWeighted(img_rgb_u8, 1.0-strength, clahe_rgb, strength, 0)

def adaptive_gamma_subtle(img_rgb_u8, min_g=0.85, max_g=1.15, strength=0.3):
    yuv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2YUV)
    Y = yuv[:,:,0].astype(np.float32) / 255.0
    meanY = max(1e-6, float(Y.mean()))
    gamma = np.log(0.5) / np.log(meanY)
    gamma = float(np.clip(gamma, min_g, max_g))
    table = ((np.arange(256)/255.0) ** gamma * 255.0).clip(0,255).astype(np.uint8)
    gimg = cv2.LUT(img_rgb_u8, table)
    return cv2.addWeighted(img_rgb_u8, 1.0-strength, gimg, strength, 0)

def photometric_normalize_subtle(img_rgb_u8):
    x = gray_world_awb_subtle(img_rgb_u8, strength=0.2)
    x = clahe_on_L_subtle(x, clip=1.3, strength=0.2)
    x = adaptive_gamma_subtle(x, min_g=0.9, max_g=1.1, strength=0.25)
    return x

# =================== LGS core ===================
def _strong_lgs_numpy(img_np, mask_bin,
                      sigma=2.5, threshold=0.03,
                      morph_kernel_size=3, dilate_iters=2,
                      iters=2, multiscale=True, debug=False,
                      exact_box_only=False):
    H, W, _ = img_np.shape
    work = img_np.copy()
    roi = (mask_bin.astype(np.uint8) > 0).astype(np.uint8)
    if exact_box_only:
        mask = roi.copy()
    else:
        gray = cv2.cvtColor((work * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        def grad_norm(gray_u8, k):
            gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=k)
            mag = np.sqrt(gx**2 + gy**2)
            return mag / (mag.max() + 1e-8)
        g3 = grad_norm(gray, 3)
        g  = np.maximum(g3, grad_norm(gray, 5)) if multiscale else g3
        grad_mask = (g > threshold).astype(np.uint8)
        mask = (roi & grad_mask)
        if morph_kernel_size and morph_kernel_size > 0:
            k = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = (mask & roi)
        if dilate_iters and dilate_iters > 0:
            k2 = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, k2, iterations=int(dilate_iters))
            mask = (mask & roi)
    for _ in range(max(1, int(iters))):
        blurred = cv2.GaussianBlur(work, (0, 0), sigmaX=sigma, sigmaY=sigma)
        work = work * (1 - mask[:, :, None]) + blurred * mask[:, :, None]
    return work

def apply_lgs_stripe(image_tensor: torch.Tensor,
                     sigma=LGS_SIGMA, threshold=LGS_THRESH,
                     morph_kernel_size=LGS_MORPH, dilate_iters=LGS_DILATE_IT,
                     iters=LGS_ITERS, multiscale=LGS_MULTISCALE,
                     feather_px=FEATHER_PX, bg_soften=BG_SOFTEN,
                     exact_box_only=EXACT_BOX_ONLY,
                     orient=STRIPE_ORIENT, fraction=STRIPE_FRACTION,
                     debug=LGS_DEBUG):
    C, H, W = image_tensor.shape
    img_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    fraction = max(0.01, min(0.99, float(fraction)))
    if orient.lower().startswith("v"):
        stripe_w = int(W * fraction)
        x1 = (W - stripe_w) // 2
        x2 = x1 + stripe_w
        y1, y2 = 0, H
    else:
        stripe_h = int(H * fraction)
        y1 = (H - stripe_h) // 2
        y2 = y1 + stripe_h
        x1, x2 = 0, W
    roi_mask[y1:y2, x1:x2] = 1
    if FEATHER_PX > 0:
        ksize = max(1, int(FEATHER_PX * 2 + 1))
        alpha = cv2.GaussianBlur(roi_mask.astype(np.float32), (ksize, ksize), FEATHER_PX)
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = roi_mask.astype(np.float32)
    base_np = img_np
    hard_mask = (alpha > 0.01).astype(np.uint8)
    lgs_np = _strong_lgs_numpy(
        img_np, hard_mask,
        sigma=sigma, threshold=threshold,
        morph_kernel_size=morph_kernel_size, dilate_iters=dilate_iters,
        iters=iters, multiscale=multiscale, debug=debug,
        exact_box_only=exact_box_only
    )
    result_np = base_np * (1 - alpha[:, :, None]) + lgs_np * alpha[:, :, None]
    out_t = torch.from_numpy(result_np.transpose(2, 0, 1)).float().to(image_tensor.device)
    return out_t.clamp(0, 1)

# ============== JPEG & strong Gaussian blur (full-res) ==============
def apply_jpeg_defense_rgb_u8(rgb_u8, quality):
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return rgb_u8
    dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    out_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb

def apply_strong_jpeg_defense_rgb_u8(rgb_u8, quality, down_scale=0.7, median_ksize=3):
    h, w = rgb_u8.shape[:2]
    dw, dh = max(8, int(w*down_scale)), max(8, int(h*down_scale))
    ds = cv2.resize(rgb_u8, (dw, dh), interpolation=cv2.INTER_AREA)
    ds = apply_jpeg_defense_rgb_u8(ds, quality)
    if median_ksize and median_ksize > 1:
        ds_bgr = cv2.cvtColor(ds, cv2.COLOR_RGB2BGR)
        ds_bgr = cv2.medianBlur(ds_bgr, median_ksize)
        ds = cv2.cvtColor(ds_bgr, cv2.COLOR_BGR2RGB)
    up = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
    return up

def apply_bitdepth_rgb_u8(rgb_u8, bits=5):
    shift = 8 - bits
    return ((rgb_u8 >> shift) << shift).astype(np.uint8)

def apply_strong_gaussian_blur_rgb_u8(rgb_u8, ksize=15, sigma=3.5):
    k = max(1, int(ksize)); k += 1 - (k % 2)
    return cv2.GaussianBlur(rgb_u8, (k, k), sigmaX=sigma, sigmaY=sigma)

# ===== Confidence percentage (bottom-right, smoothed) =====
def draw_conf_percent(img_rgb, conf_value):
    h, w = img_rgb.shape[:2]
    pad = 12
    text = f"{conf_value*100:5.1f}%"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = w - pad - tw
    y = h - pad
    cv2.putText(img_rgb, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_rgb, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def _rounded_rect_aa(img, p1, p2, radius=14,
                     fill=(18,18,18), alpha=0.58,
                     border=1, border_color=(230,230,230),
                     shadow=True, shadow_offset=(3,4), shadow_alpha=0.25):

    x1, y1 = p1; x2, y2 = p2
    r = int(max(1, min(radius, (x2 - x1)//2, (y2 - y1)//2)))

    base = img

    # optional shadow
    if shadow:
        sh = np.zeros_like(base, dtype=np.uint8)
        dx, dy = shadow_offset
        cv2.rectangle(sh, (x1+dx, y1+dy), (x2+dx, y2+dy), (0,0,0), -1, cv2.LINE_AA)
        cv2.addWeighted(sh, shadow_alpha, base, 1.0, 0, base)

    # draw filled rounded rect on overlay
    overlay = base.copy()
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), fill, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1+r, y1+r), r, fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x2-r, y1+r), r, fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1+r, y2-r), r, fill, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x2-r, y2-r), r, fill, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, base, 1-alpha, 0, base)

    # border (single outline, no internal lines)
    if border > 0:
        cv2.ellipse(base, (x1+r, y1+r), (r, r), 180, 0, 90, border_color, border, cv2.LINE_AA)
        cv2.ellipse(base, (x2-r, y1+r), (r, r), 270, 0, 90, border_color, border, cv2.LINE_AA)
        cv2.ellipse(base, (x1+r, y2-r), (r, r), 90, 0, 90, border_color, border, cv2.LINE_AA)
        cv2.ellipse(base, (x2-r, y2-r), (r, r), 0, 0, 90, border_color, border, cv2.LINE_AA)
        cv2.line(base, (x1+r, y1), (x2-r, y1), border_color, border, cv2.LINE_AA)
        cv2.line(base, (x1+r, y2), (x2-r, y2), border_color, border, cv2.LINE_AA)
        cv2.line(base, (x1, y1+r), (x1, y2-r), border_color, border, cv2.LINE_AA)
        cv2.line(base, (x2, y1+r), (x2, y2-r), border_color, border, cv2.LINE_AA)

# ======= Clean, minimal Defense Panel (uses new rounded rect) =======
def draw_defense_panel(img_rgb, active_mode):
    h, w = img_rgb.shape[:2]
    pad = 16
    panel_w, panel_h = 280, 140
    x1, y1 = pad, h - panel_h - pad
    x2, y2 = x1 + panel_w, y1 + panel_h

    _rounded_rect_aa(img_rgb, (x1, y1), (x2, y2),
                     radius=16, fill=(20,20,20), alpha=0.55,
                     border=1, border_color=(220,220,220),
                     shadow=True, shadow_offset=(3,4), shadow_alpha=0.22)

    cv2.putText(img_rgb, "Defenses", (x1 + 16, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

    items = [
        ("0", "None",            0),
        ("1", "Strong Gaussian", 3),  
        ("2", "JPEG",            2),
        ("3", "LGS stripe",      1),
    ]
    y = y1 + 58
    for key, label, mode_id in items:
        cx, cy = x1 + 24, y - 7
        cv2.circle(img_rgb, (cx, cy), 8, (235,235,235), 2, cv2.LINE_AA)
        if active_mode == mode_id:
            cv2.circle(img_rgb, (cx, cy), 5, (235,235,235), -1, cv2.LINE_AA)
        cv2.putText(img_rgb, f"{key}  {label}", (x1 + 44, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
        y += 26

# =================== MODES =================
MODE_NONE         = 0
MODE_LGS_STRIPE   = 1
MODE_JPEG         = 2
MODE_GAUSS_STRONG = 3
mode = MODE_NONE

CONF_BY_MODE = {
    MODE_NONE:         0.65,
    MODE_LGS_STRIPE:   0.30,
    MODE_JPEG:         0.30,
    MODE_GAUSS_STRONG: 0.30,
}

win_name = "Defense Demo"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
try:
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except cv2.error:
    pass

print("Keys: 0=None | 1=Strong Gauss | 2=JPEG | 3=LGS stripe")

smoothed_conf = None  # for EMA

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        print("[warn] frame read failed; stopping.")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ---- DEFENSE ON FULL-RES (only in defense modes) ----
    if mode == MODE_JPEG:
        src_for_pre = apply_strong_jpeg_defense_rgb_u8(
            frame_rgb, quality=JPEG_QUALITY, down_scale=0.7, median_ksize=3
        )
    elif mode == MODE_GAUSS_STRONG:
        src_for_pre = apply_strong_gaussian_blur_rgb_u8(frame_rgb, ksize=15, sigma=3.5)
    else:
        src_for_pre = frame_rgb

    # Resize to model size
    img_t = to_tensor(resize640(Image.fromarray(src_for_pre))).to(device)
    np_640 = tensor_to_uint8(img_t)

    # Extra LGS stripe at 640 if selected
    if mode == MODE_LGS_STRIPE:
        defended_t = apply_lgs_stripe(img_t)
        defended_np = tensor_to_uint8(defended_t)
    else:
        defended_np = np_640

    # ===== Inference =====
    conf_this = CONF_BY_MODE.get(mode, CONF_BASE)

    if mode == MODE_NONE:
        use_tta = False
        iou_nms = 0.50
        agn_nms = False
    else:
        use_tta = True
        iou_nms = 0.65
        agn_nms = True

    res = model.predict(
        source=defended_np,
        imgsz=IMG_SIZE,
        conf=conf_this,
        iou=iou_nms,
        augment=use_tta,
        agnostic_nms=agn_nms,
        classes=[0],
        device=device,
        verbose=False
    )[0]

    # Optional rescue ONLY in defense modes
    need_rescue = (res.boxes is None) or (res.boxes.conf is None) or (len(res.boxes) == 0)
    if need_rescue and (mode != MODE_NONE):
        rescue = apply_strong_jpeg_defense_rgb_u8(frame_rgb, quality=min(15, JPEG_QUALITY),
                                                  down_scale=0.6, median_ksize=5)
        rescue = apply_bitdepth_rgb_u8(rescue, bits=5)
        rescue_t = to_tensor(resize640(Image.fromarray(rescue))).to(device)
        rescue_np = tensor_to_uint8(rescue_t)
        res2 = model.predict(
            source=rescue_np,
            imgsz=IMG_SIZE,
            conf=0.12,
            iou=0.70,
            augment=True,
            agnostic_nms=True,
            classes=[0],
            device=device,
            verbose=False
        )[0]
        if res2.boxes is not None and res2.boxes.conf is not None and len(res2.boxes) > 0:
            res = res2

    vis = res.plot()

    # Clean Defense Panel + Confidence %
    draw_defense_panel(vis, mode)

    try:
        boxes = res.boxes
        curr_conf = float(boxes.conf.mean().item()) if (boxes is not None and boxes.conf is not None and len(boxes) > 0) else 0.0
    except Exception:
        curr_conf = 0.0

    smoothed_conf = curr_conf if smoothed_conf is None \
        else (1.0 - SMOOTH_ALPHA) * smoothed_conf + SMOOTH_ALPHA * curr_conf
    draw_conf_percent(vis, smoothed_conf)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imshow(win_name, vis_bgr)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('0'):
        mode = MODE_NONE
        print("[mode] None")
    elif k == ord('1'):
        mode = MODE_GAUSS_STRONG   
        print("[mode] Strong Gaussian Blur")
    elif k == ord('2'):
        mode = MODE_JPEG
        print("[mode] JPEG")
    elif k == ord('3'):
        mode = MODE_LGS_STRIPE 
        print("[mode] LGS stripe")

cap.release()
cv2.destroyAllWindows()




