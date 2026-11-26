import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import mediapipe as mp

# ══════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = os.path.join('checkpoints', 'depth_anything_v2_metric_hypersim_vits.pth')
ENCODER = 'vits'
DEPTH_INPUT_SIZE = 518
KNOWN_DISTANCE_CM = 30.0

CAM_WIDTH, CAM_HEIGHT = 960, 720

# APPROXIMATE CAMERA INTRINSICS (Must be calibrated for scientific accuracy)
# Using standard approximations if true intrinsics are unknown.
FX = 800.0
FY = 800.0
CX = CAM_WIDTH / 2.0
CY = CAM_HEIGHT / 2.0

HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'

DEPTH_SAMPLE_RADIUS = 2  # 5x5 neighborhood

# MediaPipe Hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]

# States
S_READY_P1 = 'READY_P1'
S_WAIT_RELEASE_1 = 'WAIT_RELEASE_1'
S_READY_P2 = 'READY_P2'
S_WAIT_RELEASE_2 = 'WAIT_RELEASE_2'  # After P2 marked, before running inference
S_RESULTS = 'RESULTS'

# ══════════════════════════════════════════════════════════════════
# §1  PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════
def check_required_files():
    missing = []
    if not os.path.exists(CHECKPOINT_PATH):
        missing.append(f'Metric Depth Anything checkpoint: {CHECKPOINT_PATH}')
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        missing.append(f'Hand Landmarker model: {HAND_LANDMARKER_MODEL_PATH}')
    if missing:
        print('ERROR: Required file(s) missing — cannot start.\n')
        for m in missing:
            print(f'  - {m}')
        print()
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# §2  CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════════
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam (index 0).')
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)

# ══════════════════════════════════════════════════════════════════
# §3  MEDIAPIPE TASKS — HANDS
# ══════════════════════════════════════════════════════════════════
def init_hand_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def detect_pointing_finger(hand_landmarks, w, h):
    lm = hand_landmarks
    wrist = _landmark_px(lm[0], w, h)
    tips, mcps = [8, 12, 16, 20], [5, 9, 13, 17]
    up = []
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_px = _landmark_px(lm[tip_idx], w, h)
        mcp_px = _landmark_px(lm[mcp_idx], w, h)
        up.append(_dist(wrist, tip_px) > _dist(wrist, mcp_px) * 1.6)
    index_up, middle_up, ring_up, pinky_up = up
    is_pointing = index_up and not (middle_up or ring_up or pinky_up)
    return is_pointing, _landmark_px(lm[8], w, h)

def draw_hand_pointer(frame, hand_landmarks, w, h, tip_px, is_pointing):
    if hand_landmarks is None:
        return
    pts = [_landmark_px(lm, w, h) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)
    if tip_px is not None:
        color = (0, 255, 0) if is_pointing else (0, 0, 255)
        cv2.circle(frame, tip_px, 10, color, -1)
        cv2.circle(frame, tip_px, 18, color, 2)

# ══════════════════════════════════════════════════════════════════
# §4  DEPTH ANYTHING V2 METRIC DEPTH
# ══════════════════════════════════════════════════════════════════
def init_depth_model():
    from metric_depth.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384], 'max_depth': 20.0},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'max_depth': 20.0},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'max_depth': 20.0},
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(device).eval()
    print(f'[depth] Loaded Depth Anything V2 Metric ({ENCODER}) on device: {device}')
    return model, device

def run_depth_inference(model, frame_bgr, device):
    with torch.no_grad():
        return model.infer_image(frame_bgr, input_size=DEPTH_INPUT_SIZE)

def sample_depth_neighborhood(depth_map, px, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_map.shape[:2]
    x, y = px
    x_min, x_max = max(0, x - radius), min(w - 1, x + radius)
    y_min, y_max = max(0, y - radius), min(h - 1, y + radius)
    neighborhood = depth_map[y_min:y_max+1, x_min:x_max+1]
    
    center_val = float(depth_map[int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))])
    median_val = float(np.median(neighborhood))
    return center_val, median_val

# ══════════════════════════════════════════════════════════════════
# §5  3D RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════
def pixel_to_3d(x, y, z):
    X = (x - CX) * z / FX
    Y = (y - CY) * z / FY
    return X, Y, z

def calculate_3d_distance(pt1_3d, pt2_3d):
    X1, Y1, Z1 = pt1_3d
    X2, Y2, Z2 = pt2_3d
    return math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)

# ══════════════════════════════════════════════════════════════════
# §6  UI DRAWING
# ══════════════════════════════════════════════════════════════════
def draw_text(frame, text, pos, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# ══════════════════════════════════════════════════════════════════
# §7  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    check_required_files()

    print('Loading Depth Anything V2 Metric VITS...')
    depth_model, device = init_depth_model()
    
    cap = init_camera()
    hand_landmarker = init_hand_detector()
    
    state = S_READY_P1
    p1_px = None
    p2_px = None
    
    # Results
    pixel_dist = 0.0
    p1_depth_center = 0.0
    p1_depth_median = 0.0
    p2_depth_center = 0.0
    p2_depth_median = 0.0
    depth_diff = 0.0
    p1_3d = (0.0, 0.0, 0.0)
    p2_3d = (0.0, 0.0, 0.0)
    dist_3d_cm = 0.0
    abs_err_cm = 0.0
    pct_err = 0.0
    
    last_hand_ts = 0

    print('Camera started. Press Q to quit, R to reset.')

    while True:
        frame = read_frame(cap)
        if frame is None:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        last_hand_ts = int(time.time() * 1000)
        hand_result = hand_landmarker.detect_for_video(mp_image, last_hand_ts)
        
        is_pointing, tip_px = False, None
        
        # Process first visible hand
        if hand_result.hand_landmarks:
            hl = hand_result.hand_landmarks[0]
            is_pointing, tip_px = detect_pointing_finger(hl, w, h)
            draw_hand_pointer(frame, hl, w, h, tip_px, is_pointing)

        # Draw markers and lines if they exist
        if p1_px:
            cv2.circle(frame, p1_px, 8, (0, 255, 255), -1)
            draw_text(frame, 'P1', (p1_px[0] + 10, p1_px[1]), color=(0, 255, 255))
        if p2_px:
            cv2.circle(frame, p2_px, 8, (0, 255, 255), -1)
            draw_text(frame, 'P2', (p2_px[0] + 10, p2_px[1]), color=(0, 255, 255))
        
        if p1_px and p2_px:
            cv2.line(frame, p1_px, p2_px, (255, 0, 0), 2)
            mid_x = (p1_px[0] + p2_px[0]) // 2
            mid_y = (p1_px[1] + p2_px[1]) // 2
            draw_text(frame, f'{KNOWN_DISTANCE_CM} cm', (mid_x, mid_y - 10), color=(255, 0, 0), scale=0.6)

        # State Machine
        if state == S_READY_P1:
            draw_text(frame, "RIGHT HAND RULER TEST", (10, 30))
            draw_text(frame, "Point at Ruler Point 1", (10, 60))
            if is_pointing and tip_px:
                p1_px = tip_px
                state = S_WAIT_RELEASE_1
                print(f"P1 marked at {p1_px}")
                
        elif state == S_WAIT_RELEASE_1:
            draw_text(frame, "P1 MARKED", (10, 30), color=(0, 255, 0))
            draw_text(frame, "Close your hand to release pointing", (10, 60), color=(0, 255, 0))
            if not is_pointing:
                state = S_READY_P2
                print("Gesture released. Ready for P2.")
                
        elif state == S_READY_P2:
            draw_text(frame, "Point at Ruler Point 2", (10, 30))
            if is_pointing and tip_px:
                p2_px = tip_px
                state = S_WAIT_RELEASE_2
                print(f"P2 marked at {p2_px}")
                
        elif state == S_WAIT_RELEASE_2:
            draw_text(frame, "P2 MARKED. Calculating metric depth...", (10, 30), color=(0, 255, 0))
            cv2.imshow('Ruler Test', frame)
            cv2.waitKey(1)
            
            # Run inference EXACTLY ONCE
            depth_map = run_depth_inference(depth_model, frame, device)
            
            p1_depth_center, p1_depth_median = sample_depth_neighborhood(depth_map, p1_px)
            p2_depth_center, p2_depth_median = sample_depth_neighborhood(depth_map, p2_px)
            
            Z1 = p1_depth_median
            Z2 = p2_depth_median
            
            pixel_dist = _dist(p1_px, p2_px)
            depth_diff = abs(Z2 - Z1)
            
            p1_3d = pixel_to_3d(p1_px[0], p1_px[1], Z1)
            p2_3d = pixel_to_3d(p2_px[0], p2_px[1], Z2)
            
            dist_3d = calculate_3d_distance(p1_3d, p2_3d)
            dist_3d_cm = dist_3d * 100.0
            
            abs_err_cm = abs(dist_3d_cm - KNOWN_DISTANCE_CM)
            pct_err = (abs_err_cm / KNOWN_DISTANCE_CM) * 100.0
            
            # Print to console
            print("="*50)
            print(f"P1 = {p1_px}")
            print(f"P2 = {p2_px}")
            print(f"Pixel distance = {pixel_dist:.2f} px")
            print(f"P1 depth (center, median) = {p1_depth_center:.4f} m, {p1_depth_median:.4f} m")
            print(f"P2 depth (center, median) = {p2_depth_center:.4f} m, {p2_depth_median:.4f} m")
            print(f"Depth-axis difference = {depth_diff:.4f} m (Depth-axis difference — NOT full point-to-point distance)")
            print(f"P1 3D = {p1_3d}")
            print(f"P2 3D = {p2_3d}")
            print(f"Metric 3D distance = {dist_3d_cm:.2f} cm")
            print(f"Absolute error: {abs_err_cm:.2f} cm")
            print(f"Percentage error: {pct_err:.2f} %")
            print("="*50)
            
            state = S_RESULTS
            
        elif state == S_RESULTS:
            y = 30
            draw_text(frame, f"Known distance: {KNOWN_DISTANCE_CM:.2f} cm", (10, y)); y+=30
            draw_text(frame, f"Pixel distance: {pixel_dist:.2f} px", (10, y)); y+=30
            draw_text(frame, f"Metric 3D distance: {dist_3d_cm:.2f} cm", (10, y), color=(0,255,255)); y+=30
            draw_text(frame, f"Error: {abs_err_cm:.2f} cm ({pct_err:.2f} %)", (10, y), color=(0,0,255)); y+=30
            draw_text(frame, "Press R to Reset, Q to Quit.", (10, y+20), scale=0.5)

        cv2.imshow('Ruler Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            state = S_READY_P1
            p1_px = None
            p2_px = None
            print("Resetting experiment...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
