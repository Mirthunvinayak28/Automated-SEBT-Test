import cv2
import math
import os
import sys
import time
import csv
import numpy as np
import torch
import mediapipe as mp

# ══════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
import argparse
parser = argparse.ArgumentParser(description="SEBT Video Depth Analysis")
parser.add_argument("-i", "--input", default="input_videos/test6.mp4", help="Path to the input video")
parser.add_argument("--left-leg", type=float, default=None, help="Manual left leg length in cm")
parser.add_argument("--right-leg", type=float, default=None, help="Manual right leg length in cm")
parser.add_argument("--output-dir", default="results", help="Directory to save output files")
parser.add_argument("--no-display", "--headless", dest="no_display", action="store_true", help="Run in headless mode without cv2 display window")
args, _ = parser.parse_known_args()



VIDEO_PATH = args.input
OUTPUT_DIR = args.output_dir if args.output_dir else "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
CSV_PATH = os.path.join(OUTPUT_DIR, f"{base_name}_depth_results.csv")
CSV_SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"{base_name}_sebt_summary.csv")
OUT_VIDEO = os.path.join(OUTPUT_DIR, f"{base_name}_depth_visualized.mp4")

CHECKPOINT_PATH = os.path.join("checkpoints", "depth_anything_v2_metric_hypersim_vits.pth")
ENCODER = "vits"
DEPTH_INPUT_SIZE = 518

POSE_LANDMARKER_MODEL_PATH = "pose_landmarker.task"

# ── THRESHOLDS & SETTINGS ──
STABLE_FRAME_COUNT = 15          # Frames to average for initial baseline
START_MARGIN = 25              # Pixels above baseline to trigger movement start (legacy fallback)
START_DISP_MARGIN = 18.0         # Pixels of individual ankle displacement to trigger movement
CONSECUTIVE_FRAMES = 3           # Hysteresis for state changes
DEPTH_SAMPLE_RADIUS = 2          # 5x5 neighborhood

L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

# ── SEBT DIRECTIONS (8 Sectors, 45 deg each) ──
SEBT_DIRECTIONS = [
    ("ANT",   -22.5,   22.5,  "Anterior"),
    ("AM",     22.5,   67.5,  "Anteromedial"),
    ("M",      67.5,  112.5,  "Medial"),
    ("PM",    112.5,  157.5,  "Posteromedial"),
    ("POST",  157.5,  180.0,  "Posterior"),
    ("PL",   -157.5, -112.5,  "Posterolateral"),
    ("LAT",  -112.5,  -67.5,  "Lateral"),
    ("AL",    -67.5,  -22.5,  "Anterolateral"),
]
MIN_REACH_DISP_PX = 15.0       # Minimum reach displacement to assign direction
GROUND_FORESHORTENING_K = 2.2  # Camera ground-plane vertical foreshortening ratio
PEAK_WINDOW_FRAMES = 5         # Frames around maximum reach (peak ± 5 frames) for voting
DIRECTION_CONFIRM_RATIO = 0.70 # Minimum agreement ratio (70%) required to confirm direction

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (9, 10), (0, 11), (0, 12),
]

# States
S_WAITING = 'waiting'
S_MOVING = 'moving'

# ══════════════════════════════════════════════════════════════════
# §1  PRE-FLIGHT CHECKS & SETUP
# ══════════════════════════════════════════════════════════════════
def check_required_files():
    missing = []
    if not os.path.exists(VIDEO_PATH): missing.append(f'Video: {VIDEO_PATH}')
    if not os.path.exists(CHECKPOINT_PATH): missing.append(f'Checkpoint: {CHECKPOINT_PATH}')
    if not os.path.exists(POSE_LANDMARKER_MODEL_PATH): missing.append(f'Pose Model: {POSE_LANDMARKER_MODEL_PATH}')
    if missing:
        print('ERROR: Missing files:')
        for m in missing: print(f'  - {m}')
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_pose_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    base_options = mp_tasks.BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

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

def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def draw_pose_skeleton(frame, pose_landmarks, w, h):
    pts = [_landmark_px(lm, w, h) for lm in pose_landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 180), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

def draw_text(frame, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def pixel_to_3d(px, z, cx, cy, fx, fy):
    x, y = px
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return X, Y, z

def calculate_3d_distance(pt1, pt2):
    X1, Y1, Z1 = pt1
    X2, Y2, Z2 = pt2
    return math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)

def compute_body_coordinate_axes(pl, w, h):
    """
    Computes body transverse axis pointing toward participant's LEFT
    and forward axis pointing ANTERIORLY along ground plane.
    Uses shoulders (11, 12) and hips (23, 24).
    """
    l_sh = np.array([pl[L_SHOULDER].x * w, pl[L_SHOULDER].y * h])
    r_sh = np.array([pl[R_SHOULDER].x * w, pl[R_SHOULDER].y * h])
    l_hip = np.array([pl[L_HIP].x * w, pl[L_HIP].y * h])
    r_hip = np.array([pl[R_HIP].x * w, pl[R_HIP].y * h])
    
    v_trans = 0.5 * ((l_sh - r_sh) + (l_hip - r_hip))
    norm = np.linalg.norm(v_trans)
    if norm < 1e-4:
        u_body_left = np.array([1.0, 0.0])
    else:
        u_body_left = v_trans / norm
        
    u_body_fwd = np.array([-u_body_left[1], u_body_left[0]])
    if u_body_fwd[1] < 0:
        u_body_fwd = -u_body_fwd
        
    torso_angle_deg = math.degrees(math.atan2(u_body_left[1], u_body_left[0]))
    return u_body_left, u_body_fwd, torso_angle_deg

def classify_sebt_reach(dx, dy, stance_leg, u_body_left, u_body_fwd, k_foreshorten=GROUND_FORESHORTENING_K, min_disp=MIN_REACH_DISP_PX):
    """
    Projects reaching ankle displacement (dx, dy) onto participant's body-relative
    medial/lateral and anterior/posterior axes, accounting for stance leg.
    Returns: (direction_str, angle_deg, d_medial, d_anterior, confidence_status)
    """
    disp_magnitude = math.hypot(dx, dy)
    if disp_magnitude < min_disp:
        return "Unknown", 0.0, 0.0, 0.0, "Low (Sub-threshold)"
        
    disp_vec = np.array([dx, dy], dtype=float)
    d_trans = float(np.dot(disp_vec, u_body_left))
    d_ant_raw = float(np.dot(disp_vec, u_body_fwd))
    d_ant = d_ant_raw * k_foreshorten
    
    # Stance leg determines Medial (inward across midline) vs Lateral (outward away from midline)
    if stance_leg == 'LEFT':
        d_med = d_trans
    elif stance_leg == 'RIGHT':
        d_med = -d_trans
    else:
        d_med = d_trans
        
    angle_deg = math.degrees(math.atan2(d_med, d_ant))
    
    if angle_deg >= 157.5 or angle_deg <= -157.5:
        assigned_dir = "POST"
    else:
        assigned_dir = "Unknown"
        for name, low, high, _ in SEBT_DIRECTIONS:
            if low <= angle_deg < high:
                assigned_dir = name
                break
                
    conf = "High" if disp_magnitude >= 25.0 else "Medium"
    return assigned_dir, angle_deg, d_med, d_ant, conf

def determine_final_direction(reach_trajectory, max_frame_idx, dx_max, dy_max,
                               standing_ankle, u_body_left, u_body_fwd,
                               torso_ang, baseline_torso_angle,
                               window_size=PEAK_WINDOW_FRAMES, min_ratio=DIRECTION_CONFIRM_RATIO):
    """
    Analyzes a short window of frames around the peak reach (max_frame_idx)
    and determines the final SEBT direction using majority voting.
    Confirms the direction only if at least 70% of valid frames agree;
    otherwise marks it as 'Boundary / Low confidence' or 'Unknown'.
    """
    # Angle calculation from maximum reach displacement (preserved)
    _, peak_ang, _, _, base_conf = classify_sebt_reach(
        dx_max, dy_max, standing_ankle, u_body_left, u_body_fwd
    )
    
    # Extract short window of frames around the peak frame
    window_frames = [t for t in reach_trajectory if abs(t[0] - max_frame_idx) <= window_size]
    valid_dirs = [t[1] for t in window_frames if t[1] != "Unknown"]
    
    if len(valid_dirs) == 0:
        final_dir = "Unknown"
        final_conf = "Low"
        agreement_ratio = 0.0
    else:
        counts = {d: valid_dirs.count(d) for d in set(valid_dirs)}
        majority_dir = max(counts, key=counts.get)
        majority_count = counts[majority_dir]
        agreement_ratio = majority_count / len(valid_dirs)
        
        if agreement_ratio >= min_ratio:
            final_dir = majority_dir
            final_conf = "High"
        else:
            final_dir = "Boundary / Low confidence"
            final_conf = "Low"
            
    if abs(torso_ang - baseline_torso_angle) > 25.0:
        final_conf += " (Torso Rotated)"
        
    return final_dir, peak_ang, final_conf, agreement_ratio, len(valid_dirs)

def draw_sebt_overlay(frame, state, moving_ankle, stance_leg,
                      curr_reach_disp, max_reach_disp, curr_dist_ankles, max_pixel_dist,
                      provisional_dir, provisional_angle, last_confirmed_info, w, h):
    """
    Renders high-contrast SEBT direction HUD banner near the top of the video frame.
    Visibly rendered directly into the frames written to the final output MP4.
    """
    x1, y1 = 12, 12
    x2 = min(w - 12, 530)
    y2 = 152
    
    # Semi-transparent dark background
    sub_img = frame[y1:y2, x1:x2]
    black_rect = np.full_like(sub_img, 18)
    cv2.addWeighted(sub_img, 0.20, black_rect, 0.80, 0, sub_img)
    frame[y1:y2, x1:x2] = sub_img
    
    # Line 1: Primary SEBT Direction Badge
    if state == S_MOVING:
        border_color = (0, 220, 255)
        dir_text = f"SEBT: {provisional_dir} (Reaching)"
        dir_color = (0, 255, 255)
    elif last_confirmed_info:
        m_num = last_confirmed_info.get('movement', 1)
        m_dir = last_confirmed_info.get('direction', 'Unknown')
        if m_dir in ("Boundary / Low confidence", "Unknown"):
            border_color = (0, 165, 255)
            dir_text = f"SEBT: M{m_num} {m_dir}"
            dir_color = (0, 165, 255)
        else:
            border_color = (80, 200, 120)
            dir_text = f"SEBT: M{m_num} {m_dir} (Confirmed)"
            dir_color = (80, 255, 120)
    else:
        border_color = (80, 200, 120)
        dir_text = "SEBT: Waiting for Reach"
        dir_color = (220, 220, 220)

    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    font_scale = 0.60 if len(dir_text) > 28 else 0.72
    cv2.putText(frame, dir_text, (x1 + 12, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, font_scale, dir_color, 2, cv2.LINE_AA)
    
    # Line 2: Movement State & Legs
    state_str = "REACHING" if state == S_MOVING else "WAITING"
    m_ankle = moving_ankle if moving_ankle else "--"
    s_ankle = stance_leg if stance_leg else "--"
    info_text = f"State: {state_str} | Reach Ankle: {m_ankle} | Stance: {s_ankle}"
    cv2.putText(frame, info_text, (x1 + 12, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    
    # Line 3: Reach Displacement
    disp_text = f"Reach Disp: {curr_reach_disp:5.1f} px | Max: {max_reach_disp:5.1f} px"
    cv2.putText(frame, disp_text, (x1 + 12, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 240, 255), 1, cv2.LINE_AA)
    
    # Line 4: Angle or Last Result
    if state == S_MOVING:
        stat_text = f"Angle: {provisional_angle:+5.1f} deg | Ankle Sep: {curr_dist_ankles:5.1f} px"
    elif last_confirmed_info:
        m_disp_cm = last_confirmed_info.get('disp_cm', 0.0)
        m_conf = last_confirmed_info.get('conf', 'High')
        m_dir = last_confirmed_info.get('direction', 'Unknown')
        m_score = last_confirmed_info.get('norm_score_manual')
        prefix = "3D Reach:" if m_dir in ("Boundary / Low confidence", "Unknown") else "Confirmed 3D Reach:"
        if m_score is not None:
            stat_text = f"{prefix} {m_disp_cm:.1f} cm ({m_score:.1f}%) | Conf: {m_conf}"
        else:
            stat_text = f"{prefix} {m_disp_cm:.1f} cm | Conf: {m_conf}"
    else:
        stat_text = f"Ankle Separation: {curr_dist_ankles:5.1f} px"
    cv2.putText(frame, stat_text, (x1 + 12, y1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════
# §1.5  LEG LENGTH & SCORING HELPERS
# ══════════════════════════════════════════════════════════════════
def prompt_manual_leg_lengths(cli_left=None, cli_right=None):
    """
    Acquires and validates manual left and right leg lengths in centimetres.
    Validates that both values are numeric and positive (> 0).
    Returns (left_leg_length_cm, right_leg_length_cm).
    """
    def _validate(val):
        try:
            f = float(val)
            return f if f > 0 else None
        except (ValueError, TypeError):
            return None

    left = _validate(cli_left)
    right = _validate(cli_right)

    if left is None or right is None:
        print("\n" + "="*65)
        print("SEBT CLINICAL LEG LENGTH INPUT")
        print("="*65)
        print("Please enter measured leg lengths in centimetres (positive numbers).")
        print("Clinical protocol: Measured from anterior superior iliac spine (ASIS)")
        print("to the distal tip of the medial malleolus while supine.")
        print("="*65)

    while left is None:
        try:
            raw = input("Enter LEFT leg length (cm): ").strip()
            left = _validate(raw)
            if left is None:
                print("  [ERROR] Invalid input. Must be a positive numeric value (e.g. 88.5).")
        except EOFError:
            left = 85.0
            print(f"  [WARNING] Non-interactive stdin detected. Using fallback default LEFT leg: {left:.1f} cm")

    while right is None:
        try:
            raw = input("Enter RIGHT leg length (cm): ").strip()
            right = _validate(raw)
            if right is None:
                print("  [ERROR] Invalid input. Must be a positive numeric value (e.g. 88.5).")
        except EOFError:
            right = 85.0
            print(f"  [WARNING] Non-interactive stdin detected. Using fallback default RIGHT leg: {right:.1f} cm")

    print(f"\n[Leg Lengths Registered] Left: {left:.1f} cm | Right: {right:.1f} cm\n")
    return left, right

def compute_hip_to_ankle_3d_length(pl, depth_map, w, h, cx, cy, fx, fy, side='LEFT'):
    """
    Estimates 3D hip-to-ankle Euclidean length (cm) using MediaPipe landmarks and metric depth.
    Landmarks:
      - Left:  Hip (23), Ankle (27)
      - Right: Hip (24), Ankle (28)
    IMPORTANT: This is a video-estimated hip-to-ankle length, NOT a clinically measured
    ASIS-to-medial-malleolus leg length.
    """
    hip_idx = L_HIP if side == 'LEFT' else R_HIP
    ankle_idx = L_ANKLE if side == 'LEFT' else R_ANKLE

    hip_lm = pl[hip_idx]
    ankle_lm = pl[ankle_idx]

    hip_vis = getattr(hip_lm, 'visibility', 1.0)
    ankle_vis = getattr(ankle_lm, 'visibility', 1.0)
    if (hip_vis is not None and hip_vis < 0.4) or (ankle_vis is not None and ankle_vis < 0.4):
        return None

    hip_px = _landmark_px(hip_lm, w, h)
    ankle_px = _landmark_px(ankle_lm, w, h)

    if not (0 <= hip_px[0] < w and 0 <= hip_px[1] < h and 0 <= ankle_px[0] < w and 0 <= ankle_px[1] < h):
        return None

    _, hip_depth = sample_depth_neighborhood(depth_map, hip_px)
    _, ankle_depth = sample_depth_neighborhood(depth_map, ankle_px)

    if hip_depth <= 0.05 or ankle_depth <= 0.05:
        return None

    p_hip_3d = pixel_to_3d(hip_px, hip_depth, cx, cy, fx, fy)
    p_ankle_3d = pixel_to_3d(ankle_px, ankle_depth, cx, cy, fx, fy)

    length_cm = calculate_3d_distance(p_hip_3d, p_ankle_3d) * 100.0
    if 30.0 <= length_cm <= 160.0:
        return length_cm
    return None

def calibrate_video_leg_lengths(depth_model, device, resting_frames, w, h, cx, cy, fx, fy,
                                manual_left_cm, manual_right_cm):
    """
    Estimates left and right hip-to-ankle lengths across multiple resting frames
    using MediaPipe Pose landmarks (hips 23/24, ankles 27/28) and Metric Depth.
    Aggregates measurements using the median.
    Compares video-estimated lengths against manual clinical measurements.
    """
    left_lengths = []
    right_lengths = []

    sample_frames = resting_frames
    if len(resting_frames) > 5:
        indices = np.linspace(0, len(resting_frames) - 1, 5, dtype=int)
        sample_frames = [resting_frames[i] for i in indices]

    for f_img, pl in sample_frames:
        d_map = run_depth_inference(depth_model, f_img, device)
        l_len = compute_hip_to_ankle_3d_length(pl, d_map, w, h, cx, cy, fx, fy, 'LEFT')
        r_len = compute_hip_to_ankle_3d_length(pl, d_map, w, h, cx, cy, fx, fy, 'RIGHT')
        if l_len is not None:
            left_lengths.append(l_len)
        if r_len is not None:
            right_lengths.append(r_len)

    est_left = float(np.median(left_lengths)) if left_lengths else None
    est_right = float(np.median(right_lengths)) if right_lengths else None

    l_err_cm = abs(est_left - manual_left_cm) if est_left is not None else None
    l_err_pct = (l_err_cm / manual_left_cm * 100.0) if (l_err_cm is not None and manual_left_cm > 0) else None
    r_err_cm = abs(est_right - manual_right_cm) if est_right is not None else None
    r_err_pct = (r_err_cm / manual_right_cm * 100.0) if (r_err_cm is not None and manual_right_cm > 0) else None

    print("\n" + "="*70)
    print("VIDEO-ESTIMATED HIP-TO-ANKLE LENGTH CALIBRATION")
    print("NOTE: This is video-estimated hip-to-ankle length, NOT clinical ASIS-to-medial-malleolus length.")
    print("="*70)
    print(f"Sampled resting frames for calibration: {len(sample_frames)} (valid L: {len(left_lengths)}, valid R: {len(right_lengths)})")
    if est_left is not None:
        print(f"LEFT Leg  -> Manual: {manual_left_cm:5.1f} cm | Video-Est: {est_left:5.1f} cm | Abs Error: {l_err_cm:5.1f} cm ({l_err_pct:5.1f}%)")
    else:
        print(f"LEFT Leg  -> Manual: {manual_left_cm:5.1f} cm | Video-Est: UNAVAILABLE")
    if est_right is not None:
        print(f"RIGHT Leg -> Manual: {manual_right_cm:5.1f} cm | Video-Est: {est_right:5.1f} cm | Abs Error: {r_err_cm:5.1f} cm ({r_err_pct:5.1f}%)")
    else:
        print(f"RIGHT Leg -> Manual: {manual_right_cm:5.1f} cm | Video-Est: UNAVAILABLE")
    print("="*70 + "\n")

    return est_left, est_right, l_err_cm, l_err_pct, r_err_cm, r_err_pct

def calculate_composite_scores(movement_records):
    """
    Computes SEBT composite scores from movement records.

    Documented Inclusion Rules:
    1. Only valid movements (scoring_validity == 'VALID') with confirmed SEBT directions
       (ANT, AM, M, PM, POST, PL, LAT, AL) and high/medium confidence are included.
    2. Repeated trials in the same direction: In standard clinical SEBT protocol,
       when multiple trials are performed in the same direction, the trial with the
       maximum valid reach distance (and highest normalized score) is selected for that direction.
    3. Composite Score Formula:
       composite_score = sum(normalized_reach_scores_of_included_directions) / actual_count_of_included_directions
       (The denominator is the actual number of distinct valid directions included in the sum, NOT automatically 8).
    4. Completeness:
       If actual_count == 8: labeled 'Complete 8-direction composite (8/8 directions)'
       If 1 <= actual_count < 8: labeled 'Partial composite ({actual_count}/8 directions available)'
       If actual_count == 0: labeled 'Unavailable (0 valid directions)'
    5. Manual and Video-estimated scores are calculated completely separately.
    """
    groups = {'LEFT': [], 'RIGHT': []}
    for rec in movement_records:
        s_leg = rec.get('stance_leg')
        if s_leg in groups:
            groups[s_leg].append(rec)

    composite_results = {}
    ALL_SEBT_DIRECTIONS = ["ANT", "AM", "M", "PM", "POST", "PL", "LAT", "AL"]

    for group_name, records in [('LEFT_STANCE', groups['LEFT']),
                                ('RIGHT_STANCE', groups['RIGHT']),
                                ('COMBINED_ALL_STANCE', movement_records)]:
        valid_recs = [r for r in records if r.get('scoring_validity') == 'VALID']
        excluded_recs = [r for r in records if r.get('scoring_validity') != 'VALID']

        best_trials_by_dir = {}
        for r in valid_recs:
            d = r.get('detected_direction')
            dist = r.get('valid_reach_dist_cm', 0.0)
            if dist is None:
                continue
            if d not in best_trials_by_dir or dist > best_trials_by_dir[d].get('valid_reach_dist_cm', 0.0):
                best_trials_by_dir[d] = r

        # Maintain standard SEBT anatomical ordering for distinct directions
        distinct_dirs = [d for d in ALL_SEBT_DIRECTIONS if d in best_trials_by_dir]
        for d in sorted(best_trials_by_dir.keys()):
            if d not in distinct_dirs:
                distinct_dirs.append(d)

        missing_dirs = [d for d in ALL_SEBT_DIRECTIONS if d not in best_trials_by_dir]
        num_distinct_dirs = len(distinct_dirs)
        num_valid_trials = len(valid_recs)

        manual_scores = [best_trials_by_dir[d]['norm_score_manual_pct'] for d in distinct_dirs
                         if best_trials_by_dir[d].get('norm_score_manual_pct') is not None]
        if len(manual_scores) > 0:
            manual_composite_pct = float(np.sum(manual_scores) / len(manual_scores))
        else:
            manual_composite_pct = None

        video_scores = [best_trials_by_dir[d]['norm_score_video_pct'] for d in distinct_dirs
                         if best_trials_by_dir[d].get('norm_score_video_pct') is not None]
        if len(video_scores) > 0:
            video_composite_pct = float(np.sum(video_scores) / len(video_scores))
        else:
            video_composite_pct = None

        if num_distinct_dirs == 8:
            status_label = "Complete 8-direction composite (8/8 directions)"
        elif num_distinct_dirs > 0:
            status_label = f"Partial composite ({num_distinct_dirs}/8 directions available)"
        else:
            status_label = "Unavailable (0 valid directions)"

        composite_results[group_name] = {
            'num_valid_trials': num_valid_trials,
            'num_distinct_dirs': num_distinct_dirs,
            'distinct_dirs': distinct_dirs,
            'missing_dirs': missing_dirs,
            'best_trials': best_trials_by_dir,
            'excluded_records': excluded_recs,
            'manual_composite_pct': manual_composite_pct,
            'video_composite_pct': video_composite_pct,
            'status_label': status_label,
            'actual_denominator': num_distinct_dirs
        }

    return composite_results

def write_sebt_summary_csv(summary_path, manual_l, manual_r, est_l, est_r,
                           l_err_cm, l_err_pct, r_err_cm, r_err_pct,
                           movement_records, composite_results):
    """
    Writes a structured, comprehensive SEBT clinical summary CSV.
    """
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["=== SECTION 1: LEG LENGTH MEASUREMENTS & VALIDATION ==="])
        writer.writerow(["Leg", "Manual Length (cm)", "Video-Estimated Length (cm)", "Abs Error (cm)", "Pct Error (%)", "Note"])
        writer.writerow([
            "LEFT",
            f"{manual_l:.1f}" if manual_l else "N/A",
            f"{est_l:.1f}" if est_l else "N/A",
            f"{l_err_cm:.2f}" if l_err_cm is not None else "N/A",
            f"{l_err_pct:.2f}%" if l_err_pct is not None else "N/A",
            "Video-estimated hip-to-ankle length (not clinical ASIS-to-medial-malleolus)"
        ])
        writer.writerow([
            "RIGHT",
            f"{manual_r:.1f}" if manual_r else "N/A",
            f"{est_r:.1f}" if est_r else "N/A",
            f"{r_err_cm:.2f}" if r_err_cm is not None else "N/A",
            f"{r_err_pct:.2f}%" if r_err_pct is not None else "N/A",
            "Video-estimated hip-to-ankle length (not clinical ASIS-to-medial-malleolus)"
        ])
        writer.writerow([])

        writer.writerow(["=== SECTION 2: MOVEMENT-BY-MOVEMENT REACH & NORMALIZED SCORES ==="])
        writer.writerow([
            "Movement", "Reaching Leg", "Stance Leg", "Detected Direction", "Direction Confidence",
            "Directional Reach Dist (cm)", "Stance Manual Leg Length (cm)", "Stance Video Leg Length (cm)",
            "Normalized Score (Manual) (%)", "Normalized Score (Video) (%)", "Scoring Validity", "Exclusion Reason"
        ])
        for r in movement_records:
            writer.writerow([
                r.get('movement'),
                r.get('moving_ankle'),
                r.get('stance_leg'),
                r.get('detected_direction'),
                r.get('direction_confidence'),
                f"{r.get('directional_disp_cm'):.2f}",
                f"{r.get('manual_stance_len'):.1f}" if r.get('manual_stance_len') else "N/A",
                f"{r.get('est_stance_len'):.1f}" if r.get('est_stance_len') else "N/A",
                f"{r.get('norm_score_manual_pct'):.2f}%" if r.get('norm_score_manual_pct') is not None else "N/A",
                f"{r.get('norm_score_video_pct'):.2f}%" if r.get('norm_score_video_pct') is not None else "N/A",
                r.get('scoring_validity'),
                r.get('scoring_exclusion_reason')
            ])
        writer.writerow([])

        writer.writerow(["=== SECTION 3: COMPOSITE SCORES BY STANCE LEG ==="])
        writer.writerow(["Stance Group", "Status Label", "Distinct Directions Count (Denominator)", "Included Directions",
                         "Valid Movements Count", "Manual Composite Score (%)", "Video Composite Score (%)"])
        for grp_key in ['LEFT_STANCE', 'RIGHT_STANCE', 'COMBINED_ALL_STANCE']:
            res = composite_results.get(grp_key, {})
            m_comp = f"{res.get('manual_composite_pct'):.2f}%" if res.get('manual_composite_pct') is not None else "UNAVAILABLE"
            v_comp = f"{res.get('video_composite_pct'):.2f}%" if res.get('video_composite_pct') is not None else "UNAVAILABLE"
            dirs_str = "; ".join(res.get('distinct_dirs', [])) if res.get('distinct_dirs') else "None"
            writer.writerow([
                grp_key,
                res.get('status_label'),
                res.get('actual_denominator'),
                dirs_str,
                res.get('num_valid_trials'),
                m_comp,
                v_comp
            ])
        writer.writerow([])

        writer.writerow(["=== SECTION 4: METHODOLOGICAL LIMITATIONS DISCLOSURE ==="])
        writer.writerow(["Limitation Item", "Description"])
        writer.writerow(["1. Ankle vs Toe Tracking", "MediaPipe tracks lateral ankle (landmark 27/28), whereas clinical SEBT measures reach to the furthest point reached by the toe touching the floor grid without bearing weight."])
        writer.writerow(["2. Starting Baseline Reference", "Directional displacement is measured from the moving ankle's own starting resting position, not from the center of the grid or stance foot origin."])
        writer.writerow(["3. Initial Depth Borrowing", "Starting 3D ankle position uses stance ankle metric depth at peak reach rather than sampling depth at the true movement start frame."])
        writer.writerow(["4. 3D Vector vs 1D Axis", "Displacement is computed as 3D Euclidean distance (including vertical lift and out-of-plane variation) rather than 1D scalar distance projected along the floor grid direction line."])

# ══════════════════════════════════════════════════════════════════
# §2  MAIN PROCESS LOOP
# ══════════════════════════════════════════════════════════════════
def main():
    l_pos_smooth = None
    r_pos_smooth = None
    curr_pos_red = None
    curr_pos_blue = None
    check_required_files()
    
    # §1.1 Manual Leg-Length Input
    manual_left_leg_length_cm, manual_right_leg_length_cm = prompt_manual_leg_lengths(
        args.left_leg, args.right_leg
    )

    # Init Models
    pose_landmarker = init_pose_detector()
    depth_model, device = init_depth_model()
    
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or math.isnan(fps): fps = 30.0
        
        FX = 800.0
        FY = 800.0
        CX = w / 2.0
        CY = h / 2.0
        print("\nAPPROXIMATE CAMERA INTRINSICS")
        print(f"fx = {FX}, fy = {FY}, cx = {CX}, cy = {CY}\n")
        
        video_writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # CSV Init: Preserve 27 original columns, append extended SEBT scoring columns
        csv_f = open(CSV_PATH, 'w', newline='')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow([
            "movement", "moving_ankle", "start_frame", "max_frame", "start_time_sec", "max_time_sec",
            "initial_x", "initial_y", "max_x", "max_y", "pixel_distance_px",
            "initial_depth_m", "max_depth_m", "depth_difference_m",
            "initial_X_m", "initial_Y_m", "initial_Z_m", "max_X_m", "max_Y_m", "max_Z_m", "distance_3d_cm",
            "stance_leg", "detected_direction", "direction_confidence", "direction_confirmed_frame", "direction_confirmed_time", "directional_displacement_cm",
            "manual_stance_leg_length_cm", "video_estimated_stance_leg_length_cm",
            "manual_leg_length_abs_error_cm", "manual_leg_length_pct_error",
            "valid_directional_reach_distance_cm",
            "normalized_score_manual_pct", "normalized_score_video_pct",
            "scoring_validity", "scoring_exclusion_reason"
        ])
        
        # State Variables
        state = S_WAITING
        frame_idx = 0
        last_pose_ts = 0
        
        # Leg Length & Calibration State
        video_leg_calibrated = False
        est_left_leg_cm = None
        est_right_leg_cm = None
        left_error_cm = None
        left_error_pct = None
        right_error_cm = None
        right_error_pct = None
        calib_resting_frames = []
        movement_records = []
        
        l_ankle_hist = []
        r_ankle_hist = []
        l_pos_hist = []
        r_pos_hist = []
        l_z_hist = []
        r_z_hist = []
        dist_hist = []
        
        u_body_left = np.array([1.0, 0.0])
        u_body_fwd = np.array([0.0, 1.0])
        baseline_torso_angle = 0.0
        
        active_baseline = 0.0
        active_base_l = None
        active_base_r = None
        consec_start = 0
        consec_return = 0
        curr_movement_num = 1
        
        moving_ankle = None
        standing_ankle = None
        
        start_frame_idx = 0
        start_timestamp = 0.0
        start_dist = 0.0
        
        max_pixel_dist = 0.0
        max_reach_disp = 0.0
        max_pos_red = None
        max_pos_blue = None
        max_frame_img = None
        max_frame_idx = 0
        max_timestamp = 0.0
        
        init_pos_red = None
        init_pos_blue = None
        
        provisional_dir = "Unknown"
        provisional_angle = 0.0
        reach_trajectory = []
        last_confirmed_info = None
        results_summary = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        
        if not getattr(args, 'no_display', False):
            cv2.namedWindow("Offline Video Analysis", cv2.WINDOW_NORMAL)
        print(f"Processing Video ({total_frames} total frames)...", flush=True)
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            timestamp = frame_idx / fps
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            last_pose_ts = int(timestamp * 1000)
            
            pose_result = pose_landmarker.detect_for_video(mp_image, last_pose_ts)
            
            l_pos = None
            r_pos = None
            l_z = 0.0
            r_z = 0.0
            
            if pose_result.pose_landmarks:
                pl = pose_result.pose_landmarks[0]
                draw_pose_skeleton(frame, pl, w, h)
                
                # Estimate participant's body axes from torso and pelvis
                u_body_left_curr, u_body_fwd_curr, torso_ang = compute_body_coordinate_axes(pl, w, h)
                
                l_z = pl[L_ANKLE].z
                r_z = pl[R_ANKLE].z
                
                l_px = _landmark_px(pl[L_ANKLE], w, h)
                r_px = _landmark_px(pl[R_ANKLE], w, h)
                
                if 0 <= l_px[0] < w and 0 <= l_px[1] < h:
                    if l_pos_smooth is None:
                        l_pos_smooth = list(l_px)
                    else:
                        l_pos_smooth = [0.6 * l_pos_smooth[0] + 0.4 * l_px[0], 0.6 * l_pos_smooth[1] + 0.4 * l_px[1]]
                    l_pos = (int(l_pos_smooth[0]), int(l_pos_smooth[1]))
                else:
                    l_pos = None
                    
                if 0 <= r_px[0] < w and 0 <= r_px[1] < h:
                    if r_pos_smooth is None:
                        r_pos_smooth = list(r_px)
                    else:
                        r_pos_smooth = [0.6 * r_pos_smooth[0] + 0.4 * r_px[0], 0.6 * r_pos_smooth[1] + 0.4 * r_px[1]]
                    r_pos = (int(r_pos_smooth[0]), int(r_pos_smooth[1]))
                else:
                    r_pos = None
                
                # Highlight ankles
                if l_pos:
                    cv2.circle(frame, l_pos, 8, (255, 0, 0), -1) # Blue Left
                    draw_text(frame, 'L', (l_pos[0]+10, l_pos[1]), (255, 0, 0))
                if r_pos:
                    cv2.circle(frame, r_pos, 8, (0, 0, 255), -1) # Red Right
                    draw_text(frame, 'R', (r_pos[0]+10, r_pos[1]), (0, 0, 255))
            
            curr_reach_disp = 0.0
            curr_dist_ankles = 0.0
            
            # Main State Machine
            if l_pos and r_pos:
                curr_dist_ankles = _dist(l_pos, r_pos)
                
                if state == S_WAITING:
                    dist_hist.append(curr_dist_ankles)
                    l_pos_hist.append(l_pos)
                    r_pos_hist.append(r_pos)
                    l_z_hist.append(l_z)
                    r_z_hist.append(r_z)
                    l_ankle_hist.append(l_pos)
                    r_ankle_hist.append(r_pos)
                    
                    # Update baseline body axes while resting
                    u_body_left = u_body_left_curr
                    u_body_fwd = u_body_fwd_curr
                    baseline_torso_angle = torso_ang
                    
                    # Accumulate resting frames for video leg calibration
                    if not video_leg_calibrated and pl is not None and len(calib_resting_frames) < 10:
                        calib_resting_frames.append((frame.copy(), pl))
                    
                    if len(dist_hist) > 15:
                        dist_hist.pop(0)
                    if len(l_pos_hist) > 15:
                        l_pos_hist.pop(0)
                        r_pos_hist.pop(0)
                        l_z_hist.pop(0)
                        r_z_hist.pop(0)
                    if len(l_ankle_hist) > 10:
                        l_ankle_hist.pop(0)
                        r_ankle_hist.pop(0)
                        
                    if len(l_pos_hist) >= 8:
                        base_l = (float(np.mean([p[0] for p in l_pos_hist[:-4]])), float(np.mean([p[1] for p in l_pos_hist[:-4]])))
                        base_r = (float(np.mean([p[0] for p in r_pos_hist[:-4]])), float(np.mean([p[1] for p in r_pos_hist[:-4]])))
                        base_lz = float(np.mean(l_z_hist[:-4]))
                        base_rz = float(np.mean(r_z_hist[:-4]))
                        baseline_dist = float(np.mean(dist_hist[:-4])) if len(dist_hist) >= 8 else float(np.mean(dist_hist))
                    else:
                        base_l = (float(np.mean([p[0] for p in l_pos_hist])), float(np.mean([p[1] for p in l_pos_hist])))
                        base_r = (float(np.mean([p[0] for p in r_pos_hist])), float(np.mean([p[1] for p in r_pos_hist])))
                        base_lz = float(np.mean(l_z_hist))
                        base_rz = float(np.mean(r_z_hist))
                        baseline_dist = float(np.mean(dist_hist)) if len(dist_hist) > 0 else curr_dist_ankles

                    # Track individual ankle displacements from stable baseline
                    l_disp = _dist(l_pos, base_l)
                    r_disp = _dist(r_pos, base_r)
                    l_dz = abs(l_z - base_lz)
                    r_dz = abs(r_z - base_rz)
                    
                    # Movement trigger: individual ankle shows sustained displacement.
                    l_triggered = (l_disp > START_DISP_MARGIN) or (l_dz > 0.08 and l_disp > 12.0)
                    r_triggered = (r_disp > START_DISP_MARGIN) or (r_dz > 0.08 and r_disp > 12.0)
                    
                    if l_triggered or r_triggered:
                        consec_start += 1
                        if consec_start >= CONSECUTIVE_FRAMES and len(l_pos_hist) >= 5:
                            if l_disp > r_disp:
                                moving_ankle = 'LEFT'
                                standing_ankle = 'RIGHT'
                            else:
                                moving_ankle = 'RIGHT'
                                standing_ankle = 'LEFT'
                                
                            state = S_MOVING
                            start_frame_idx = frame_idx
                            start_timestamp = timestamp
                            start_dist = curr_dist_ankles
                            active_baseline = baseline_dist
                            active_base_l = base_l
                            active_base_r = base_r
                            
                            # Calibrate video leg lengths on first movement onset
                            if not video_leg_calibrated:
                                frames_to_use = calib_resting_frames if len(calib_resting_frames) > 0 else [(frame.copy(), pl)]
                                est_left_leg_cm, est_right_leg_cm, left_error_cm, left_error_pct, right_error_cm, right_error_pct = calibrate_video_leg_lengths(
                                    depth_model, device, frames_to_use, w, h, CX, CY, FX, FY,
                                    manual_left_leg_length_cm, manual_right_leg_length_cm
                                )
                                video_leg_calibrated = True
                            
                            max_pixel_dist = curr_dist_ankles
                            max_pos_red = l_pos if moving_ankle == 'LEFT' else r_pos
                            max_pos_blue = r_pos if moving_ankle == 'LEFT' else l_pos
                            moving_base = active_base_l if moving_ankle == 'LEFT' else active_base_r
                            max_reach_disp = _dist(max_pos_red, moving_base)
                            
                            max_frame_img = frame.copy()
                            max_frame_idx = frame_idx
                            max_timestamp = timestamp
                            
                            init_pos_red = max_pos_red
                            init_pos_blue = max_pos_blue
                            
                            reach_trajectory = []
                            provisional_dir = "Unknown"
                            provisional_angle = 0.0
                            consec_return = 0
                            print(f"MOVEMENT START at frame {frame_idx} / {timestamp:.2f}s | Reaching: {moving_ankle} | Stance: {standing_ankle}")
                    else:
                        consec_start = 0
                        
                elif state == S_MOVING:
                    curr_pos_red = l_pos if moving_ankle == 'LEFT' else r_pos
                    curr_pos_blue = r_pos if moving_ankle == 'LEFT' else l_pos
                    moving_base = active_base_l if moving_ankle == 'LEFT' else active_base_r
                    curr_reach_disp = _dist(curr_pos_red, moving_base)
                    
                    # Real-time provisional direction classification
                    dx_curr = curr_pos_red[0] - moving_base[0]
                    dy_curr = curr_pos_red[1] - moving_base[1]
                    prov_dir, prov_ang, _, _, _ = classify_sebt_reach(
                        dx_curr, dy_curr, standing_ankle, u_body_left, u_body_fwd
                    )
                    if prov_dir != "Unknown":
                        provisional_dir = prov_dir
                        provisional_angle = prov_ang
                    reach_trajectory.append((frame_idx, prov_dir, prov_ang, curr_reach_disp))
                    
                    # Track maximum reach displacement and maximum pixel distance
                    if curr_reach_disp > max_reach_disp or curr_dist_ankles > max_pixel_dist:
                        if curr_reach_disp > max_reach_disp:
                            max_reach_disp = curr_reach_disp
                        if curr_dist_ankles > max_pixel_dist:
                            max_pixel_dist = curr_dist_ankles
                        max_pos_red = curr_pos_red
                        max_pos_blue = curr_pos_blue
                        max_frame_img = frame.copy()
                        max_frame_idx = frame_idx
                        max_timestamp = timestamp
                    
                    # Visualization of reach vector and ankle markers
                    cv2.line(frame, curr_pos_blue, curr_pos_red, (0, 255, 0), 2)
                    cv2.arrowedLine(frame, (int(moving_base[0]), int(moving_base[1])), curr_pos_red, (0, 220, 255), 2, tipLength=0.2)
                    cv2.circle(frame, max_pos_red, 5, (0, 0, 255), -1)
                    cv2.circle(frame, max_pos_blue, 5, (255, 0, 0), -1)
                    draw_text(frame, f"Reach: {provisional_dir}", (curr_pos_red[0]+12, curr_pos_red[1]-8), (0, 255, 255), scale=0.55)
                    
                    # Adaptive return threshold relative to moving foot's starting position
                    return_threshold = max(12.0, max_reach_disp * 0.25)
                    
                    if curr_reach_disp < return_threshold:
                        consec_return += 1
                        if consec_return >= CONSECUTIVE_FRAMES:
                            print(f"MOVEMENT COMPLETE at frame {frame_idx} / {timestamp:.2f}s, dist={curr_dist_ankles:.1f}")
                            
                            draw_text(frame, "PROCESSING DEPTH...", (10, 180), (0, 255, 255))
                            
                            # Run Metric Depth on exactly the max frame
                            depth_map = run_depth_inference(depth_model, max_frame_img, device)
                            max_center_red, max_depth_red = sample_depth_neighborhood(depth_map, max_pos_red)
                            max_center_blue, max_depth_blue = sample_depth_neighborhood(depth_map, max_pos_blue)
                            
                            depth_diff = abs(max_depth_red - max_depth_blue)
                            
                            p_max_red_3d = pixel_to_3d(max_pos_red, max_depth_red, CX, CY, FX, FY)
                            p_max_blue_3d = pixel_to_3d(max_pos_blue, max_depth_blue, CX, CY, FX, FY)
                            
                            # Preserved experimental ankle-to-ankle 3D distance
                            dist_3d = calculate_3d_distance(p_max_red_3d, p_max_blue_3d) * 100.0 # cm
                            
                            # Directional 3D displacement of reaching foot from its starting baseline
                            p_init_red_3d = pixel_to_3d(init_pos_red, max_depth_blue, CX, CY, FX, FY)
                            directional_disp_3d = calculate_3d_distance(p_max_red_3d, p_init_red_3d) * 100.0 # cm
                            
                            # Final direction classification via majority voting on peak reach window
                            dx_max = max_pos_red[0] - moving_base[0]
                            dy_max = max_pos_red[1] - moving_base[1]
                            final_dir, final_ang, final_conf, agree_ratio, valid_cnt = determine_final_direction(
                                reach_trajectory, max_frame_idx, dx_max, dy_max,
                                standing_ankle, u_body_left, u_body_fwd,
                                torso_ang, baseline_torso_angle
                            )
                                
                            confirmed_frame = max_frame_idx
                            confirmed_time = max_timestamp
                            
                            # Stance leg selection for normalized scoring
                            # Stance leg determines denominator; reaching leg determines reach movement
                            if standing_ankle == 'LEFT':
                                manual_stance_len = manual_left_leg_length_cm
                                est_stance_len = est_left_leg_cm
                                stance_err_cm = left_error_cm
                                stance_err_pct = left_error_pct
                            elif standing_ankle == 'RIGHT':
                                manual_stance_len = manual_right_leg_length_cm
                                est_stance_len = est_right_leg_cm
                                stance_err_cm = right_error_cm
                                stance_err_pct = right_error_pct
                            else:
                                manual_stance_len = None
                                est_stance_len = None
                                stance_err_cm = None
                                stance_err_pct = None

                            VALID_SEBT_DIR_NAMES = {"ANT", "AM", "M", "PM", "POST", "PL", "LAT", "AL"}
                            is_valid_score = True
                            exclusion_reason = "VALID"

                            if standing_ankle not in ('LEFT', 'RIGHT'):
                                is_valid_score = False
                                exclusion_reason = "EXCLUDED_INVALID_STANCE_LEG"
                            elif final_dir not in VALID_SEBT_DIR_NAMES:
                                is_valid_score = False
                                if "Boundary" in final_dir:
                                    exclusion_reason = "EXCLUDED_BOUNDARY_DIRECTION"
                                elif final_dir == "Unknown":
                                    exclusion_reason = "EXCLUDED_UNKNOWN_DIRECTION"
                                else:
                                    exclusion_reason = f"EXCLUDED_UNCONFIRMED_DIRECTION_{final_dir}"
                            elif final_conf.startswith("Low"):
                                is_valid_score = False
                                exclusion_reason = "EXCLUDED_LOW_CONFIDENCE"
                            elif directional_disp_3d <= 0.0:
                                is_valid_score = False
                                exclusion_reason = "EXCLUDED_ZERO_OR_NEGATIVE_REACH"
                            elif manual_stance_len is None or manual_stance_len <= 0:
                                is_valid_score = False
                                exclusion_reason = "EXCLUDED_MISSING_MANUAL_LEG_LENGTH"

                            if is_valid_score:
                                valid_reach_dist_cm = directional_disp_3d
                                norm_score_manual_pct = (valid_reach_dist_cm / manual_stance_len) * 100.0
                                if est_stance_len and est_stance_len > 0:
                                    norm_score_video_pct = (valid_reach_dist_cm / est_stance_len) * 100.0
                                else:
                                    norm_score_video_pct = None
                                scoring_validity = "VALID"
                            else:
                                valid_reach_dist_cm = None
                                norm_score_manual_pct = None
                                norm_score_video_pct = None
                                scoring_validity = "EXCLUDED"
                                
                            last_confirmed_info = {
                                'movement': curr_movement_num,
                                'direction': final_dir,
                                'angle': final_ang,
                                'conf': final_conf,
                                'disp_cm': directional_disp_3d,
                                'ankle': moving_ankle,
                                'stance': standing_ankle,
                                'norm_score_manual': norm_score_manual_pct
                            }
                            
                            duration = timestamp - start_timestamp
                            
                            print("\n" + "="*50)
                            print(f"MOVEMENT {curr_movement_num} SUMMARY:")
                            print("="*50)
                            print(f"Reaching ankle: {moving_ankle} | Stance leg: {standing_ankle}")
                            print(f"Detected Direction: {final_dir} (Angle: {final_ang:+.1f} deg, Conf: {final_conf})")
                            print(f"Start frame/time: {start_frame_idx} / {start_timestamp:.2f} sec")
                            print(f"Maximum frame/time: {max_frame_idx} / {max_timestamp:.2f} sec")
                            print(f"Return frame/time: {frame_idx} / {timestamp:.2f} sec")
                            print(f"Movement duration: {duration:.2f} sec")
                            print(f"Initial reaching position: {init_pos_red}")
                            print(f"Maximum reaching position: {max_pos_red}")
                            print(f"MAX reach displacement: {max_reach_disp:.2f} px")
                            print(f"Maximum depth (Red, Blue): {max_depth_red:.4f} m, {max_depth_blue:.4f} m")
                            print(f"Directional 3D displacement: {directional_disp_3d:.2f} cm")
                            print(f"Ankle-to-ankle 3D distance: {dist_3d:.2f} cm")
                            if is_valid_score:
                                print(f"Stance Leg Length: Manual={manual_stance_len:.1f} cm | Video-Est=" + (f"{est_stance_len:.1f} cm" if est_stance_len else "N/A"))
                                print(f"Normalized Score (Manual Leg):    {norm_score_manual_pct:.2f}%")
                                if norm_score_video_pct is not None:
                                    print(f"Normalized Score (Video-Est Leg): {norm_score_video_pct:.2f}%")
                            else:
                                print(f"Scoring: EXCLUDED ({exclusion_reason})")
                            print("-" * 50)
                            
                            # Preserve 27 original columns, append extended SEBT scoring fields
                            csv_writer.writerow([
                                curr_movement_num, moving_ankle, start_frame_idx, max_frame_idx, start_timestamp, max_timestamp,
                                init_pos_red[0], init_pos_red[1], max_pos_red[0], max_pos_red[1], round(max_pixel_dist, 2),
                                round(max_depth_red, 4), round(max_depth_blue, 4), round(depth_diff, 4),
                                round(p_max_red_3d[0], 4), round(p_max_red_3d[1], 4), round(p_max_red_3d[2], 4),
                                round(p_max_blue_3d[0], 4), round(p_max_blue_3d[1], 4), round(p_max_blue_3d[2], 4),
                                round(dist_3d, 2),
                                standing_ankle, final_dir, final_conf, confirmed_frame, round(confirmed_time, 2), round(directional_disp_3d, 2),
                                round(manual_stance_len, 1) if manual_stance_len is not None else "",
                                round(est_stance_len, 1) if est_stance_len is not None else "",
                                round(stance_err_cm, 2) if stance_err_cm is not None else "",
                                round(stance_err_pct, 2) if stance_err_pct is not None else "",
                                round(valid_reach_dist_cm, 2) if valid_reach_dist_cm is not None else "",
                                round(norm_score_manual_pct, 2) if norm_score_manual_pct is not None else "",
                                round(norm_score_video_pct, 2) if norm_score_video_pct is not None else "",
                                scoring_validity,
                                exclusion_reason
                            ])
                            csv_f.flush()
                            
                            rec_dict = {
                                'movement': curr_movement_num,
                                'moving_ankle': moving_ankle,
                                'stance_leg': standing_ankle,
                                'detected_direction': final_dir,
                                'direction_confidence': final_conf,
                                'direction_angle': final_ang,
                                'directional_disp_cm': directional_disp_3d,
                                'valid_reach_dist_cm': valid_reach_dist_cm,
                                'manual_stance_len': manual_stance_len,
                                'est_stance_len': est_stance_len,
                                'stance_err_cm': stance_err_cm,
                                'stance_err_pct': stance_err_pct,
                                'norm_score_manual_pct': norm_score_manual_pct,
                                'norm_score_video_pct': norm_score_video_pct,
                                'scoring_validity': scoring_validity,
                                'scoring_exclusion_reason': exclusion_reason
                            }
                            movement_records.append(rec_dict)
                            results_summary.append((curr_movement_num, moving_ankle, standing_ankle, final_dir, final_ang, final_conf, directional_disp_3d, dist_3d))
                            print("RESULT SAVED")
                            
                            # Reset state to waiting
                            state = S_WAITING
                            consec_start = 0
                            curr_movement_num += 1
                            dist_hist = []
                            l_pos_hist = []
                            r_pos_hist = []
                            l_z_hist = []
                            r_z_hist = []
                            l_ankle_hist = []
                            r_ankle_hist = []
                            
                            max_pixel_dist = 0.0
                            max_reach_disp = 0.0
                            max_pos_red = None
                            max_pos_blue = None
                            max_frame_img = None
                    else:
                        consec_return = 0
                
            # Render the prominent SEBT Direction HUD Overlay directly onto frame
            draw_sebt_overlay(
                frame, state, moving_ankle, standing_ankle,
                curr_reach_disp, max_reach_disp, curr_dist_ankles, max_pixel_dist,
                provisional_dir, provisional_angle, last_confirmed_info, w, h
            )
            
            video_writer.write(frame)
            
            # Aspect-ratio preserving scaled copy for preview
            max_disp_h = 800
            scale = max_disp_h / h if h > max_disp_h else 1.0
            disp_w = int(w * scale)
            disp_h = int(h * scale)
            display_frame = cv2.resize(frame, (disp_w, disp_h))
            
            if total_frames > 0 and (frame_idx % 10 == 0 or frame_idx == total_frames):
                print(f"PROGRESS:frame:{frame_idx}:{total_frames}", flush=True)
            
            if not getattr(args, 'no_display', False):
                cv2.imshow("Offline Video Analysis", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    sys.exit(0)
                elif key == ord('r'):
                    break # break inner loop to restart
            else:
                key = 0
                
        # Inner loop ended
        if state == S_MOVING:
            print(f"MOVEMENT COMPLETE at end of video: frame {frame_idx} / {timestamp:.2f}s")
            depth_map = run_depth_inference(depth_model, max_frame_img, device)
            max_center_red, max_depth_red = sample_depth_neighborhood(depth_map, max_pos_red)
            max_center_blue, max_depth_blue = sample_depth_neighborhood(depth_map, max_pos_blue)
            
            depth_diff = abs(max_depth_red - max_depth_blue)
            
            p_max_red_3d = pixel_to_3d(max_pos_red, max_depth_red, CX, CY, FX, FY)
            p_max_blue_3d = pixel_to_3d(max_pos_blue, max_depth_blue, CX, CY, FX, FY)
            dist_3d = calculate_3d_distance(p_max_red_3d, p_max_blue_3d) * 100.0 # cm
            
            p_init_red_3d = pixel_to_3d(init_pos_red, max_depth_blue, CX, CY, FX, FY)
            directional_disp_3d = calculate_3d_distance(p_max_red_3d, p_init_red_3d) * 100.0 # cm
            
            dx_max = max_pos_red[0] - moving_base[0]
            dy_max = max_pos_red[1] - moving_base[1]
            final_dir, final_ang, final_conf, agree_ratio, valid_cnt = determine_final_direction(
                reach_trajectory, max_frame_idx, dx_max, dy_max,
                standing_ankle, u_body_left, u_body_fwd,
                torso_ang, baseline_torso_angle
            )
            
            duration = timestamp - start_timestamp
            
            # Stance leg selection for normalized scoring
            if standing_ankle == 'LEFT':
                manual_stance_len = manual_left_leg_length_cm
                est_stance_len = est_left_leg_cm
                stance_err_cm = left_error_cm
                stance_err_pct = left_error_pct
            elif standing_ankle == 'RIGHT':
                manual_stance_len = manual_right_leg_length_cm
                est_stance_len = est_right_leg_cm
                stance_err_cm = right_error_cm
                stance_err_pct = right_error_pct
            else:
                manual_stance_len = None
                est_stance_len = None
                stance_err_cm = None
                stance_err_pct = None

            VALID_SEBT_DIR_NAMES = {"ANT", "AM", "M", "PM", "POST", "PL", "LAT", "AL"}
            is_valid_score = True
            exclusion_reason = "VALID"

            if standing_ankle not in ('LEFT', 'RIGHT'):
                is_valid_score = False
                exclusion_reason = "EXCLUDED_INVALID_STANCE_LEG"
            elif final_dir not in VALID_SEBT_DIR_NAMES:
                is_valid_score = False
                if "Boundary" in final_dir:
                    exclusion_reason = "EXCLUDED_BOUNDARY_DIRECTION"
                elif final_dir == "Unknown":
                    exclusion_reason = "EXCLUDED_UNKNOWN_DIRECTION"
                else:
                    exclusion_reason = f"EXCLUDED_UNCONFIRMED_DIRECTION_{final_dir}"
            elif final_conf.startswith("Low"):
                is_valid_score = False
                exclusion_reason = "EXCLUDED_LOW_CONFIDENCE"
            elif directional_disp_3d <= 0.0:
                is_valid_score = False
                exclusion_reason = "EXCLUDED_ZERO_OR_NEGATIVE_REACH"
            elif manual_stance_len is None or manual_stance_len <= 0:
                is_valid_score = False
                exclusion_reason = "EXCLUDED_MISSING_MANUAL_LEG_LENGTH"

            if is_valid_score:
                valid_reach_dist_cm = directional_disp_3d
                norm_score_manual_pct = (valid_reach_dist_cm / manual_stance_len) * 100.0
                if est_stance_len and est_stance_len > 0:
                    norm_score_video_pct = (valid_reach_dist_cm / est_stance_len) * 100.0
                else:
                    norm_score_video_pct = None
                scoring_validity = "VALID"
            else:
                valid_reach_dist_cm = None
                norm_score_manual_pct = None
                norm_score_video_pct = None
                scoring_validity = "EXCLUDED"

            csv_writer.writerow([
                curr_movement_num, moving_ankle, start_frame_idx, max_frame_idx, start_timestamp, max_timestamp,
                init_pos_red[0], init_pos_red[1], max_pos_red[0], max_pos_red[1], round(max_pixel_dist, 2),
                round(max_depth_red, 4), round(max_depth_blue, 4), round(depth_diff, 4),
                round(p_max_red_3d[0], 4), round(p_max_red_3d[1], 4), round(p_max_red_3d[2], 4),
                round(p_max_blue_3d[0], 4), round(p_max_blue_3d[1], 4), round(p_max_blue_3d[2], 4),
                round(dist_3d, 2),
                standing_ankle, final_dir, final_conf, max_frame_idx, round(max_timestamp, 2), round(directional_disp_3d, 2),
                round(manual_stance_len, 1) if manual_stance_len is not None else "",
                round(est_stance_len, 1) if est_stance_len is not None else "",
                round(stance_err_cm, 2) if stance_err_cm is not None else "",
                round(stance_err_pct, 2) if stance_err_pct is not None else "",
                round(valid_reach_dist_cm, 2) if valid_reach_dist_cm is not None else "",
                round(norm_score_manual_pct, 2) if norm_score_manual_pct is not None else "",
                round(norm_score_video_pct, 2) if norm_score_video_pct is not None else "",
                scoring_validity,
                exclusion_reason
            ])
            csv_f.flush()
            
            rec_dict = {
                'movement': curr_movement_num,
                'moving_ankle': moving_ankle,
                'stance_leg': standing_ankle,
                'detected_direction': final_dir,
                'direction_confidence': final_conf,
                'direction_angle': final_ang,
                'directional_disp_cm': directional_disp_3d,
                'valid_reach_dist_cm': valid_reach_dist_cm,
                'manual_stance_len': manual_stance_len,
                'est_stance_len': est_stance_len,
                'stance_err_cm': stance_err_cm,
                'stance_err_pct': stance_err_pct,
                'norm_score_manual_pct': norm_score_manual_pct,
                'norm_score_video_pct': norm_score_video_pct,
                'scoring_validity': scoring_validity,
                'scoring_exclusion_reason': exclusion_reason
            }
            movement_records.append(rec_dict)
            results_summary.append((curr_movement_num, moving_ankle, standing_ankle, final_dir, final_ang, final_conf, directional_disp_3d, dist_3d))
            print("RESULT SAVED")

        cap.release()
        video_writer.release()
        csv_f.close()
        if not getattr(args, 'no_display', False):
            cv2.destroyAllWindows()
        
        # Calculate composite scores and generate summary CSV
        print("PROGRESS:status:Generating measurement CSV", flush=True)
        composite_results = calculate_composite_scores(movement_records)
        print("PROGRESS:status:Generating SEBT summary", flush=True)
        write_sebt_summary_csv(
            CSV_SUMMARY_PATH,
            manual_left_leg_length_cm, manual_right_leg_length_cm,
            est_left_leg_cm, est_right_leg_cm,
            left_error_cm, left_error_pct,
            right_error_cm, right_error_pct,
            movement_records, composite_results
        )
        print("PROGRESS:status:COMPLETE", flush=True)
        
        # Print Comprehensive Summary
        print("\n" + "="*85)
        print("FINAL SUMMARY - 8-DIRECTION SEBT DETECTION & NORMALIZED SCORING")
        print("="*85)
        print(f"Total Completed Movements: {len(movement_records)}")
        print(f"Output video saved to:\n  {OUT_VIDEO}")
        print(f"Output detailed CSV saved to:\n  {CSV_PATH}")
        print(f"Output summary CSV saved to:\n  {CSV_SUMMARY_PATH}")
        print("-" * 85)
        print("LEG LENGTH COMPARISON (cm):")
        l_est_str = f"{est_left_leg_cm:.1f} cm (Abs Error: {left_error_cm:.1f} cm, {left_error_pct:.1f}%)" if est_left_leg_cm else "UNAVAILABLE"
        r_est_str = f"{est_right_leg_cm:.1f} cm (Abs Error: {right_error_cm:.1f} cm, {right_error_pct:.1f}%)" if est_right_leg_cm else "UNAVAILABLE"
        print(f"  LEFT Leg  -> Manual: {manual_left_leg_length_cm:5.1f} cm | Video-Est: {l_est_str}")
        print(f"  RIGHT Leg -> Manual: {manual_right_leg_length_cm:5.1f} cm | Video-Est: {r_est_str}")
        print("-" * 85)
        # ──────────────────────────────────────────────────────────────────
        # TERMINAL SECTION 1: ALL INDIVIDUAL MOVEMENTS (TRIAL-BY-TRIAL LOG)
        # ──────────────────────────────────────────────────────────────────
        print("\n" + "="*110)
        print("TERMINAL SECTION 1 — ALL INDIVIDUAL MOVEMENTS (TRIAL-BY-TRIAL LOG)")
        print("Note: Every completed movement is displayed separately. Repeated directions are never merged or suppressed.")
        print("="*110)
        s1_header = f"{'Move':<5} | {'Reach':<6} | {'Stance':<6} | {'Direction':<10} | {'Angle':<10} | {'Confidence':<22} | {'Reach(cm)':<10} | {'Manual Score':<13} | {'Video Score':<13} | {'Status'}"
        print(s1_header)
        print("-" * len(s1_header))
        for r in movement_records:
            m_num = r.get('movement', 0)
            r_leg = r.get('moving_ankle', '--')
            s_leg = r.get('stance_leg', '--')
            d_dir = r.get('detected_direction', 'Unknown')
            ang = r.get('direction_angle')
            ang_str = f"{ang:+5.1f} deg" if ang is not None else "   --   "
            conf = r.get('direction_confidence', '--')
            reach_cm = f"{r.get('directional_disp_cm', 0.0):5.1f}"
            m_sc = f"{r['norm_score_manual_pct']:5.1f}%" if r.get('norm_score_manual_pct') is not None else "    --   "
            v_sc = f"{r['norm_score_video_pct']:5.1f}%" if r.get('norm_score_video_pct') is not None else "    --   "
            status = r.get('scoring_validity', '--')
            if status != 'VALID':
                status = f"EXCLUDED ({r.get('scoring_exclusion_reason', '')})"
            print(f"{m_num:<5} | {r_leg:<6} | {s_leg:<6} | {d_dir:<10} | {ang_str:<10} | {conf:<22} | {reach_cm:<10} | {m_sc:<13} | {v_sc:<13} | {status}")
        print("-" * len(s1_header))

        # ──────────────────────────────────────────────────────────────────
        # TERMINAL SECTION 2: COMPOSITE SEBT SCORES
        # ──────────────────────────────────────────────────────────────────
        print("\n" + "="*110)
        print("TERMINAL SECTION 2 — COMPOSITE SEBT SCORES")
        print("Clinical SEBT Protocol: When multiple trials are performed in the same direction,")
        print("the trial with maximum valid reach distance is selected. Each direction counts once.")
        print("Denominator is the actual number of distinct valid directions performed.")
        print("="*110)
        for grp_key in ['LEFT_STANCE', 'RIGHT_STANCE', 'COMBINED_ALL_STANCE']:
            res = composite_results.get(grp_key, {})
            status_lbl = res.get('status_label', 'Unavailable')
            denom = res.get('actual_denominator', 0)
            num_valid = res.get('num_valid_trials', 0)
            m_comp = f"{res.get('manual_composite_pct'):.2f}%" if res.get('manual_composite_pct') is not None else "UNAVAILABLE"
            v_comp = f"{res.get('video_composite_pct'):.2f}%" if res.get('video_composite_pct') is not None else "UNAVAILABLE"
            distinct_dirs = res.get('distinct_dirs', [])
            missing_dirs = res.get('missing_dirs', [])
            best_trials = res.get('best_trials', {})
            excluded_recs = res.get('excluded_records', [])

            print(f"\n[STANCE GROUP: {grp_key}] — {status_lbl}")
            print(f"  Distinct Valid Directions Count (Denominator): {denom}/8")
            print(f"  Total Valid Movements Evaluated: {num_valid}")
            print(f"  Manual-Length Composite Score:    {m_comp}")
            print(f"  Video-Estimated Composite Score:  {v_comp}")

            if denom > 0:
                print(f"  Included Directions Breakdown (Selected Maximum Reach Trials):")
                sub_hdr = f"    {'Direction':<10} | {'Selected Trial':<16} | {'Reach Dist (cm)':<16} | {'Manual Score':<14} | {'Video Score':<14}"
                print(sub_hdr)
                print("    " + "-" * (len(sub_hdr) - 4))
                for d in distinct_dirs:
                    tr = best_trials.get(d, {})
                    tr_m_num = f"Movement #{tr.get('movement')}"
                    tr_dist = f"{tr.get('valid_reach_dist_cm', 0.0):.1f} cm"
                    tr_m_sc = f"{tr.get('norm_score_manual_pct'):.1f}%" if tr.get('norm_score_manual_pct') is not None else "    --   "
                    tr_v_sc = f"{tr.get('norm_score_video_pct'):.1f}%" if tr.get('norm_score_video_pct') is not None else "    --   "
                    print(f"    {d:<10} | {tr_m_num:<16} | {tr_dist:<16} | {tr_m_sc:<14} | {tr_v_sc:<14}")

            if missing_dirs:
                print(f"  Missing Directions ({len(missing_dirs)}/8): {', '.join(missing_dirs)}")
            else:
                print("  Missing Directions: None (Complete 8/8 directions performed)")

            if excluded_recs:
                print(f"  Excluded / Non-Scoring Trials in Group ({len(excluded_recs)}):")
                for ex in excluded_recs:
                    print(f"    - Movement #{ex.get('movement')}: Reaching={ex.get('moving_ankle')}, Direction={ex.get('detected_direction')}, Reason={ex.get('scoring_exclusion_reason')}")

        print("\n" + "="*110 + "\n")
        
        if key != ord('r'):
            break

if __name__ == '__main__':
    main()
