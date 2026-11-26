"""
sebt_depth_prototype.py
══════════════════════════════════════════════════════════════════════
FRONT-VIEW 2-DIRECTION (MEDIAL / LATERAL) SEBT PROTOTYPE
with MediaPipe full-body Pose tracking + Depth Anything V2 comparison.

Flow:
  Camera
    -> 30cm hand/ruler calibration
    -> Leg-length marking (hip, ankle)     (needed for SEBT score normalization)
    -> Shoulder + Hip body reference       (front-facing only, no side/rotation)
    -> MANUAL reference-point marking      (index finger, fixed distance origin)
    -> Countdown -> MEDIAL: first-movement auto-locks reaching toe -> reach -> return
    -> Countdown -> LATERAL: same process
    -> Results: pixel-only baseline vs depth-assisted experimental comparison

══════════════════════════════════════════════════════════════════════
REFERENCE POINT vs. REACHING TOE — these are two different things
══════════════════════════════════════════════════════════════════════
REFERENCE POINT (calib.toe_ref_px):
  - Manually marked with the index-finger gesture, right after hip
    calibration — no leg-lift required.
  - Fixed distance origin for the entire session. Never moves.

REACHING TOE (DirectionResult.locked_toe_landmark):
  - NOT chosen by finger position. Detected automatically from
    MediaPipe Pose at the start of each direction's active test:
    both toes' starting positions are recorded, and whichever toe
    moves significantly farther during the participant's first
    movement is locked as the reaching toe for that direction.
  - Locked once per direction, then only ever read — never
    recomputed, so it cannot flicker between feet.
  - Shown with the large BLUE marker once locked.
══════════════════════════════════════════════════════════════════════

UNITS — READ THIS BEFORE INTERPRETING ANY OUTPUT
──────────────────────────────────────────────────────────────────────
Depth Anything V2 produces RELATIVE monocular depth — not metres, not
centimetres, and no universal conversion factor is invented anywhere
in this file.

  pixel_distance_cm                 -> METHOD 1, physically valid cm.
                                        Existing 30cm ruler calibration,
                                        unchanged. This is the baseline
                                        and the ONLY thing the SEBT
                                        score is computed from.

  relative_depth_difference         -> raw Depth Anything units, UNITLESS.
                                        maximum_depth_raw - reference_depth_raw.

  depth_assisted_experimental_measure -> EXPERIMENTAL ONLY. Combines a cm
                                        quantity (dx, dy) with a raw/
                                        unitless quantity (dz). NOT a
                                        physical distance. Always printed
                                        with an "[experimental]" tag and an
                                        "EXPERIMENTAL — NOT PHYSICAL CM" note.

No depth-based SEBT score is ever computed, anywhere in this file.
══════════════════════════════════════════════════════════════════════

MAX-REACH / DEPTH SYNCHRONIZATION (unchanged from the previous version)
──────────────────────────────────────────────────────────────────────
MediaPipe pose tracking runs every frame during the active reach.
Whenever a new maximum pixel-distance occurs, the exact toe pixel AND a
full copy of that exact frame are saved. Depth Anything does NOT run
during live tracking. Only once the participant has returned to the
reference point does Depth Anything run — exactly once, on the SAVED
maximum-reach frame, sampled at the SAVED maximum toe position.
══════════════════════════════════════════════════════════════════════
"""

import csv
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import mediapipe as mp


# ══════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CHECKPOINT_PATH   = os.path.join('checkpoints', 'depth_anything_v2_vits.pth')
ENCODER           = 'vits'
DEPTH_INPUT_SIZE  = 518
KNOWN_DISTANCE_CM = 30.0
CSV_PATH          = 'sebt_depth_prototype_results.csv'
CAM_WIDTH, CAM_HEIGHT = 960, 720

HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'
# Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
POSE_LANDMARKER_MODEL_PATH = 'pose_landmarker.task'
# Download ("lite" variant, recommended for CPU):
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

MARK_COOLDOWN_SEC         = 1.0     # debounce for calibration point marking
FIRST_MOVEMENT_THRESHOLD_PX = 25    # sensitivity for auto-detecting the reaching toe
MOVED_AWAY_THRESHOLD_PX   = 80      # same thresholds as the existing SEBT app
RETURN_TOLERANCE_PX       = 35
HIP_TOLERANCE_PX          = 80      # lightweight stance-drift warning during reach
DIRECTION_EPSILON_PX      = 10      # small dead-zone for medial/lateral direction check

DIRECTION_ORDER = ['MEDIAL', 'LATERAL']
DIRECTION_LABELS = {
    'MEDIAL':  'MEDIAL — reach across the midline, toward your stance leg',
    'LATERAL': 'LATERAL — reach away from the midline, outward',
}

CSV_FIELDS = [
    'session_id', 'timestamp', 'tested_leg', 'direction',
    'leg_length_cm',
    'pixel_reach_cm',
    'reference_depth_raw', 'maximum_depth_raw', 'relative_depth_difference',
    'maximum_toe_x', 'maximum_toe_y',
    'depth_assisted_experimental_measure', 'difference_from_pixel_baseline',
]

# Pose Landmarker (Tasks API) indices — standard 33-point MediaPipe Pose
# topology, same indices the existing sebt.html JS code already uses.
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP           = 23, 24
L_ANKLE, R_ANKLE       = 27, 28
L_TOE, R_TOE           = 31, 32

# Standard MediaPipe Pose connections — enough for a clear full-body
# skeleton overlay (head/shoulders/arms/hips/legs/ankles/feet).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),            # shoulders/arms
    (11, 23), (12, 24), (23, 24),                                # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),            # left leg/foot
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),            # right leg/foot
    (9, 10), (0, 11), (0, 12),                                   # face/neck approx
]
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
    (5, 9),
    (9, 13),
    (13, 17),
]

# States
S_CALIB_RULER      = 'CALIB_RULER'
S_CALIB_LEG_LENGTH = 'CALIB_LEG_LENGTH'
S_CALIB_SHOULDER   = 'CALIB_SHOULDER'
S_CALIB_HIP        = 'CALIB_HIP'
S_MARK_REFERENCE   = 'MARK_REFERENCE'   # manual reference-point mark (separate from reaching-toe lock)
S_COUNTDOWN        = 'COUNTDOWN'
S_TESTING          = 'TESTING'
S_BREAK            = 'BREAK'
S_RESULTS          = 'RESULTS'

COUNTDOWN_SEC = 3   # 3-2-1-GO — unchanged, never shortened
BREAK_SEC     = 2

# States that need the hand detector running
HAND_STATES = (S_CALIB_RULER, S_CALIB_LEG_LENGTH, S_CALIB_SHOULDER, S_CALIB_HIP, S_MARK_REFERENCE)
# States that need the pose detector running (reference marking no longer needs pose at all)
POSE_STATES = (S_TESTING,)


# ══════════════════════════════════════════════════════════════════
# §1  PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════
def check_required_files():
    """Checks every required file up front and fails loudly with a
    complete list, rather than dying partway through startup."""
    missing = []
    if not os.path.exists(CHECKPOINT_PATH):
        missing.append(f'Depth Anything checkpoint: {CHECKPOINT_PATH}')
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        missing.append(
            f'Hand Landmarker model: {HAND_LANDMARKER_MODEL_PATH}\n'
            '      Download: https://storage.googleapis.com/mediapipe-models/'
            'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        )
    if not os.path.exists(POSE_LANDMARKER_MODEL_PATH):
        missing.append(
            f'Pose Landmarker model: {POSE_LANDMARKER_MODEL_PATH}\n'
            '      Download: https://storage.googleapis.com/mediapipe-models/'
            'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        )
    if missing:
        print('ERROR: Required file(s) missing — cannot start.\n')
        for m in missing:
            print(f'  - {m}')
        print()
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════
# §2  CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════════
def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam (index 0). Check camera permissions/connection.')
    return cap


def read_frame(cap):
    """Flips horizontally once; every downstream coordinate (hands, pose,
    depth map) is computed on this SAME flipped frame for consistency."""
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)


# ══════════════════════════════════════════════════════════════════
# §3  MEDIAPIPE TASKS — HANDS (calibration marking) + POSE (body tracking)
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


def _landmark_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detect_pointing_finger(hand_landmarks, w, h):
    """Index tip far from wrist relative to its MCP, other three
    fingertips not extended — same rule the existing SEBT app uses."""
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
    """Draw the complete hand skeleton and highlight the pointing fingertip."""

    if hand_landmarks is None:
        return

    # Convert all hand landmarks to pixel coordinates
    pts = [_landmark_px(lm, w, h) for lm in hand_landmarks]

    # Draw hand/finger bone connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pts[a],
            pts[b],
            (255, 255, 0),   # cyan/yellow-ish in BGR
            2
        )

    # Draw all hand landmarks
    for i, (x, y) in enumerate(pts):
        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 0, 255),  # magenta
            -1
        )

    # Highlight index fingertip
    if tip_px is not None:
        color = (0, 255, 0) if is_pointing else (0, 0, 255)

        cv2.circle(frame, tip_px, 10, color, -1)
        cv2.circle(frame, tip_px, 18, color, 2)


def draw_pose_skeleton(frame, pose_landmarks, w, h):
    """Full-body skeleton overlay — visual feedback only, does not feed
    back into any measurement (measurement always reads landmarks fresh)."""
    pts = [_landmark_px(lm, w, h) for lm in pose_landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 180), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)


# ══════════════════════════════════════════════════════════════════
# §4  DEPTH ANYTHING V2 INFERENCE — model unchanged.
#     Calling strategy: ONCE per completed direction, on the saved
#     maximum-reach frame — never during live per-frame tracking.
# ══════════════════════════════════════════════════════════════════
def init_depth_model():
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(device).eval()
    print(f'[depth] Loaded Depth Anything V2 ({ENCODER}) on device: {device}')
    return model, device


def run_depth_inference(model, frame_bgr, device):
    with torch.no_grad():
        return model.infer_image(frame_bgr, input_size=DEPTH_INPUT_SIZE)


def sample_depth(depth_map, px):
    h, w = depth_map.shape[:2]
    x = int(np.clip(px[0], 0, w - 1))
    y = int(np.clip(px[1], 0, h - 1))
    return float(depth_map[y, x])


# ══════════════════════════════════════════════════════════════════
# §5  CALIBRATION STATE (ruler / leg length / shoulder / hip / reference point)
# ══════════════════════════════════════════════════════════════════
class CalibState:
    def __init__(self):
        self.ruler_points = []          # 2 hand-marked points, 30cm apart
        self.px_per_cm = 0.0
        self.leg_points = []            # [hip, ankle]
        self.leg_length_cm = 0.0
        self.shoulder_points = []       # [left, right]
        self.hip_point = None           # also used as the body-midline proxy for direction checks
        self.toe_ref_px = None          # MANUAL reference point — fixed distance origin, never moves
        self.toe_ref_depth = None       # raw depth sampled ONCE, at marking time
        self.last_mark_time = 0.0

    def is_complete(self):
        return self.toe_ref_px is not None


def try_mark_calib_point(calib, is_pointing, tip_px):
    """Generic cooldown-debounced point marker, reused across ruler/leg/shoulder/hip/reference."""
    now = time.time()
    if not is_pointing or (now - calib.last_mark_time) < MARK_COOLDOWN_SEC:
        return False
    calib.last_mark_time = now
    return True


# ══════════════════════════════════════════════════════════════════
# §6  PER-DIRECTION TRIAL STATE (MEDIAL / LATERAL)
# ══════════════════════════════════════════════════════════════════
class DirectionResult:
    def __init__(self):
        self.completed = False

        # Reaching-toe auto-detection (first-movement method) — NEW per FIX 2.
        # Locked once per direction (including on redo), then only ever read.
        self.initial_l_toe_px = None
        self.initial_r_toe_px = None
        self.locked_toe_landmark = None   # 31 or 32, set once first valid movement is detected

        # Live tracking (every frame once locked, MediaPipe only — cheap)
        self.max_pixel_dist = 0.0
        self.max_toe_px = None
        self.max_frame = None       # COMPLETE frame image saved at the moment of each new max
        self.has_moved_away = False

        # Filled in once, by finalize_direction(), after return-to-reference
        self.pixel_distance_cm = 0.0                       # METHOD 1 — physically valid cm baseline
        self.reference_depth_raw = None                    # raw, from calibration
        self.maximum_depth_raw = None                       # raw, from the SAME frame as max_toe_px
        self.relative_depth_difference = 0.0                # raw, UNITLESS
        self.depth_assisted_experimental_measure = 0.0       # EXPERIMENTAL — NOT centimetres
        self.difference_from_pixel_baseline = 0.0            # raw numeric diff, NOT a physical cm difference

    def reset(self):
        self.__init__()


def try_lock_reaching_toe(dr, pose_landmarks, w, h):
    """
    FIX 2 — reaching-toe auto-detection, separate from the manual
    reference point. Called every TESTING frame while dr.locked_toe_landmark
    is still None.

    First call: records both toes' starting positions.
    Subsequent calls: whichever toe has moved farther from ITS OWN
    starting position, once past FIRST_MOVEMENT_THRESHOLD_PX, is locked
    as the reaching toe. After that this function is never called again
    for this direction (guarded by the caller checking locked_toe_landmark).
    """
    l_now = _landmark_px(pose_landmarks[L_TOE], w, h)
    r_now = _landmark_px(pose_landmarks[R_TOE], w, h)

    if dr.initial_l_toe_px is None:
        dr.initial_l_toe_px = l_now
        dr.initial_r_toe_px = r_now
        return  # baseline captured this frame; can't detect movement yet

    l_moved = _dist(l_now, dr.initial_l_toe_px)
    r_moved = _dist(r_now, dr.initial_r_toe_px)

    if max(l_moved, r_moved) > FIRST_MOVEMENT_THRESHOLD_PX:
        dr.locked_toe_landmark = L_TOE if l_moved > r_moved else R_TOE
        print(f'[test] Reaching toe auto-locked: landmark {dr.locked_toe_landmark} '
              f'(L moved {l_moved:.0f}px, R moved {r_moved:.0f}px)')


def check_direction_valid(direction, toe_px, calib):
    """
    Basic directional gate (kept intentionally simple, per spec):
    uses the marked hip-center x-coordinate as a body-midline proxy.
    MEDIAL must move the toe TOWARD that midline (relative to where the
    reference point started); LATERAL must move it AWAY. Movement in the
    wrong direction does not count toward max_pixel_dist.
    """
    if calib.hip_point is None or calib.toe_ref_px is None:
        return True  # nothing to validate against yet — don't block
    midline_x = calib.hip_point[0]
    ref_x = calib.toe_ref_px[0]
    side_sign = 1 if ref_x >= midline_x else -1
    ref_extent = (ref_x - midline_x) * side_sign
    cur_extent = (toe_px[0] - midline_x) * side_sign
    if direction == 'MEDIAL':
        return cur_extent < ref_extent - DIRECTION_EPSILON_PX
    if direction == 'LATERAL':
        return cur_extent > ref_extent + DIRECTION_EPSILON_PX
    return True


# ══════════════════════════════════════════════════════════════════
# §7  RESULTS / SCORING
# ══════════════════════════════════════════════════════════════════
def finalize_direction(dr, calib, depth_model, device):
    """
    Called once, right after return-to-reference completes a direction.
    METHOD 1 (pixel_distance_cm) is pure geometry — computed unconditionally.
    METHOD 2 requires Depth Anything, run here EXACTLY ONCE, on the saved
    maximum-reach frame — this synchronization is UNCHANGED from before.
    """
    dr.pixel_distance_cm = dr.max_pixel_dist / calib.px_per_cm if calib.px_per_cm > 0 else 0.0
    dr.reference_depth_raw = calib.toe_ref_depth

    if dr.max_frame is not None and dr.max_toe_px is not None and calib.toe_ref_depth is not None:
        depth_map = run_depth_inference(depth_model, dr.max_frame, device)
        dr.maximum_depth_raw = sample_depth(depth_map, dr.max_toe_px)
        dr.relative_depth_difference = dr.maximum_depth_raw - dr.reference_depth_raw

        dx_cm = (dr.max_toe_px[0] - calib.toe_ref_px[0]) / calib.px_per_cm if calib.px_per_cm > 0 else 0.0
        dy_cm = (dr.max_toe_px[1] - calib.toe_ref_px[1]) / calib.px_per_cm if calib.px_per_cm > 0 else 0.0
        dz_raw = dr.relative_depth_difference
        # EXPERIMENTAL ONLY — mixes cm (dx, dy) with raw/unitless depth (dz).
        # NOT a physical distance. See module docstring.
        dr.depth_assisted_experimental_measure = math.sqrt(dx_cm ** 2 + dy_cm ** 2 + dz_raw ** 2)
        dr.difference_from_pixel_baseline = dr.depth_assisted_experimental_measure - dr.pixel_distance_cm
    else:
        # No maximum frame was captured (e.g. reach never exceeded a valid-
        # direction max) — leave depth fields empty rather than guessing.
        dr.maximum_depth_raw = None
        dr.relative_depth_difference = 0.0
        dr.depth_assisted_experimental_measure = dr.pixel_distance_cm
        dr.difference_from_pixel_baseline = 0.0

    dr.completed = True
    dr.max_frame = None  # no longer needed once depth has been sampled from it


def compute_2dir_pixel_score(results, calib):
    """
    Same methodology as the existing 8-direction score (avg reach / leg
    length * 100), applied to only the 2 completed directions. Explicitly
    NOT the final 8-direction SEBT score. Only ever computed from
    pixel_distance_cm — no depth-based score is ever computed (see
    module docstring).
    """
    done = [d for d in DIRECTION_ORDER if results[d].completed]
    if not done or calib.leg_length_cm <= 0:
        return None
    avg_pixel = sum(results[d].pixel_distance_cm for d in done) / len(done)
    return (avg_pixel / calib.leg_length_cm) * 100


# ══════════════════════════════════════════════════════════════════
# §8  CSV LOGGING
# ══════════════════════════════════════════════════════════════════
def init_csv(path=CSV_PATH):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def log_direction_to_csv(session_id, tested_leg, direction, dr, calib, path=CSV_PATH):
    row = {
        'session_id': session_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'tested_leg': tested_leg,
        'direction': direction,
        'leg_length_cm': round(calib.leg_length_cm, 2),
        'pixel_reach_cm': round(dr.pixel_distance_cm, 3),
        'reference_depth_raw': round(dr.reference_depth_raw, 6) if dr.reference_depth_raw is not None else '',
        'maximum_depth_raw': round(dr.maximum_depth_raw, 6) if dr.maximum_depth_raw is not None else '',
        'relative_depth_difference': round(dr.relative_depth_difference, 6),
        'maximum_toe_x': dr.max_toe_px[0] if dr.max_toe_px else '',
        'maximum_toe_y': dr.max_toe_px[1] if dr.max_toe_px else '',
        'depth_assisted_experimental_measure': round(dr.depth_assisted_experimental_measure, 3),
        'difference_from_pixel_baseline': round(dr.difference_from_pixel_baseline, 3),
    }
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
    print(f'[csv] {direction} logged -> {path}')


# ══════════════════════════════════════════════════════════════════
# §9  UI DRAWING
# ══════════════════════════════════════════════════════════════════
def draw_top_banner(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_calib_markers(frame, points, labels):
    for (x, y), label in zip(points, labels):
        cv2.circle(frame, (x, y), 14, (0, 255, 255), -1)
        cv2.putText(frame, label, (x + 18, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_reference_and_toe(frame, calib, current_toe_px, max_toe_px=None):
    if calib.toe_ref_px is not None:
        cv2.circle(frame, calib.toe_ref_px, 16, (0, 255, 0), -1)
        cv2.putText(frame, 'REF', (calib.toe_ref_px[0] + 20, calib.toe_ref_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if current_toe_px is not None:
        # Large blue point — same visual language as the existing SEBT app
        cv2.circle(frame, current_toe_px, 20, (255, 150, 0), -1)
        cv2.putText(frame, 'TOE', (current_toe_px[0] + 24, current_toe_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    if calib.toe_ref_px is not None and current_toe_px is not None:
        cv2.line(frame, calib.toe_ref_px, current_toe_px, (0, 255, 255), 2)
    if max_toe_px is not None:
        cv2.circle(frame, max_toe_px, 10, (0, 0, 255), 2)
        cv2.putText(frame, 'MAX', (max_toe_px[0] + 14, max_toe_px[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_side_panel(frame, lines, color=(0, 255, 255)):
    panel_h = 22 * len(lines) + 20
    y0 = frame.shape[0] - panel_h
    cv2.rectangle(frame, (0, y0 - 6), (480, frame.shape[0]), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, y0 + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_controls_hint(frame, text):
    cv2.putText(frame, text, (frame.shape[1] - 330, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def print_results_table(results, score_pixel):
    print('\n' + '=' * 90)
    print('2-DIRECTION SEBT PROTOTYPE RESULTS  (MEDIAL + LATERAL only — NOT the')
    print('full 8-direction SEBT score)')
    print('=' * 90)
    for d in DIRECTION_ORDER:
        r = results[d]
        print(f'\n{d}')
        if not r.completed:
            print('    — not completed —')
            continue
        print(f'    Pixel-only reach:                  {r.pixel_distance_cm:.2f} cm')
        print(f'    Relative depth difference:         {r.relative_depth_difference:.4f} raw units')
        print(f'    Depth-assisted experimental measure: {r.depth_assisted_experimental_measure:.2f} [experimental]')
        print('    NOTE: Depth Anything V2 output is relative depth. The experimental')
        print('          value is NOT a physical centimetre measurement.')
    print('\n' + '-' * 90)
    if score_pixel is not None:
        print(f'2-direction SEBT score (pixel-only, physically valid): {score_pixel:.1f}%')
    else:
        print('2-direction SEBT score: unavailable (leg length or directions missing)')
    print('No depth-based score is computed — see notes above.')
    print('=' * 90 + '\n')


# ══════════════════════════════════════════════════════════════════
# §10  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    check_required_files()

    tested_leg = ''
    while tested_leg not in ('L', 'R'):
        tested_leg = input('Which leg is the STANCE (tested) leg? Enter L or R: ').strip().upper()
    tested_leg_full = 'Left' if tested_leg == 'L' else 'Right'
    print(f'Tested (stance) leg: {tested_leg_full}. The OTHER leg reaches for MEDIAL/LATERAL.')

    print('Loading Depth Anything V2 VITS...')
    depth_model, device = init_depth_model()
    init_csv()

    cap = init_camera()
    hand_landmarker = init_hand_detector()
    pose_landmarker = init_pose_detector()

    calib = CalibState()
    results = {d: DirectionResult() for d in DIRECTION_ORDER}
    dir_idx = 0
    state = S_CALIB_RULER
    countdown_start = None
    break_start = None
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # FIX 1 — when True, the direction currently running was started via
    # Redo (key 1/2), and must go straight to S_RESULTS on completion
    # instead of chaining into S_BREAK / the other direction.
    redo_active = False

    last_hand_ts = 0
    last_pose_ts = 0
    window_name = 'SEBT + Depth Anything — 2-Direction Prototype (MEDIAL/LATERAL)'

    def next_ts(last):
        t = int(time.time() * 1000)
        return t if t > last else last + 1

    print('Camera started. Press Q to quit, F for full reset, 1/2 to redo Medial/Lateral.')

    while True:
        frame = read_frame(cap)
        if frame is None:
            print('[camera] Failed to read frame — stopping.')
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Hands (calibration/marking states only) ──
        is_pointing, tip_px = False, None
        if state in HAND_STATES:
            last_hand_ts = next_ts(last_hand_ts)
            hand_result = hand_landmarker.detect_for_video(mp_image, last_hand_ts)
            if hand_result.hand_landmarks:
                hl = hand_result.hand_landmarks[0]
                is_pointing, tip_px = detect_pointing_finger(hl, w, h)
                draw_hand_pointer(frame, hl, w, h, tip_px, is_pointing)

        # ── Pose (active testing only — reference marking no longer needs it) ──
        pose_landmarks = None
        if state in POSE_STATES:
            last_pose_ts = next_ts(last_pose_ts)
            pose_result = pose_landmarker.detect_for_video(mp_image, last_pose_ts)
            if pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                draw_pose_skeleton(frame, pose_landmarks, w, h)

        # ══════════════════════ CALIBRATION ══════════════════════
        if state == S_CALIB_RULER:
            draw_top_banner(frame, f'STEP 1: Mark ruler point 1 & 2 ({KNOWN_DISTANCE_CM:.0f}cm apart)')
            draw_calib_markers(frame, calib.ruler_points, ['P1', 'P2'][:len(calib.ruler_points)])
            if try_mark_calib_point(calib, is_pointing, tip_px) and len(calib.ruler_points) < 2:
                calib.ruler_points.append(tip_px)
                if len(calib.ruler_points) == 2:
                    px_dist = _dist(calib.ruler_points[0], calib.ruler_points[1])
                    if px_dist <= 0:
                        print('[calib] Ruler points identical — please re-mark.')
                        calib.ruler_points = []
                    else:
                        calib.px_per_cm = px_dist / KNOWN_DISTANCE_CM
                        print(f'[calib] Ruler: {px_dist:.1f}px = {KNOWN_DISTANCE_CM}cm '
                              f'-> {calib.px_per_cm:.3f} px/cm')
                        state = S_CALIB_LEG_LENGTH

        elif state == S_CALIB_LEG_LENGTH:
            draw_top_banner(frame, 'STEP 2: Mark HIP, then ANKLE (leg length)')
            draw_calib_markers(frame, calib.leg_points, ['HIP', 'ANKLE'][:len(calib.leg_points)])
            if try_mark_calib_point(calib, is_pointing, tip_px) and len(calib.leg_points) < 2:
                calib.leg_points.append(tip_px)
                if len(calib.leg_points) == 2:
                    leg_px = _dist(calib.leg_points[0], calib.leg_points[1])
                    calib.leg_length_cm = leg_px / calib.px_per_cm if calib.px_per_cm > 0 else 0.0
                    print(f'[calib] Leg length: {calib.leg_length_cm:.1f} cm')
                    state = S_CALIB_SHOULDER

        elif state == S_CALIB_SHOULDER:
            draw_top_banner(frame, 'STEP 3: Face camera. Mark LEFT shoulder, then RIGHT shoulder')
            draw_calib_markers(frame, calib.shoulder_points,
                                ['L.SHOULDER', 'R.SHOULDER'][:len(calib.shoulder_points)])
            if try_mark_calib_point(calib, is_pointing, tip_px) and len(calib.shoulder_points) < 2:
                calib.shoulder_points.append(tip_px)
                if len(calib.shoulder_points) == 2:
                    state = S_CALIB_HIP

        elif state == S_CALIB_HIP:
            draw_top_banner(frame, 'STEP 4: Mark HIP CENTER')
            draw_calib_markers(frame, [calib.hip_point] if calib.hip_point else [], ['HIP'])
            if try_mark_calib_point(calib, is_pointing, tip_px) and calib.hip_point is None:
                calib.hip_point = tip_px
                print('[calib] Shoulder + hip reference saved (front-facing only).')
                state = S_MARK_REFERENCE

        # ══════════════════════ MANUAL REFERENCE POINT (FIX 2) ══════════════════════
        elif state == S_MARK_REFERENCE:
            draw_top_banner(frame, 'STEP 5: Mark your SEBT REFERENCE POINT (index finger)')
            if calib.toe_ref_px is None and try_mark_calib_point(calib, is_pointing, tip_px):
                calib.toe_ref_px = tip_px
                # Reference depth sampled ONCE, right now — this point never moves again.
                depth_map = run_depth_inference(depth_model, frame, device)
                calib.toe_ref_depth = sample_depth(depth_map, calib.toe_ref_px)
                print(f'[calib] Reference point locked at {calib.toe_ref_px}, '
                      f'ref depth(raw)={calib.toe_ref_depth:.4f}')
                state = S_COUNTDOWN
                countdown_start = time.time()
            draw_reference_and_toe(frame, calib, None)

        # ══════════════════════ COUNTDOWN ══════════════════════
        elif state == S_COUNTDOWN:
            direction = DIRECTION_ORDER[dir_idx]
            remaining = COUNTDOWN_SEC - int(time.time() - countdown_start)
            if remaining > 0:
                draw_top_banner(frame, f'Get ready: {DIRECTION_LABELS[direction]}')
                cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
            else:
                draw_top_banner(frame, 'GO!')
                state = S_TESTING
                results[direction].reset()
                print(f'[test] Starting {direction}')

        # ══════════════════════ TESTING (reach) ══════════════════════
        elif state == S_TESTING:
            direction = DIRECTION_ORDER[dir_idx]
            dr = results[direction]
            draw_top_banner(frame, DIRECTION_LABELS[direction])

            if pose_landmarks is not None and calib.toe_ref_px is not None:

                # ── FIX 2: reaching toe not yet locked — try to auto-detect it ──
                if dr.locked_toe_landmark is None:
                    try_lock_reaching_toe(dr, pose_landmarks, w, h)
                    draw_reference_and_toe(frame, calib, None)
                    draw_side_panel(frame, [
                        f'{direction}: waiting for first movement to identify reaching leg...',
                        'Move either foot noticeably to lock the reaching toe',
                    ])
                else:
                    # Toe identity was locked once above and is only ever READ
                    # from here on — never re-derived, so it cannot flicker.
                    toe_px = _landmark_px(pose_landmarks[dr.locked_toe_landmark], w, h)
                    pixel_dist = _dist(toe_px, calib.toe_ref_px)

                    # Lightweight stance-drift warning (does not block/finish the trial).
                    opposite_hip_idx = R_HIP if dr.locked_toe_landmark == L_TOE else L_HIP
                    hip_now = _landmark_px(pose_landmarks[opposite_hip_idx], w, h)
                    if calib.hip_point is not None and _dist(hip_now, calib.hip_point) > HIP_TOLERANCE_PX:
                        cv2.putText(frame, 'Stance drifted — keep supporting leg steady',
                                    (12, h - 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    valid_direction = check_direction_valid(direction, toe_px, calib)
                    if not valid_direction and pixel_dist > MOVED_AWAY_THRESHOLD_PX / 2:
                        cv2.putText(frame, f'Wrong direction for {direction} — not counted',
                                    (12, h - 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

                    # MediaPipe runs every frame; only a NEW MAX in the correct
                    # direction saves the toe position + a full frame copy for
                    # later (and only later) depth sampling. UNCHANGED (FIX-2
                    # only changed WHICH toe is tracked, not this sync logic).
                    if pixel_dist > dr.max_pixel_dist and valid_direction:
                        dr.max_pixel_dist = pixel_dist
                        dr.max_toe_px = toe_px
                        dr.max_frame = frame.copy()
                    if pixel_dist > MOVED_AWAY_THRESHOLD_PX:
                        dr.has_moved_away = True

                    draw_reference_and_toe(frame, calib, toe_px, dr.max_toe_px)
                    draw_side_panel(frame, [
                        f'{direction}: current {pixel_dist / calib.px_per_cm:.1f} cm '
                        f'| max {dr.max_pixel_dist / calib.px_per_cm:.1f} cm',
                        'Depth Anything runs once you return to the reference point',
                        'Return BLUE toe to GREEN reference to finish'
                        if dr.has_moved_away else 'Reach out, then return to the reference point',
                    ])

                    returned = dr.has_moved_away and pixel_dist < RETURN_TOLERANCE_PX
                    if returned:
                        finalize_direction(dr, calib, depth_model, device)
                        log_direction_to_csv(session_id, tested_leg_full, direction, dr, calib)
                        print(f'[test] {direction} complete: pixel={dr.pixel_distance_cm:.1f}cm '
                              f'depth_diff(raw)={dr.relative_depth_difference:.4f} '
                              f'depth_assisted(EXP.)={dr.depth_assisted_experimental_measure:.4f}')

                        # ── FIX 1: redo never auto-chains into the other direction ──
                        if redo_active:
                            redo_active = False
                            state = S_RESULTS
                        elif dir_idx < len(DIRECTION_ORDER) - 1:
                            state = S_BREAK
                            break_start = time.time()
                        else:
                            state = S_RESULTS

        # ══════════════════════ BREAK (transition) ══════════════════════
        elif state == S_BREAK:
            direction_done = DIRECTION_ORDER[dir_idx]
            remaining = BREAK_SEC - int(time.time() - break_start)
            draw_top_banner(frame, f'{direction_done} complete!')
            if remaining > 0:
                cv2.putText(frame, f'Next: {DIRECTION_ORDER[dir_idx + 1]} in {remaining}s',
                            (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                dir_idx += 1
                state = S_COUNTDOWN
                countdown_start = time.time()

        # ══════════════════════ RESULTS ══════════════════════
        elif state == S_RESULTS:
            score_pixel = compute_2dir_pixel_score(results, calib)
            draw_top_banner(frame, '2-DIRECTION PROTOTYPE RESULTS (NOT full 8-dir SEBT score)')
            lines = []
            for d in DIRECTION_ORDER:
                r = results[d]
                if r.completed:
                    lines.append(f'{d}: pixel={r.pixel_distance_cm:.1f}cm  '
                                  f'depth_diff(raw)={r.relative_depth_difference:.3f}  '
                                  f'depth_assisted={r.depth_assisted_experimental_measure:.2f} [experimental]')
                else:
                    lines.append(f'{d}: not completed')
            if score_pixel is not None:
                lines.append(f'2-dir SEBT score (pixel-only): {score_pixel:.1f}%')
            else:
                lines.append('Score unavailable (leg length or directions missing)')
            lines.append('EXPERIMENTAL — NOT PHYSICAL CM: depth-assisted value mixes cm')
            lines.append('with raw relative depth units. No depth-based score is computed.')
            lines.append('Press P for full detail in the console.')
            draw_side_panel(frame, lines, color=(0, 255, 0))

        draw_controls_hint(frame, 'F=Full Reset  1=Redo Medial  2=Redo Lateral  P=Print  Q=Quit')
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print('Quitting.')
            break

        if key in (ord('f'), ord('F')):
            # Full reset — clears EVERYTHING (ruler, leg length, shoulder,
            # hip, reference point, both directions) and restarts calibration.
            calib = CalibState()
            results = {d: DirectionResult() for d in DIRECTION_ORDER}
            dir_idx = 0
            state = S_CALIB_RULER
            redo_active = False
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            print('[reset] Full reset — recalibrating from scratch.')

        # Redo (FIX 1): only allowed when not actively counting down/testing/
        # transitioning — mirrors isTestInProgress() gating in the existing
        # JS app. Resets ONLY the chosen direction's result; calibration and
        # the manual reference point are untouched. redo_active ensures the
        # OTHER direction's result is left exactly as it was and the flow
        # goes straight to RESULTS afterward, never auto-chaining onward.
        test_in_progress = state in (S_COUNTDOWN, S_TESTING, S_BREAK)
        if key == ord('1') and not test_in_progress and calib.is_complete():
            results['MEDIAL'].reset()
            dir_idx = DIRECTION_ORDER.index('MEDIAL')
            redo_active = True
            state = S_COUNTDOWN
            countdown_start = time.time()
            print('[redo] Redoing MEDIAL only — calibration, reference point, '
                  'and the LATERAL result are kept.')
        if key == ord('2') and not test_in_progress and calib.is_complete():
            results['LATERAL'].reset()
            dir_idx = DIRECTION_ORDER.index('LATERAL')
            redo_active = True
            state = S_COUNTDOWN
            countdown_start = time.time()
            print('[redo] Redoing LATERAL only — calibration, reference point, '
                  'and the MEDIAL result are kept.')

        if key in (ord('p'), ord('P')) and state == S_RESULTS:
            print_results_table(results, compute_2dir_pixel_score(results, calib))

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except SystemExit:
        raise
    except Exception as exc:
        print(f'\nFATAL ERROR: {exc}')
        sys.exit(1) 