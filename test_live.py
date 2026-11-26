"""
ruler_calibration_test.py
══════════════════════════════════════════════════════════════════════
Standalone experiment: is Depth Anything V2's relative depth output
calibratable against a known physical distance (a 30 cm ruler)?

This file is completely independent of the existing SEBT web app.
It does not modify, import, or depend on anything under the SEBT
project other than the already-verified Depth Anything V2 checkpoint.

Workflow (see README-style docstring in main() for the state machine):
  1. Open webcam, show live video.
  2. Mark Point 1 with an index-finger pointing gesture (MediaPipe Hands).
  3. Mark Point 2 the same way.
  4. Compute pixel distance -> pixel-to-cm scale (30 cm known distance).
  5. Run Depth Anything V2 ONCE on that frame, sample depth at P1 & P2.
  6. Display everything, log the trial to calibration_results.csv.
  7. R = reset for a new trial. Q = quit.

IMPORTANT: Depth Anything's raw output is RELATIVE monocular depth —
this script never assumes it is metres/centimetres and never applies
any invented depth->cm conversion. It only records depth1, depth2,
and abs(depth2 - depth1) for later analysis.
══════════════════════════════════════════════════════════════════════
"""

import csv
import os
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
DEPTH_INPUT_SIZE  = 518          # standard Depth Anything V2 inference size
KNOWN_DISTANCE_CM = 30.0         # physical distance between the two ruler points
CSV_PATH          = 'calibration_results.csv'
MARK_COOLDOWN_SEC = 1.0          # debounce so one gesture doesn't mark twice
CAM_WIDTH, CAM_HEIGHT = 960, 720

# MediaPipe Tasks (current API) requires a downloaded model bundle — it is
# NOT bundled with the mediapipe package itself. Download once from:
#   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# and place it at this path (project root by default).
HAND_LANDMARKER_MODEL_PATH = 'hand_landmarker.task'

# Same 21-point hand skeleton connections the SEBT web app uses — needed
# because mp.solutions.drawing_utils (legacy API) is not available either.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

CSV_FIELDS = [
    'trial_number', 'timestamp',
    'P1_x', 'P1_y', 'P2_x', 'P2_y',
    'pixel_distance', 'known_distance_cm', 'pixel_scale_cm_per_pixel',
    'depth_P1', 'depth_P2', 'depth_difference',
]

# State machine states
STATE_WAIT_P1  = 'WAIT_P1'
STATE_WAIT_P2  = 'WAIT_P2'
STATE_RESULTS  = 'RESULTS'   # both points locked, depth sampled, showing results


# ══════════════════════════════════════════════════════════════════
# §1  CAMERA CAPTURE
# ══════════════════════════════════════════════════════════════════
def init_camera():
    """Opens the default webcam at a fixed resolution."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam (index 0). Check camera permissions/connection.')
    return cap


def read_frame(cap):
    """
    Reads one frame and flips it horizontally (mirror view).
    Every downstream pixel coordinate — hand landmarks AND the depth
    map — is computed on THIS flipped frame, so (x, y) always refers
    to the same physical location in both. Nothing re-flips later.
    """
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.flip(frame, 1)


# ══════════════════════════════════════════════════════════════════
# §2  HAND / INDEX-FINGER DETECTION
# ══════════════════════════════════════════════════════════════════
def init_hand_detector():
    """
    Sets up a single-hand detector using the MediaPipe Tasks API
    (mp.solutions.hands is not available in this MediaPipe build).

    Requires hand_landmarker.task to already be downloaded — see
    HAND_LANDMARKER_MODEL_PATH at the top of this file for the URL
    and expected location.
    """
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f'Hand Landmarker model not found at "{HAND_LANDMARKER_MODEL_PATH}".\n'
            'Download it from:\n'
            '  https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task\n'
            'and place it at that path (project root by default), or update '
            'HAND_LANDMARKER_MODEL_PATH to point at it.'
        )

    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,   # continuous webcam stream, not single images
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    return landmarker


def draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h):
    """
    Draws the 21-point hand skeleton. Replaces mp.solutions.drawing_utils,
    which does not exist in this MediaPipe build — same HAND_CONNECTIONS
    layout as the existing SEBT web app, just drawn with plain cv2 calls.
    """
    points = [_landmark_px(lm, frame_w, frame_h) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def _landmark_px(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def detect_pointing_finger(hand_landmarks, frame_w, frame_h):
    """
    Given one hand's list of 21 landmarks (Tasks API: a plain list of
    NormalizedLandmark objects — no `.landmark` wrapper like the old
    mp.solutions API had), decides whether the hand is doing the
    "index finger pointing, other fingers closed" gesture, and returns
    the index fingertip's pixel coordinates.

    Same rule the SEBT web app uses: a finger counts as "up" if its
    tip is noticeably farther from the wrist than its MCP joint is
    (tip_dist > mcp_dist * 1.6).

    Returns: (is_pointing: bool, tip_px: (x, y))
    """
    lm = hand_landmarks
    wrist = _landmark_px(lm[0], frame_w, frame_h)

    tips = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    mcps = [5, 9, 13, 17]    # their corresponding MCP joints

    finger_up = []
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_px = _landmark_px(lm[tip_idx], frame_w, frame_h)
        mcp_px = _landmark_px(lm[mcp_idx], frame_w, frame_h)
        finger_up.append(_dist(wrist, tip_px) > _dist(wrist, mcp_px) * 1.6)

    index_up, middle_up, ring_up, pinky_up = finger_up
    is_pointing = index_up and not (middle_up or ring_up or pinky_up)

    index_tip_px = _landmark_px(lm[8], frame_w, frame_h)
    return is_pointing, index_tip_px


# ══════════════════════════════════════════════════════════════════
# §3  POINT MARKING (state machine helper)
# ══════════════════════════════════════════════════════════════════
class TrialState:
    """Holds everything about the trial currently in progress."""
    def __init__(self):
        self.state = STATE_WAIT_P1
        self.p1 = None          # (x, y) once locked
        self.p2 = None          # (x, y) once locked
        self.last_mark_time = 0.0
        self.trial_number = 1
        # results, filled in once depth inference runs (§4/§5)
        self.pixel_distance = None
        self.pixel_scale = None
        self.depth1 = None
        self.depth2 = None
        self.depth_diff = None

    def reset_for_new_trial(self, keep_trial_number=True):
        """R key handler: clears the current points/results and starts over."""
        next_trial = self.trial_number + 1 if keep_trial_number else self.trial_number
        self.__init__()
        self.trial_number = next_trial


def try_mark_point(trial: TrialState, is_pointing: bool, tip_px):
    """
    Called every frame. If the gesture is being held and the cooldown
    has elapsed, locks the current point (P1 first, then P2) and
    advances the state machine. Returns True if a point was just locked.
    """
    now = time.time()
    if not is_pointing:
        return False
    if now - trial.last_mark_time < MARK_COOLDOWN_SEC:
        return False

    if trial.state == STATE_WAIT_P1:
        trial.p1 = tip_px
        trial.state = STATE_WAIT_P2
        trial.last_mark_time = now
        return True

    if trial.state == STATE_WAIT_P2:
        trial.p2 = tip_px
        trial.state = STATE_RESULTS
        trial.last_mark_time = now
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# §4  DEPTH ANYTHING V2 INFERENCE
# ══════════════════════════════════════════════════════════════════
def init_depth_model():
    """
    Loads the existing Depth Anything V2 VITS implementation/checkpoint.
    Assumes the standard official repo layout already present in
    depth_anything_v2/ (DepthAnythingV2 class in depth_anything_v2/dpt.py).
    """
    from depth_anything_v2.dpt import DepthAnythingV2  # reuse existing implementation

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f'Checkpoint not found at {CHECKPOINT_PATH}. '
            'Make sure you are running this from the Depth_SEBT/ project root.'
        )

    # CPU by default for this initial experiment; uses GPU automatically if present.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthAnythingV2(**model_configs[ENCODER])
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f'[depth] Loaded Depth Anything V2 ({ENCODER}) on device: {device}')
    return model, device


def run_depth_inference(model, frame_bgr, device):
    """
    Runs Depth Anything V2 ONCE on the given frame and returns a
    (H, W) numpy depth map at the frame's original resolution.

    This is only called once per trial (when P2 locks) — NOT every
    frame — to keep the CPU-based live preview responsive.
    """
    with torch.no_grad():
        depth_map = model.infer_image(frame_bgr, input_size=DEPTH_INPUT_SIZE)
    return depth_map  # relative depth, NOT metric — do not convert here


def sample_depth(depth_map, px):
    """Reads the depth map at pixel (x, y), clamped to valid bounds."""
    h, w = depth_map.shape[:2]
    x = int(np.clip(px[0], 0, w - 1))
    y = int(np.clip(px[1], 0, h - 1))
    return float(depth_map[y, x])


# ══════════════════════════════════════════════════════════════════
# §5  PIXEL DISTANCE / SCALE CALCULATION
# ══════════════════════════════════════════════════════════════════
def compute_pixel_distance_and_scale(p1, p2, known_distance_cm=KNOWN_DISTANCE_CM):
    """
    Pure geometry — no depth values are used here at all.
    pixelScale = known_distance_cm / pixelDistance  (cm per pixel)
    """
    pixel_distance = _dist(p1, p2)
    pixel_scale = known_distance_cm / pixel_distance if pixel_distance > 0 else 0.0
    return pixel_distance, pixel_scale


def finalize_trial_measurements(trial: TrialState, depth_map):
    """
    Called once, right after depth inference for this trial completes.
    Fills in trial.pixel_distance / pixel_scale / depth1 / depth2 / depth_diff.
    """
    trial.pixel_distance, trial.pixel_scale = compute_pixel_distance_and_scale(trial.p1, trial.p2)
    trial.depth1 = sample_depth(depth_map, trial.p1)
    trial.depth2 = sample_depth(depth_map, trial.p2)
    trial.depth_diff = abs(trial.depth2 - trial.depth1)


# ══════════════════════════════════════════════════════════════════
# §6  DATA RECORDING (CSV)
# ══════════════════════════════════════════════════════════════════
def init_csv(path=CSV_PATH):
    """Creates the CSV with a header row if it doesn't already exist."""
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def save_trial_to_csv(trial: TrialState, path=CSV_PATH):
    """Appends one completed trial as a new row."""
    row = {
        'trial_number':              trial.trial_number,
        'timestamp':                 datetime.now().isoformat(timespec='seconds'),
        'P1_x':                      trial.p1[0],
        'P1_y':                      trial.p1[1],
        'P2_x':                      trial.p2[0],
        'P2_y':                      trial.p2[1],
        'pixel_distance':            round(trial.pixel_distance, 4),
        'known_distance_cm':         KNOWN_DISTANCE_CM,
        'pixel_scale_cm_per_pixel':  round(trial.pixel_scale, 6),
        'depth_P1':                  round(trial.depth1, 6),
        'depth_P2':                  round(trial.depth2, 6),
        'depth_difference':          round(trial.depth_diff, 6),
    }
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
    print(f'[csv] Trial {trial.trial_number} saved -> {path}')


# ══════════════════════════════════════════════════════════════════
# §7  UI / DRAWING
# ══════════════════════════════════════════════════════════════════
def draw_ui(frame, trial: TrialState, index_tip_px, is_pointing, saved_this_trial):
    """Draws all overlays: finger pointer, locked points, and the text panel."""
    # Live index-finger pointer
    if index_tip_px is not None:
        color = (0, 255, 0) if is_pointing else (0, 0, 255)
        cv2.circle(frame, index_tip_px, 10, color, -1)
        cv2.circle(frame, index_tip_px, 18, color, 2)

    # Locked point markers
    if trial.p1 is not None:
        cv2.circle(frame, trial.p1, 12, (255, 255, 0), -1)
        cv2.putText(frame, 'P1', (trial.p1[0] + 16, trial.p1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if trial.p2 is not None:
        cv2.circle(frame, trial.p2, 12, (255, 128, 0), -1)
        cv2.putText(frame, 'P2', (trial.p2[0] + 16, trial.p2[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    if trial.p1 is not None and trial.p2 is not None:
        cv2.line(frame, trial.p1, trial.p2, (0, 255, 255), 2)

    # Instruction text
    instructions = {
        STATE_WAIT_P1: 'Point at RULER POINT 1 (index finger only) and hold',
        STATE_WAIT_P2: 'P1 locked. Now point at RULER POINT 2 and hold',
        STATE_RESULTS: 'Trial complete — press R for new trial, Q to quit',
    }
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, instructions[trial.state], (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Results / info panel (bottom-left)
    panel_y = frame.shape[0] - 150
    cv2.rectangle(frame, (0, panel_y - 10), (420, frame.shape[0]), (20, 20, 20), -1)
    lines = [f'Trial #{trial.trial_number}']
    if trial.pixel_distance is not None:
        lines.append(f'Pixel distance: {trial.pixel_distance:.1f} px')
        lines.append(f'Known distance: {KNOWN_DISTANCE_CM:.1f} cm')
        lines.append(f'Pixel scale: {trial.pixel_scale:.4f} cm/px')
        lines.append(f'Depth P1: {trial.depth1:.4f}   Depth P2: {trial.depth2:.4f}')
        lines.append(f'Depth diff: {trial.depth_diff:.4f}')
        if saved_this_trial:
            lines.append('Saved to calibration_results.csv')
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, panel_y + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.putText(frame, 'R = reset   Q = quit', (frame.shape[1] - 220, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


# ══════════════════════════════════════════════════════════════════
# §8  MAIN LOOP
# ══════════════════════════════════════════════════════════════════
def main():
    """
    State machine per trial:

        WAIT_P1 --(gesture held)--> WAIT_P2 --(gesture held)--> RESULTS
                                                                    |
                                          (depth inference runs HERE, once)
                                                                    |
                                                          trial saved to CSV
                                                                    |
                                                    R -> back to WAIT_P1 (new trial)

    Depth Anything only runs at the WAIT_P2 -> RESULTS transition,
    i.e. once per trial — never in the per-frame preview loop.
    """
    print('Loading Depth Anything V2 VITS... (this happens once, at startup)')
    depth_model, device = init_depth_model()

    init_csv()

    cap = init_camera()
    landmarker = init_hand_detector()

    trial = TrialState()
    saved_this_trial = False
    last_timestamp_ms = 0   # Tasks API VIDEO mode needs strictly increasing timestamps

    window_name = 'Ruler <-> Depth Calibration Experiment'
    print('Camera started. Press Q at any time to quit, R to reset a trial.')

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                print('[camera] Failed to read frame — stopping.')
                break

            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            is_pointing = False
            index_tip_px = None

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h)
                is_pointing, index_tip_px = detect_pointing_finger(hand_landmarks, frame_w, frame_h)

                # Only try to mark while we're still waiting for a point
                if trial.state in (STATE_WAIT_P1, STATE_WAIT_P2):
                    just_locked = try_mark_point(trial, is_pointing, index_tip_px)

                    # The moment P2 locks, run Depth Anything ONCE on this
                    # exact frame and compute all measurements for the trial.
                    if just_locked and trial.state == STATE_RESULTS:
                        print(f'[depth] Running inference for trial {trial.trial_number}...')
                        depth_map = run_depth_inference(depth_model, frame, device)
                        finalize_trial_measurements(trial, depth_map)
                        save_trial_to_csv(trial)
                        saved_this_trial = True

            draw_ui(frame, trial, index_tip_px, is_pointing, saved_this_trial)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print('Quitting.')
                break
            if key == ord('r') or key == ord('R'):
                trial.reset_for_new_trial()
                saved_this_trial = False
                print(f'[trial] Reset — starting trial {trial.trial_number}.')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == '__main__':
    main()