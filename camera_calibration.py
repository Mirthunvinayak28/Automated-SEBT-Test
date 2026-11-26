
import cv2
import numpy as np
import os
import glob

# =====================================================
# CAMERA CALIBRATION - AUTOMATED SEBT PROJECT
# =====================================================

IMAGE_FOLDER = r"calibration_images"
OUTPUT_FOLDER = r"calibration_output"

CHECKERBOARD = (7, 7)
SQUARE_SIZE_MM = 20.6

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DETECTION_FOLDER = os.path.join(
    OUTPUT_FOLDER, "detections"
)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# =====================================================
# 1. GENERATE 3D OBJECT POINTS
# =====================================================

objp = np.zeros((49, 3), np.float32)

objp[:, :2] = np.mgrid[
    0:7, 0:7
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints = []

accepted_images = []
rejected_images = []
per_view_errors = []

image_size = None

# =====================================================
# 2. LOAD IMAGES
# =====================================================

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

image_paths = []

for ext in extensions:
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_FOLDER, ext))
    )

image_paths = sorted(set(image_paths))

print("\n==========================================")
print("       CAMERA INTRINSIC CALIBRATION")
print("==========================================")

print("Images found:", len(image_paths))
print("Pattern:", CHECKERBOARD)
print("Square size:", SQUARE_SIZE_MM, "mm")

if len(image_paths) < 10:
    raise RuntimeError("Not enough calibration images")

# =====================================================
# 3. DETECT CHECKERBOARD
# =====================================================

for index, path in enumerate(image_paths, start=1):

    filename = os.path.basename(path)

    print(f"\nProcessing {index}: {filename}")

    image = cv2.imread(path)

    if image is None:
        rejected_images.append(filename)
        print("REJECTED: Cannot read image")
        continue

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size

    if current_size != image_size:
        rejected_images.append(filename)
        print("REJECTED: Resolution mismatch")
        continue

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            None
        )

    if not found:
        rejected_images.append(filename)
        print("REJECTED: 7x7 corners not detected")
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CRITERIA
    )

    objpoints.append(objp.copy())
    imgpoints.append(corners)

    accepted_images.append(filename)

    display = image.copy()

    cv2.drawChessboardCorners(
        display,
        CHECKERBOARD,
        corners,
        True
    )

    cv2.imwrite(
        os.path.join(DETECTION_FOLDER, filename),
        display
    )

    print("ACCEPTED: 49 corners detected")

# =====================================================
# 4. DETECTION SUMMARY
# =====================================================

print("\n==========================================")
print("          DETECTION SUMMARY")
print("==========================================")

print("Accepted:", len(accepted_images))
print("Rejected:", len(rejected_images))

if rejected_images:
    print("Rejected files:", rejected_images)

if len(accepted_images) < 10:
    raise RuntimeError(
        "Insufficient valid calibration images"
    )

# =====================================================
# 5. CAMERA CALIBRATION
# =====================================================

print("\nCalculating camera intrinsics...")

rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
)


# =====================================================
# 6. REPROJECTION ERROR
# =====================================================

total_squared_error = 0.0
total_points = 0

for i in range(len(objpoints)):

    projected, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    observed_points = np.asarray(
        imgpoints[i], dtype=np.float64
    ).reshape(-1, 2)

    projected_points = np.asarray(
        projected, dtype=np.float64
    ).reshape(-1, 2)

    differences = observed_points - projected_points

    point_errors = np.linalg.norm(
        differences, axis=1
    )

    per_view_error = np.sqrt(
        np.mean(point_errors ** 2)
    )

    per_view_errors.append(per_view_error)

    total_squared_error += np.sum(
        point_errors ** 2
    )

    total_points += len(observed_points)

reprojection_error = np.sqrt(
    total_squared_error / total_points
)

# =====================================================
# 7. DISPLAY RESULTS
# =====================================================

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print("\n==========================================")
print("          CALIBRATION RESULTS")
print("==========================================")

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(dist_coeffs)

print(f"\nfx = {fx:.4f}")
print(f"fy = {fy:.4f}")
print(f"cx = {cx:.4f}")
print(f"cy = {cy:.4f}")

print("\nImage resolution:", image_size)

print(f"\nCalibration RMS: {rms:.6f} pixels")

print(
    f"Overall Reprojection Error: "
    f"{reprojection_error:.6f} pixels"
)

print("\nIndividual image errors:")

for filename, error in zip(
    accepted_images, per_view_errors
):
    print(f"{filename}: {error:.6f} pixels")

# =====================================================
# 8. SAVE RESULTS
# =====================================================

save_path = os.path.join(
    OUTPUT_FOLDER,
    "camera_intrinsics.npz"
)

np.savez(
    save_path,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
    image_width=image_size[0],
    image_height=image_size[1],
    reprojection_error_px=reprojection_error,
    calibration_rms=rms,
    per_view_errors_px=np.array(per_view_errors),
    pattern_size=np.array(CHECKERBOARD),
    square_size_mm=SQUARE_SIZE_MM,
    accepted_images=np.array(accepted_images),
    rejected_images=np.array(rejected_images)
)

print("\n==========================================")
print("CALIBRATION COMPLETED")
print("Saved:", save_path)
print("==========================================")
