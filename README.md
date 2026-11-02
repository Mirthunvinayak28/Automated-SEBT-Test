
# Automated SEBT Test

An automated video-based Star Excursion Balance Test (SEBT) system using computer vision and monocular depth estimation.

## Technologies Used

### Backend
- Python
- FastAPI
- OpenCV
- MediaPipe
- Depth Anything V2 Metric

### Frontend
- React
- JavaScript
- Vite
- CSS

## Project Structure

```text
Automated-SEBT-Test/
├── backend/
├── frontend/
├── depth_anything_v2/
├── metric_depth/
├── checkpoints/
├── sebt_video_depth_test.py
├── sebt_depth_video.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Checkpoints

Download the required Depth Anything V2 checkpoint files and place them inside `checkpoints/`.

- depth_anything_v2_metric_hypersim_vits.pth
- depth_anything_v2_vits.pth

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Automated-SEBT-Test.git
cd Automated-SEBT-Test
```

### 2. Install Python dependencies

```bash
python -m venv venv
pip install -r requirements.txt
```

### 3. Start the backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

### 4. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

## Experimental Scope

This project is an experimental computer-vision-based SEBT assessment system. Its measurements are subject to depth-estimation and camera-calibration limitations and should not be interpreted as clinically validated measurements.
