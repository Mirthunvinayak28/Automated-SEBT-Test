"""
backend/main.py
══════════════════════════════════════════════════════════════════════
FastAPI Application for Automated SEBT Test
Provides endpoints for video upload, analysis status polling,
streaming depth visualization video, downloading CSVs, and sample datasets.
══════════════════════════════════════════════════════════════════════
"""
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.job_manager import job_manager
from backend.video_service import stream_video_file

app = FastAPI(
    title="Automated SEBT Test API",
    description="Video-based experimental assessment of dynamic balance using the Star Excursion Balance Test (SEBT).",
    version="1.0.0"
)

# Enable CORS for local Vite dev server and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB limit

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Automated SEBT Test API",
        "version": "1.0.0"
    }

@app.get("/api/sebt/samples")
def list_samples():
    """Returns available sample datasets (disabled to ensure current analysis logic is used)."""
    return []

@app.post("/api/sebt/load-sample/{sample_id}")
def load_sample(sample_id: str):
    """Sample datasets have been removed as they used obsolete precomputed logic."""
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Sample datasets have been removed. Please upload and analyze a video using the current pipeline."
    )

@app.post("/api/sebt/analyze")
async def analyze_sebt_video(
    video: UploadFile = File(..., description="SEBT video recording in MP4 format"),
    left_leg: float = Form(..., description="Manual left clinical leg length in cm"),
    right_leg: float = Form(..., description="Manual right clinical leg length in cm"),
):
    """
    Receives an SEBT video recording and clinical leg measurements,
    validates input criteria, creates an isolated processing job, and launches the pipeline.
    """
    # 1. Validate leg length values
    if left_leg <= 0 or right_leg <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Clinical leg lengths must be positive numbers greater than 0 cm."
        )

    if left_leg < 35.0 or left_leg > 140.0 or right_leg < 35.0 or right_leg > 140.0:
        # Plausible human adult/adolescent leg length warning
        print(f"[warning] Leg length out of typical clinical range: L={left_leg}, R={right_leg}")

    # 2. Validate video format
    filename = video.filename or "video.mp4"
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video format. Please upload an MP4 file."
        )

    # 3. Read video contents and enforce size limit
    video_bytes = await video.read()
    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty."
        )

    if len(video_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file exceeds size limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 4. Dispatch job
    job_id = job_manager.create_job(
        video_bytes=video_bytes,
        filename=filename,
        left_leg=left_leg,
        right_leg=right_leg
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video successfully uploaded. Analysis job initialized."
    }

@app.get("/api/sebt/jobs/{job_id}")
def get_job_status(job_id: str):
    """Returns the current execution status, progress percentage, and stage message."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "video_filename": job.get("video_filename"),
        "left_leg": job.get("left_leg"),
        "right_leg": job.get("right_leg"),
        "created_at": job.get("created_at"),
        "error": job.get("error")
    }

@app.get("/api/sebt/jobs/{job_id}/results")
def get_job_results(job_id: str):
    """Returns parsed measurements from the generated CSV files."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if job["status"] == "processing" or job["status"] == "queued":
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": job["status"],
                "message": "Analysis is currently running. Please poll until completed."
            }
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=job.get("error") or "Analysis pipeline failed."
        )

    return {
        "job_id": job_id,
        "status": job["status"],
        "video_filename": job.get("video_filename"),
        "left_leg": job.get("left_leg"),
        "right_leg": job.get("right_leg"),
        "data": job.get("results")
    }

@app.get("/api/sebt/jobs/{job_id}/video")
def get_processed_video(job_id: str, request: Request):
    """Streams the depth-visualized MP4 with HTTP 206 Partial Content / Range support."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    video_path = job.get("output_video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video not available yet.")

    return stream_video_file(video_path, request)

@app.get("/api/sebt/jobs/{job_id}/files/{file_type}")
def download_artifact(job_id: str, file_type: str):
    """Allows downloading the processed MP4, detailed CSV, or summary CSV."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    file_mapping = {
        "video": (job.get("output_video_path"), f"{job.get('video_filename', 'sebt')}_depth_visualized.mp4", "video/mp4"),
        "results_csv": (job.get("results_csv_path"), f"{job.get('video_filename', 'sebt')}_depth_results.csv", "text/csv"),
        "summary_csv": (job.get("summary_csv_path"), f"{job.get('video_filename', 'sebt')}_sebt_summary.csv", "text/csv"),
    }

    if file_type not in file_mapping:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file type '{file_type}'.")

    file_path, download_name, media_type = file_mapping[file_type]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requested artifact file is not available.")

    return FileResponse(
        path=file_path,
        filename=download_name,
        media_type=media_type
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
