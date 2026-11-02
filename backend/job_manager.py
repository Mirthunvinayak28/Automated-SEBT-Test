"""
backend/job_manager.py
══════════════════════════════════════════════════════════════════════
Manages SEBT video analysis background jobs:
  - Isolated working directories
  - Non-blocking asynchronous execution via subprocess
  - Real-time progress monitoring and log retention
  - Result extraction and video optimization
══════════════════════════════════════════════════════════════════════
"""
import os
import sys
import time
import uuid
import threading
import subprocess
from typing import Dict, Any, Optional

from backend.parser import get_complete_sebt_results
from backend.video_service import optimize_video_for_web

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PYTHON_EXE = os.path.join(BASE_DIR, "venv311", "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

os.makedirs(JOBS_DIR, exist_ok=True)

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._init_sample_jobs()

    def _init_sample_jobs(self):
        """Registers built-in sample datasets from existing results for instant demonstration."""
        sample_configs = [
            {
                'id': 'sample-test6',
                'title': 'Sample Dataset 1 (Test 6 — 12 movements, 7 directions)',
                'base_name': 'test6',
                'left_leg': 90.0,
                'right_leg': 91.0,
                'filename': 'test6.mp4'
            },
            {
                'id': 'sample-test5',
                'title': 'Sample Dataset 2 (Test 5 — 6 movements, quick trial)',
                'base_name': 'test5',
                'left_leg': 88.0,
                'right_leg': 88.0,
                'filename': 'test5.mp4'
            }
        ]

        for cfg in sample_configs:
            base = cfg['base_name']
            video_path = os.path.join(RESULTS_DIR, f"{base}_depth_visualized.mp4")
            csv_results = os.path.join(RESULTS_DIR, f"{base}_depth_results.csv")
            csv_summary = os.path.join(RESULTS_DIR, f"{base}_sebt_summary.csv")

            if os.path.exists(csv_results) and os.path.exists(csv_summary):
                parsed = get_complete_sebt_results(csv_results, csv_summary)
                self.jobs[cfg['id']] = {
                    'id': cfg['id'],
                    'is_sample': True,
                    'title': cfg['title'],
                    'video_filename': cfg['filename'],
                    'left_leg': cfg['left_leg'],
                    'right_leg': cfg['right_leg'],
                    'status': 'completed',
                    'created_at': time.time(),
                    'progress': {
                        'stage': 'Completed (Sample Dataset)',
                        'percent': 100.0,
                        'frame': 100,
                        'total_frames': 100,
                        'message': 'Sample analysis ready for exploration.'
                    },
                    'output_video_path': video_path,
                    'results_csv_path': csv_results,
                    'summary_csv_path': csv_summary,
                    'results': parsed,
                    'logs': ['Sample dataset preloaded.'],
                    'error': None
                }

    def list_samples(self):
        return [
            {
                'id': j['id'],
                'title': j.get('title'),
                'video_filename': j['video_filename'],
                'left_leg': j['left_leg'],
                'right_leg': j['right_leg']
            }
            for j in self.jobs.values() if j.get('is_sample')
        ]

    def create_job(self, video_bytes: bytes, filename: str, left_leg: float, right_leg: float) -> str:
        job_id = f"job-{uuid.uuid4().hex[:10]}"
        job_dir = os.path.join(JOBS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        input_video_path = os.path.join(job_dir, "input.mp4")
        with open(input_video_path, "wb") as f:
            f.write(video_bytes)

        job_info = {
            'id': job_id,
            'is_sample': False,
            'title': filename,
            'video_filename': filename,
            'left_leg': left_leg,
            'right_leg': right_leg,
            'status': 'queued',
            'created_at': time.time(),
            'job_dir': job_dir,
            'input_video_path': input_video_path,
            'progress': {
                'stage': 'Video uploaded successfully',
                'percent': 0.0,
                'frame': 0,
                'total_frames': 0,
                'message': 'Job queued for processing.'
            },
            'output_video_path': None,
            'results_csv_path': None,
            'summary_csv_path': None,
            'results': None,
            'logs': [f"Job {job_id} created with input video {filename} ({len(video_bytes)} bytes)"],
            'error': None
        }

        with self._lock:
            self.jobs[job_id] = job_info

        # Launch processing in a background worker thread
        t = threading.Thread(target=self._run_job_process, args=(job_id,), daemon=True)
        t.start()
        return job_id

    def _run_job_process(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return

        job_dir = job['job_dir']
        input_video = job['input_video_path']
        left_leg = job['left_leg']
        right_leg = job['right_leg']

        job['status'] = 'processing'
        job['progress']['stage'] = 'Analysis started'
        job['progress']['percent'] = 5.0
        job['progress']['message'] = 'Initializing pose and metric depth models...'

        cmd = [
            PYTHON_EXE,
            "sebt_depth_video.py",
            "-i", input_video,
            "--left-leg", str(left_leg),
            "--right-leg", str(right_leg),
            "--output-dir", job_dir,
            "--headless"
        ]

        print(f"[job_manager] Running pipeline: {' '.join(cmd)}")
        job['logs'].append(f"Executing: {' '.join(cmd)}")

        try:
            # Execute with unbuffered stdout
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            for line in process.stdout:
                line_str = line.strip()
                if not line_str:
                    continue

                # Parse frame-level progress
                if line_str.startswith("PROGRESS:frame:"):
                    parts = line_str.split(":")
                    if len(parts) >= 4:
                        try:
                            f_idx = int(parts[2])
                            f_total = int(parts[3])
                            pct = min(95.0, round(5.0 + (f_idx / f_total) * 85.0, 1)) if f_total > 0 else 10.0
                            job['progress']['percent'] = pct
                            job['progress']['frame'] = f_idx
                            job['progress']['total_frames'] = f_total
                            job['progress']['stage'] = f"Processing frame {f_idx} of {f_total}"
                            job['progress']['message'] = f"Tracking ankle displacement and estimating depth ({f_idx}/{f_total} frames)"
                        except ValueError:
                            pass
                elif line_str.startswith("PROGRESS:status:"):
                    stage_name = line_str.replace("PROGRESS:status:", "").strip()
                    job['progress']['stage'] = stage_name
                    if "Generating" in stage_name:
                        job['progress']['percent'] = 92.0
                        job['progress']['message'] = stage_name
                elif line_str.startswith("MOVEMENT START"):
                    job['logs'].append(line_str)
                    job['progress']['message'] = line_str
                elif line_str.startswith("MOVEMENT ") and "SUMMARY" in line_str:
                    job['logs'].append(line_str)
                else:
                    if len(job['logs']) < 300:
                        job['logs'].append(line_str)

            process.wait()
            ret_code = process.returncode

            if ret_code != 0:
                job['status'] = 'failed'
                job['error'] = f"Pipeline terminated with non-zero exit code ({ret_code})."
                job['progress']['stage'] = 'Analysis failed'
                job['progress']['message'] = 'An error occurred during video analysis.'
                print(f"[job_manager] Job {job_id} failed with exit code {ret_code}")
                return

            # Locate generated output files
            out_video = os.path.join(job_dir, "input_depth_visualized.mp4")
            csv_results = os.path.join(job_dir, "input_depth_results.csv")
            csv_summary = os.path.join(job_dir, "input_sebt_summary.csv")

            if not os.path.exists(csv_results) or not os.path.exists(csv_summary):
                job['status'] = 'failed'
                job['error'] = "Pipeline completed but expected CSV results were not generated."
                job['progress']['stage'] = 'Analysis failed'
                return

            job['output_video_path'] = out_video
            job['results_csv_path'] = csv_results
            job['summary_csv_path'] = csv_summary

            # Web optimization: Transcode OpenCV video to browser-playable H.264
            if os.path.exists(out_video):
                job['progress']['stage'] = 'Optimizing depth visualization for web playback...'
                optimize_video_for_web(out_video)

            # Parse results
            job['progress']['stage'] = 'Preparing final results...'
            job['results'] = get_complete_sebt_results(csv_results, csv_summary)

            job['status'] = 'completed'
            job['progress']['percent'] = 100.0
            job['progress']['stage'] = 'Analysis complete'
            job['progress']['message'] = 'SEBT analysis finished successfully.'
            print(f"[job_manager] Job {job_id} completed successfully.")

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = f"Unexpected backend execution error: {str(e)}"
            job['progress']['stage'] = 'Analysis failed'
            print(f"[job_manager] Error executing job {job_id}: {e}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

job_manager = JobManager()
