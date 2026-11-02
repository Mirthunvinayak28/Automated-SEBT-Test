"""
backend/video_service.py
══════════════════════════════════════════════════════════════════════
Video streaming service supporting HTTP 206 Range Requests (seeking)
and automated H.264 transcoding for maximum HTML5 browser playback.
══════════════════════════════════════════════════════════════════════
"""
import os
import subprocess
from typing import Generator
from fastapi import Request, HTTPException, status
from fastapi.responses import StreamingResponse

def get_ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def optimize_video_for_web(input_video_path: str, output_video_path: str = None) -> bool:
    """
    Transcodes an OpenCV-produced video to web-standard H.264 (yuv420p) with faststart metadata.
    This ensures smooth playback, seeking, and scrubbing across all modern web browsers.
    """
    if not os.path.exists(input_video_path):
        return False

    ffmpeg = get_ffmpeg_path()
    if output_video_path is None:
        target_path = input_video_path + ".web.mp4"
    else:
        target_path = output_video_path

    cmd = [
        ffmpeg,
        "-y",
        "-i", input_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        target_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if res.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            if output_video_path is None:
                # Replace original with optimized web version
                os.replace(target_path, input_video_path)
            return True
        return False
    except Exception as e:
        print(f"[video_service] Web video optimization warning: {e}")
        if os.path.exists(target_path) and output_video_path is None:
            try: os.remove(target_path)
            except OSError: pass
        return False

def stream_video_file(file_path: str, request: Request):
    """
    Streams a video file with full HTTP 206 Partial Content (Range request) support.
    Allows browser video controls to seek forward/backward instantly.
    """
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range")

    if range_header:
        # Format: "bytes=start-end"
        try:
            byte_range = range_header.replace("bytes=", "").split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
            if start >= file_size:
                raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
            end = min(end, file_size - 1)
            content_length = (end - start) + 1

            def iter_file() -> Generator[bytes, None, None]:
                with open(file_path, "rb") as video:
                    video.seek(start)
                    bytes_left = content_length
                    chunk_size = 1024 * 512 # 512KB chunks
                    while bytes_left > 0:
                        chunk = video.read(min(bytes_left, chunk_size))
                        if not chunk:
                            break
                        bytes_left -= len(chunk)
                        yield chunk

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
                "Cache-Control": "no-cache",
            }
            return StreamingResponse(iter_file(), status_code=status.HTTP_206_PARTIAL_CONTENT, headers=headers)
        except (ValueError, IndexError):
            pass

    # No Range header: full video
    def iter_full_file() -> Generator[bytes, None, None]:
        with open(file_path, "rb") as video:
            chunk_size = 1024 * 512
            while chunk := video.read(chunk_size):
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(iter_full_file(), status_code=status.HTTP_200_OK, headers=headers)
