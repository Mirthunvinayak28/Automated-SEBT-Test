"""
sebt_depth_video.py
══════════════════════════════════════════════════════════════════════
Automated SEBT Video Depth Processing Pipeline Entrypoint
Wraps sebt_video_depth_test.py to provide backward-compatible CLI execution.
══════════════════════════════════════════════════════════════════════
"""
import sys
import runpy

if __name__ == '__main__':
    # Forward execution to sebt_video_depth_test.py
    runpy.run_module('sebt_video_depth_test', run_name='__main__')
