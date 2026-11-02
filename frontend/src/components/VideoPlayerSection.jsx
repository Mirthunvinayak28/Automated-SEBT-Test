import React, { useRef } from 'react';
import { Download, Film, Eye, Sparkles } from 'lucide-react';

export default function VideoPlayerSection({ jobId, videoFilename }) {
  const videoRef = useRef(null);

  if (!jobId) return null;

  const videoStreamUrl = `/api/sebt/jobs/${jobId}/video`;
  const downloadUrl = `/api/sebt/jobs/${jobId}/files/video`;

  return (
    <section className="dashboard-card" id="processed-video-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Processed Video</span>
          <h2>Processed Video — Depth Visualization</h2>
        </div>
        <a
          href={downloadUrl}
          download={`${videoFilename || 'sebt'}_depth_visualized.mp4`}
          className="btn btn-sm btn-primary"
        >
          <Download size={14} />
          Download Processed MP4
        </a>
      </div>

      <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '12px 16px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Eye size={18} color="var(--primary-600)" />
        <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
          <strong>On-Screen Visual Overlays:</strong> The processed video embeds the prominent high-contrast SEBT Direction HUD overlay banner, active reaching vs stance ankle tracking markers, 3D displacement vectors, and MediaPipe full-body pose skeleton directly into every frame.
        </span>
      </div>

      <div style={{ background: '#0a0f1d', borderRadius: 'var(--radius-lg)', overflow: 'hidden', boxShadow: 'var(--shadow-md)', display: 'flex', justifyContent: 'center' }}>
        <video
          ref={videoRef}
          src={videoStreamUrl}
          controls
          playsInline
          preload="metadata"
          style={{ width: '100%', maxHeight: '560px', objectFit: 'contain' }}
        >
          Your browser does not support HTML5 video playback. Please use the download link above.
        </video>
      </div>

      <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        <span>Video Source: Real generated <code>*_depth_visualized.mp4</code></span>
        <span>Streaming with HTTP 206 Range Request Seek Support</span>
      </div>
    </section>
  );
}
