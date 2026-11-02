import React from 'react';
import { Download, FileSpreadsheet, Film, FileText, CheckCircle2 } from 'lucide-react';

export default function DownloadSection({ jobId, videoFilename, summary = {}, observations = {} }) {
  if (!jobId) return null;

  const baseFilename = videoFilename?.replace(/\.[^/.]+$/, '') || 'sebt';
  const videoUrl = `/api/sebt/jobs/${jobId}/files/video`;
  const resultsCsvUrl = `/api/sebt/jobs/${jobId}/files/results_csv`;
  const summaryCsvUrl = `/api/sebt/jobs/${jobId}/files/summary_csv`;

  const leftStance = summary.composite_scores?.['LEFT_STANCE'] || {};
  const rightStance = summary.composite_scores?.['RIGHT_STANCE'] || {};

  return (
    <section className="dashboard-card" id="download-results-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Downloads & Reports</span>
          <h2>Download Analysis Artifacts</h2>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Job ID: <code style={{ fontFamily: 'var(--font-mono)' }}>{jobId}</code>
        </div>
      </div>

      {/* Artifact Download Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        {/* Card 1: Processed MP4 */}
        <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px', background: '#ffffff', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <div style={{ width: 36, height: 36, borderRadius: '8px', background: 'var(--primary-50)', color: 'var(--primary-600)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Film size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-main)' }}>Depth-Visualized Video</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>MP4 format with HUD overlays</div>
              </div>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.4 }}>
              Includes real-time SEBT direction badges, ankle displacement vectors, and MediaPipe pose skeleton.
            </p>
          </div>
          <a
            href={videoUrl}
            download={`${baseFilename}_depth_visualized.mp4`}
            className="btn btn-primary"
            style={{ width: '100%' }}
          >
            <Download size={15} />
            Download Visualized MP4
          </a>
        </div>

        {/* Card 2: Detailed CSV */}
        <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px', background: '#ffffff', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <div style={{ width: 36, height: 36, borderRadius: '8px', background: 'var(--success-bg)', color: 'var(--success-text)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FileSpreadsheet size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-main)' }}>Trial-by-Trial Results CSV</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>36-column detailed metrics</div>
              </div>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.4 }}>
              Frame timings, pixel coordinates, 3D metric camera space coordinates, reach distances, and trial scores.
            </p>
          </div>
          <a
            href={resultsCsvUrl}
            download={`${baseFilename}_depth_results.csv`}
            className="btn btn-secondary"
            style={{ width: '100%' }}
          >
            <Download size={15} />
            Download Detailed CSV
          </a>
        </div>

        {/* Card 3: Summary CSV */}
        <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px', background: '#ffffff', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <div style={{ width: 36, height: 36, borderRadius: '8px', background: 'var(--primary-50)', color: 'var(--primary-600)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FileText size={20} />
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-main)' }}>SEBT Summary Report CSV</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Structured 4-section summary</div>
              </div>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '14px', lineHeight: 1.4 }}>
              Leg lengths, stance composites, selected maximum trials, and methodological limitation disclosures.
            </p>
          </div>
          <a
            href={summaryCsvUrl}
            download={`${baseFilename}_sebt_summary.csv`}
            className="btn btn-secondary"
            style={{ width: '100%' }}
          >
            <Download size={15} />
            Download Summary CSV
          </a>
        </div>
      </div>

      {/* Final Executive Summary Panel */}
      <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px 20px' }}>
        <h3 style={{ fontSize: '0.96rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CheckCircle2 size={18} color="var(--success-dot)" />
          Executive Test Summary
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', fontSize: '0.88rem' }}>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Left-Stance Composite:</span>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-main)', marginTop: '2px' }}>
              {leftStance.manual_composite_pct ? `${leftStance.manual_composite_pct.toFixed(2)}%` : 'Unavailable'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {leftStance.denominator ? `${leftStance.denominator}/8 distinct directions` : '--'}
            </div>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Right-Stance Composite:</span>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-main)', marginTop: '2px' }}>
              {rightStance.manual_composite_pct ? `${rightStance.manual_composite_pct.toFixed(2)}%` : 'Unavailable'}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {rightStance.denominator ? `${rightStance.denominator}/8 distinct directions` : '--'}
            </div>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Total Movements Logged:</span>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-main)', marginTop: '2px' }}>
              {observations.total_movements || 0} trials ({observations.valid_movements || 0} valid, {observations.excluded_movements || 0} excluded)
            </div>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Evaluated Directions:</span>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--text-main)', marginTop: '2px' }}>
              {observations.detected_directions ? `${observations.detected_directions.length}/8 Directions` : '0/8'}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
