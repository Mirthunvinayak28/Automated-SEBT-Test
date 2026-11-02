import React, { useState } from 'react';
import { Loader2, CheckCircle2, AlertCircle, Terminal, ChevronDown, ChevronUp } from 'lucide-react';

const PIPELINE_STAGES = [
  { id: 'upload', label: 'Video Uploaded' },
  { id: 'start', label: 'Analysis Initialized' },
  { id: 'frames', label: 'Processing Frames & Pose Tracking' },
  { id: 'reaches', label: 'Detecting & Measuring Reaches' },
  { id: 'csv', label: 'Generating Measurement CSV' },
  { id: 'summary', label: 'Generating SEBT Summary' },
  { id: 'final', label: 'Finalizing Results' }
];

export default function ProcessingStatus({ statusData, error }) {
  const [showLogs, setShowLogs] = useState(false);

  if (!statusData && !error) return null;

  const progress = statusData?.progress || { percent: 0, stage: 'Initializing...', message: '' };
  const currentPercent = progress.percent || 0;
  const stage = progress.stage || 'Processing';
  const isFailed = statusData?.status === 'failed' || !!error;
  const isCompleted = statusData?.status === 'completed';

  return (
    <section className="dashboard-card" id="processing-status-section" style={{ borderLeft: isFailed ? '4px solid var(--danger-dot)' : isCompleted ? '4px solid var(--success-dot)' : '4px solid var(--primary-600)' }}>
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Step 4</span>
          <h2>Processing Pipeline Status</h2>
        </div>
        <div>
          {isCompleted ? (
            <span className="badge badge-success">
              <CheckCircle2 size={12} />
              Analysis Completed
            </span>
          ) : isFailed ? (
            <span className="badge badge-danger">
              <AlertCircle size={12} />
              Analysis Failed
            </span>
          ) : (
            <span className="badge badge-info">
              <Loader2 size={12} className="spin-animation" />
              Processing
            </span>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-container">
        <div className="progress-header">
          <span className="progress-stage-title">
            {stage}
          </span>
          <span className="progress-percent-text">
            {currentPercent.toFixed(1)}%
          </span>
        </div>
        <div className="progress-bar-bg">
          <div
            className="progress-bar-fill"
            style={{
              width: `${Math.min(100, Math.max(0, currentPercent))}%`,
              background: isFailed ? 'var(--danger-dot)' : isCompleted ? 'var(--success-dot)' : undefined
            }}
          />
        </div>
        {progress.message && (
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '6px' }}>
            {progress.message}
          </div>
        )}
      </div>

      {/* Pipeline Milestone Stepper */}
      <div className="pipeline-steps">
        {PIPELINE_STAGES.map((s, idx) => {
          let stepClass = '';
          const stagePct = (idx + 1) * (100 / PIPELINE_STAGES.length);
          if (isCompleted || currentPercent >= stagePct) {
            stepClass = 'is-done';
          } else if (currentPercent >= stagePct - 15) {
            stepClass = 'is-active';
          }

          return (
            <div key={s.id} className={`step-item ${stepClass}`}>
              {stepClass === 'is-done' ? (
                <CheckCircle2 size={14} color="var(--success-dot)" />
              ) : stepClass === 'is-active' ? (
                <Loader2 size={14} color="var(--primary-600)" className="spin-animation" />
              ) : (
                <div style={{ width: 14, height: 14, borderRadius: '50%', border: '1px solid var(--border-strong)' }} />
              )}
              <span>{s.label}</span>
            </div>
          );
        })}
      </div>

      {/* Failure Message */}
      {isFailed && (
        <div style={{ marginTop: '18px', padding: '14px 18px', background: 'var(--danger-bg)', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-md)', color: 'var(--danger-text)', display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
          <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Processing Error:</strong> {error || statusData?.error || 'The video processing pipeline encountered an error. Please verify full-body and floor visibility and try again.'}
          </div>
        </div>
      )}

      {/* Internal Log Drawer */}
      {statusData?.logs && statusData.logs.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <button
            type="button"
            onClick={() => setShowLogs(!showLogs)}
            className="btn btn-sm btn-secondary"
            style={{ fontSize: '0.78rem' }}
          >
            <Terminal size={14} />
            {showLogs ? 'Hide Execution Logs' : 'View Pipeline Execution Logs'}
            {showLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {showLogs && (
            <pre
              style={{
                marginTop: '10px',
                padding: '12px',
                background: '#0f172a',
                color: '#38bdf8',
                borderRadius: 'var(--radius-md)',
                fontSize: '0.76rem',
                fontFamily: 'var(--font-mono)',
                maxHeight: '220px',
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.4
              }}
            >
              {statusData.logs.slice(-50).join('\n')}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
