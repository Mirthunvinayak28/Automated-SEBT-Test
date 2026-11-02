import React from 'react';
import { Activity, RotateCcw, Database, Play } from 'lucide-react';

export default function Header({ onReset, onLoadSample, samples = [], currentJobId }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="header-icon-box">
          <Activity size={30} strokeWidth={2.4} />
        </div>
        <div className="header-title-area">
          <h1>Automated SEBT Test</h1>
          <p>Video-based experimental assessment of dynamic balance using the Star Excursion Balance Test (SEBT).</p>
        </div>
      </div>

      <div className="header-actions">
        {samples.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontWeight: 500 }}>
              Sample Datasets:
            </span>
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => onLoadSample(s.id)}
                className={`btn btn-sm ${currentJobId === s.id ? 'btn-primary' : 'btn-secondary'}`}
                title={`Load precomputed benchmark: ${s.title}`}
              >
                <Database size={14} />
                {s.video_filename}
              </button>
            ))}
          </div>
        )}

        {currentJobId && (
          <button onClick={onReset} className="btn btn-sm btn-secondary" title="Start a fresh test session">
            <RotateCcw size={14} />
            New Test Session
          </button>
        )}
      </div>
    </header>
  );
}
