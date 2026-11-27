import React from 'react';
import { Activity, RotateCcw } from 'lucide-react';

export default function Header({ onReset, currentJobId }) {
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
