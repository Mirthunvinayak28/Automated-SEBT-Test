import React from 'react';
import { Ruler, AlertCircle, CheckCircle, Info } from 'lucide-react';

export default function LegMeasurementForm({
  leftLeg,
  rightLeg,
  onChangeLeft,
  onChangeRight,
  disabled = false
}) {
  const validateLeg = (val) => {
    if (val === '' || val === null || val === undefined) return { valid: false, msg: 'Leg length is required.' };
    const num = parseFloat(val);
    if (isNaN(num)) return { valid: false, msg: 'Must be a valid numerical value.' };
    if (num <= 0) return { valid: false, msg: 'Must be greater than 0 cm.' };
    if (num < 40 || num > 140) return { valid: true, warning: 'Value is outside typical adult range (40–140 cm). Please verify.' };
    return { valid: true, msg: 'Valid measurement.' };
  };

  const leftValidation = validateLeg(leftLeg);
  const rightValidation = validateLeg(rightLeg);

  return (
    <section className="dashboard-card" id="leg-measurements-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Step 2</span>
          <h2>Participant Measurement Details</h2>
        </div>
      </div>

      <div style={{ marginBottom: '18px', background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px 18px', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
        <Info size={20} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <strong>Clinical Protocol:</strong> Measure true clinical leg length from the <strong>anterior superior iliac spine (ASIS)</strong> to the <strong>distal tip of the medial malleolus</strong> while the participant is supine.
          <div style={{ marginTop: '4px', color: 'var(--text-muted)' }}>
            <strong>Important:</strong> These manual measurements serve as the primary denominator for all normalized SEBT reach calculations. Video-derived hip-to-ankle estimates are experimental and approximate.
          </div>
        </div>
      </div>

      <div className="form-grid">
        {/* Left Leg Input */}
        <div className="form-group">
          <label className="form-label" htmlFor="left-leg-input">
            <span>Left Clinical Leg Length</span>
            <span className="form-label-hint">ASIS to medial malleolus</span>
          </label>
          <div className="input-wrapper">
            <input
              id="left-leg-input"
              type="number"
              step="0.1"
              min="1"
              max="200"
              placeholder="e.g. 88.5"
              value={leftLeg}
              onChange={(e) => onChangeLeft(e.target.value)}
              disabled={disabled}
              className={`form-input ${!leftValidation.valid && leftLeg !== '' ? 'is-invalid' : ''}`}
            />
            <span className="input-unit">cm</span>
          </div>
          {leftLeg !== '' && (
            <div className={`form-feedback ${leftValidation.valid ? (leftValidation.warning ? 'is-warning' : '') : 'is-error'}`} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              {leftValidation.valid ? (
                leftValidation.warning ? (
                  <>
                    <AlertCircle size={14} color="var(--warning-dot)" />
                    <span style={{ color: 'var(--warning-text)' }}>{leftValidation.warning}</span>
                  </>
                ) : (
                  <>
                    <CheckCircle size={14} color="var(--success-dot)" />
                    <span style={{ color: 'var(--success-text)' }}>{leftValidation.msg}</span>
                  </>
                )
              ) : (
                <>
                  <AlertCircle size={14} color="var(--danger-dot)" />
                  <span>{leftValidation.msg}</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Right Leg Input */}
        <div className="form-group">
          <label className="form-label" htmlFor="right-leg-input">
            <span>Right Clinical Leg Length</span>
            <span className="form-label-hint">ASIS to medial malleolus</span>
          </label>
          <div className="input-wrapper">
            <input
              id="right-leg-input"
              type="number"
              step="0.1"
              min="1"
              max="200"
              placeholder="e.g. 89.0"
              value={rightLeg}
              onChange={(e) => onChangeRight(e.target.value)}
              disabled={disabled}
              className={`form-input ${!rightValidation.valid && rightLeg !== '' ? 'is-invalid' : ''}`}
            />
            <span className="input-unit">cm</span>
          </div>
          {rightLeg !== '' && (
            <div className={`form-feedback ${rightValidation.valid ? (rightValidation.warning ? 'is-warning' : '') : 'is-error'}`} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              {rightValidation.valid ? (
                rightValidation.warning ? (
                  <>
                    <AlertCircle size={14} color="var(--warning-dot)" />
                    <span style={{ color: 'var(--warning-text)' }}>{rightValidation.warning}</span>
                  </>
                ) : (
                  <>
                    <CheckCircle size={14} color="var(--success-dot)" />
                    <span style={{ color: 'var(--success-text)' }}>{rightValidation.msg}</span>
                  </>
                )
              ) : (
                <>
                  <AlertCircle size={14} color="var(--danger-dot)" />
                  <span>{rightValidation.msg}</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
