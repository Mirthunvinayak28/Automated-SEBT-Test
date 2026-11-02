import React from 'react';
import { Ruler, CheckCircle2, AlertTriangle, Activity, HelpCircle } from 'lucide-react';

export default function CalibrationObservations({ observations = {}, summary = {} }) {
  if (!observations || Object.keys(observations).length === 0) {
    return null;
  }

  const legLengths = summary.leg_lengths || {};
  const leftLeg = legLengths['LEFT'] || {};
  const rightLeg = legLengths['RIGHT'] || {};

  return (
    <section className="dashboard-card" id="calibration-observations-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Observations</span>
          <h2>Measurement and Calibration Observations</h2>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Clinical ASIS-to-malleolus vs Monocular Depth Metric Hip-to-Ankle
        </div>
      </div>

      {/* Experimental vs Clinical Disclosure Note */}
      <div style={{ marginBottom: '20px', background: '#fffbeb', border: '1px solid var(--warning-border)', borderRadius: 'var(--radius-md)', padding: '14px 18px', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
        <AlertTriangle size={20} color="var(--warning-dot)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.86rem', color: 'var(--warning-text)', lineHeight: 1.5 }}>
          <strong>Experimental Measurement Note:</strong> The automated pipeline computes 3D Euclidean hip-to-ankle length across resting frames using Depth Anything V2 metric prediction. This is an experimental biometric estimate, <strong>not a substitute for the clinical manual measurement</strong> from the anterior superior iliac spine (ASIS) to the medial malleolus. The manual measurement was utilized as the ground-truth denominator for all clinical SEBT scores.
        </div>
      </div>

      {/* Leg Length Comparison Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        {/* Left Leg Card */}
        <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px', background: '#ffffff' }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Left Leg Calibration</span>
            <span className="badge badge-info">Left Limb</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Manual Clinical Length:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{leftLeg.manual_cm ? `${leftLeg.manual_cm.toFixed(1)} cm` : '--'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Video-Estimated Length:</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--primary-700)' }}>
                {leftLeg.video_est_cm ? `${leftLeg.video_est_cm.toFixed(1)} cm` : 'Unavailable'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Absolute Difference:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                {leftLeg.abs_error_cm !== null && leftLeg.abs_error_cm !== undefined ? `${leftLeg.abs_error_cm.toFixed(2)} cm` : '--'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Relative Error (%):</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                {leftLeg.pct_error !== null && leftLeg.pct_error !== undefined ? `${leftLeg.pct_error.toFixed(2)}%` : '--'}
              </span>
            </div>
          </div>
        </div>

        {/* Right Leg Card */}
        <div style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '18px', background: '#ffffff' }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Right Leg Calibration</span>
            <span className="badge badge-info">Right Limb</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Manual Clinical Length:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{rightLeg.manual_cm ? `${rightLeg.manual_cm.toFixed(1)} cm` : '--'}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Video-Estimated Length:</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--primary-700)' }}>
                {rightLeg.video_est_cm ? `${rightLeg.video_est_cm.toFixed(1)} cm` : 'Unavailable'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Absolute Difference:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                {rightLeg.abs_error_cm !== null && rightLeg.abs_error_cm !== undefined ? `${rightLeg.abs_error_cm.toFixed(2)} cm` : '--'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Relative Error (%):</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                {rightLeg.pct_error !== null && rightLeg.pct_error !== undefined ? `${rightLeg.pct_error.toFixed(2)}%` : '--'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Movement Quality & Reach Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px' }}>
        <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px', textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Completed Movements</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-main)', marginTop: '4px' }}>{observations.total_movements || 0}</div>
        </div>
        <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px', textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--success-text)', textTransform: 'uppercase' }}>Valid Scored Trials</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--success-text)', marginTop: '4px' }}>{observations.valid_movements || 0}</div>
        </div>
        <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px', textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--danger-text)', textTransform: 'uppercase' }}>Excluded Trials</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--danger-text)', marginTop: '4px' }}>{observations.excluded_movements || 0}</div>
        </div>
        <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '14px', textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--primary-700)', textTransform: 'uppercase' }}>Directions Evaluated</div>
          <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--primary-700)', marginTop: '4px' }}>
            {observations.detected_directions ? `${observations.detected_directions.length}/8` : '0/8'}
          </div>
        </div>
      </div>
    </section>
  );
}
