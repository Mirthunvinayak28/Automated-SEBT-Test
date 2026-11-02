import React from 'react';
import { Award, CheckCircle2, AlertCircle, Compass, HelpCircle } from 'lucide-react';

export default function CompositeScoresCard({ compositeScores = {} }) {
  if (!compositeScores || Object.keys(compositeScores).length === 0) {
    return null;
  }

  const leftStance = compositeScores['LEFT_STANCE'] || {};
  const rightStance = compositeScores['RIGHT_STANCE'] || {};
  const combined = compositeScores['COMBINED_ALL_STANCE'] || {};

  const renderCompositeCard = (groupData, title, stanceDesc) => {
    const denom = groupData.denominator ?? 0;
    const manualPct = groupData.manual_composite_pct;
    const videoPct = groupData.video_composite_pct;
    const statusLabel = groupData.status_label || (denom === 8 ? 'Complete 8/8 directions' : denom > 0 ? `Partial composite (${denom}/8)` : 'Unavailable');
    const isComplete = denom === 8;
    const isUnavailable = denom === 0 || manualPct === null;
    const includedDirs = groupData.included_dirs || [];
    const missingDirs = groupData.missing_dirs || [];
    const bestTrials = groupData.best_trials || {};

    return (
      <div className="composite-card" key={title}>
        <div>
          <div className="composite-header">
            <div>
              <div className="composite-title">{title}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                {stanceDesc}
              </div>
            </div>
            <span
              className={`badge ${
                isComplete
                  ? 'badge-success'
                  : !isUnavailable
                  ? 'badge-warning'
                  : 'badge-danger'
              }`}
            >
              {isComplete ? (
                <CheckCircle2 size={12} />
              ) : !isUnavailable ? (
                <Compass size={12} />
              ) : (
                <AlertCircle size={12} />
              )}
              {isComplete ? 'Complete 8/8 Directions' : denom > 0 ? `Partial Composite (${denom}/8)` : 'Unavailable'}
            </span>
          </div>

          {/* Big Score Display */}
          <div className="composite-score-big">
            {isUnavailable ? (
              <span style={{ fontSize: '1.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                Unavailable
              </span>
            ) : (
              <>
                <span>{manualPct.toFixed(2)}</span>
                <span className="composite-score-unit">% of leg length</span>
              </>
            )}
          </div>

          {/* Video-Estimated Secondary Score */}
          <div className="composite-video-subscore">
            <strong>Video-Estimated Length Score:</strong>{' '}
            {videoPct !== null && videoPct !== undefined ? `${videoPct.toFixed(2)}%` : 'Unavailable'}
            <span style={{ display: 'block', fontSize: '0.74rem', color: 'var(--text-light)', marginTop: '2px' }}>
              (Experimental hip-to-ankle approximation — not clinical ASIS)
            </span>
          </div>

          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
            <strong>Valid Distinct Directions:</strong> {denom}/8 evaluated ({groupData.valid_movements_count || 0} valid trials total)
          </div>

          {/* Included & Missing Directions */}
          <div className="composite-directions-section">
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '4px' }}>
              Included Directions ({includedDirs.length}):
            </div>
            <div className="dir-tags-wrap">
              {includedDirs.length > 0 ? (
                includedDirs.map((d) => (
                  <span key={d} className="dir-tag dir-tag-included">
                    {d}
                  </span>
                ))
              ) : (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None</span>
              )}
            </div>

            {missingDirs.length > 0 && (
              <div style={{ marginTop: '10px' }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
                  Missing Directions ({missingDirs.length}/8):
                </div>
                <div className="dir-tags-wrap">
                  {missingDirs.map((d) => (
                    <span key={d} className="dir-tag dir-tag-missing">
                      {d}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Selected Trials Breakdown */}
        {Object.keys(bestTrials).length > 0 && (
          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px' }}>
              Selected Best Trials (Maximum Reach):
            </div>
            <div style={{ maxHeight: '160px', overflowY: 'auto', fontSize: '0.78rem' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
                    <th style={{ padding: '4px 6px' }}>Dir</th>
                    <th style={{ padding: '4px 6px' }}>Trial</th>
                    <th style={{ padding: '4px 6px' }}>Reach (cm)</th>
                    <th style={{ padding: '4px 6px' }}>Manual %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bestTrials).map(([dirName, tr]) => (
                    <tr key={dirName} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 6px', fontWeight: 600 }}>{dirName}</td>
                      <td style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>#{tr.movement}</td>
                      <td style={{ padding: '4px 6px', fontFamily: 'var(--font-mono)' }}>
                        {tr.valid_reach_dist_cm ? `${tr.valid_reach_dist_cm.toFixed(1)} cm` : '--'}
                      </td>
                      <td style={{ padding: '4px 6px', fontWeight: 600, color: 'var(--primary-700)', fontFamily: 'var(--font-mono)' }}>
                        {tr.norm_score_manual_pct ? `${tr.norm_score_manual_pct.toFixed(1)}%` : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <section className="dashboard-card" id="composite-scores-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Composite Scores</span>
          <h2>Overall SEBT Results</h2>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Normalized to participant's manual clinical leg lengths
        </div>
      </div>

      <div style={{ marginBottom: '18px', background: 'var(--primary-50)', border: '1px solid var(--primary-100)', borderRadius: 'var(--radius-md)', padding: '12px 16px', fontSize: '0.86rem', color: 'var(--primary-700)' }}>
        <strong>Scoring Protocol Formulation:</strong> When multiple trials are performed in the same direction, the trial with maximum valid reach distance is selected. The composite score is the arithmetic mean of normalized scores across all distinct valid directions performed (actual count used as denominator).
      </div>

      <div className="composite-grid">
        {renderCompositeCard(
          leftStance,
          'Left-Stance Composite Score',
          'Standing on Left leg, reaching with Right leg'
        )}
        {renderCompositeCard(
          rightStance,
          'Right-Stance Composite Score',
          'Standing on Right leg, reaching with Left leg'
        )}
        {combined.denominator !== undefined &&
          renderCompositeCard(
            combined,
            'Combined Stance Composite Score',
            'Pooled across both stance leg conditions'
          )}
      </div>
    </section>
  );
}
