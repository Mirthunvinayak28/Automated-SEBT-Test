import React, { useState, useMemo } from 'react';
import { Filter, CheckCircle2, XCircle, AlertTriangle, Layers } from 'lucide-react';

export default function MovementsTable({ movements = [] }) {
  const [directionFilter, setDirectionFilter] = useState('ALL');
  const [validityFilter, setValidityFilter] = useState('ALL');

  // Extract distinct directions present in data
  const distinctDirections = useMemo(() => {
    const set = new Set();
    movements.forEach((m) => {
      if (m.detected_direction) set.add(m.detected_direction);
    });
    return Array.from(set).sort();
  }, [movements]);

  // Filtered rows
  const filteredRows = useMemo(() => {
    return movements.filter((m) => {
      const matchDir = directionFilter === 'ALL' || m.detected_direction === directionFilter;
      const matchVal =
        validityFilter === 'ALL' ||
        (validityFilter === 'VALID' && m.scoring_validity === 'VALID') ||
        (validityFilter === 'EXCLUDED' && m.scoring_validity !== 'VALID');
      return matchDir && matchVal;
    });
  }, [movements, directionFilter, validityFilter]);

  if (!movements || movements.length === 0) {
    return null;
  }

  return (
    <section className="dashboard-card" id="movements-table-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Individual Movements</span>
          <h2>Individual Movement Results ({movements.length} Total Trials)</h2>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Showing {filteredRows.length} of {movements.length} records • All repeated trials preserved separately
        </div>
      </div>

      {/* Filter Controls */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', background: '#f8fafc', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
          <Filter size={16} color="var(--primary-600)" />
          Filters:
        </div>

        {/* Direction Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label htmlFor="direction-filter-select" style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Direction:</label>
          <select
            id="direction-filter-select"
            value={directionFilter}
            onChange={(e) => setDirectionFilter(e.target.value)}
            className="form-input"
            style={{ padding: '6px 10px', fontSize: '0.82rem', width: 'auto' }}
          >
            <option value="ALL">All Directions</option>
            {distinctDirections.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {/* Validity Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <label htmlFor="validity-filter-select" style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Validity Status:</label>
          <select
            id="validity-filter-select"
            value={validityFilter}
            onChange={(e) => setValidityFilter(e.target.value)}
            className="form-input"
            style={{ padding: '6px 10px', fontSize: '0.82rem', width: 'auto' }}
          >
            <option value="ALL">All Trials</option>
            <option value="VALID">Valid Scored Only</option>
            <option value="EXCLUDED">Excluded Only</option>
          </select>
        </div>

        {(directionFilter !== 'ALL' || validityFilter !== 'ALL') && (
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => {
              setDirectionFilter('ALL');
              setValidityFilter('ALL');
            }}
            style={{ marginLeft: 'auto' }}
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* Table */}
      <div className="table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              <th>Trial #</th>
              <th>Reaching Leg</th>
              <th>Stance Leg</th>
              <th>Direction</th>
              <th>Angle (°)</th>
              <th>Confidence</th>
              <th>Reach Disp (cm)</th>
              <th>Manual Score (%)</th>
              <th>Video-Est Score (%)</th>
              <th>Duration</th>
              <th>Validity</th>
              <th>Exclusion Reason</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row) => {
              const isValid = row.scoring_validity === 'VALID';
              const mScore = row.normalized_score_manual_pct;
              const vScore = row.normalized_score_video_pct;
              const reachCm = row.directional_displacement_cm;

              return (
                <tr key={row.movement} style={{ backgroundColor: !isValid ? '#fffaf8' : undefined }}>
                  <td className="col-num" style={{ fontWeight: 600 }}>
                    #{row.movement}
                  </td>
                  <td>
                    <span style={{ fontWeight: 500, color: row.moving_ankle === 'LEFT' ? '#2563eb' : '#dc2626' }}>
                      {row.moving_ankle}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontWeight: 500 }}>
                      {row.stance_leg}
                    </span>
                  </td>
                  <td>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 700,
                        padding: '3px 7px',
                        borderRadius: '4px',
                        background: isValid ? 'var(--primary-50)' : '#f1f5f9',
                        color: isValid ? 'var(--primary-700)' : 'var(--text-muted)'
                      }}
                    >
                      {row.detected_direction || 'Unknown'}
                    </span>
                  </td>
                  <td className="col-num" style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    {row.direction_confirmed_frame ? `Frame ${row.direction_confirmed_frame}` : '--'}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        row.direction_confidence?.startsWith('High')
                          ? 'badge-success'
                          : row.direction_confidence?.startsWith('Medium')
                          ? 'badge-info'
                          : 'badge-warning'
                      }`}
                      style={{ fontSize: '0.74rem' }}
                    >
                      {row.direction_confidence || '--'}
                    </span>
                  </td>
                  <td className="col-num" style={{ fontWeight: 600 }}>
                    {reachCm !== null ? `${reachCm.toFixed(1)} cm` : '--'}
                  </td>
                  <td className="col-num" style={{ fontWeight: 700, color: isValid ? 'var(--primary-700)' : 'var(--text-muted)' }}>
                    {mScore !== null ? `${mScore.toFixed(1)}%` : '--'}
                  </td>
                  <td className="col-num" style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    {vScore !== null ? `${vScore.toFixed(1)}%` : '--'}
                  </td>
                  <td className="col-num" style={{ fontSize: '0.82rem' }}>
                    {row.duration_sec !== null ? `${row.duration_sec.toFixed(2)}s` : '--'}
                  </td>
                  <td>
                    <span className={`badge ${isValid ? 'badge-success' : 'badge-danger'}`}>
                      {isValid ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                      {row.scoring_validity}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.78rem', color: isValid ? 'var(--text-muted)' : 'var(--danger-text)', maxWidth: '200px' }}>
                    {row.scoring_exclusion_reason || 'VALID'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: '12px', fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <span>Data extracted from <code>*_depth_results.csv</code></span>
        <span>Peak displacement measured in 3D metric camera space</span>
      </div>
    </section>
  );
}
