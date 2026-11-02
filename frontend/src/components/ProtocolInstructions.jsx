import React, { useState } from 'react';
import { ShieldCheck, Video, CheckCircle2, AlertTriangle, Compass, ChevronDown, ChevronUp } from 'lucide-react';

const SEBT_DIRECTIONS = [
  { code: 'ANT', name: 'Anterior', angle: '0°', desc: 'Directly forward' },
  { code: 'AM', name: 'Anteromedial', angle: '+45°', desc: 'Forward and inward across midline' },
  { code: 'M', name: 'Medial', angle: '+90°', desc: 'Directly inward toward opposite side' },
  { code: 'PM', name: 'Posteromedial', angle: '+135°', desc: 'Backward and inward across midline' },
  { code: 'POST', name: 'Posterior', angle: '±180°', desc: 'Directly backward' },
  { code: 'PL', name: 'Posterolateral', angle: '-135°', desc: 'Backward and outward away from midline' },
  { code: 'LAT', name: 'Lateral', angle: '-90°', desc: 'Directly outward away from midline' },
  { code: 'AL', name: 'Anterolateral', angle: '-45°', desc: 'Forward and outward away from midline' },
];

export default function ProtocolInstructions({ isAcknowledged, onToggleAcknowledge }) {
  const [showFullDetails, setShowFullDetails] = useState(true);

  return (
    <section className="dashboard-card" id="test-protocol-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Step 1</span>
          <h2>Test Instructions and Safety Guidelines</h2>
        </div>
        <button
          onClick={() => setShowFullDetails(!showFullDetails)}
          className="btn btn-sm btn-secondary"
          aria-expanded={showFullDetails}
        >
          {showFullDetails ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          {showFullDetails ? 'Collapse Guidelines' : 'Expand Guidelines'}
        </button>
      </div>

      {showFullDetails && (
        <>
          {/* Procedure & Safety */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '24px' }}>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldCheck size={18} color="var(--primary-600)" />
                Standard Test Protocol
              </h3>
              <ol style={{ paddingLeft: '20px', fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <li>Perform the test on a flat, non-slip surface with sufficient clear space around the participant.</li>
                <li>Arrange eight tape lines in the standard SEBT directions radiating from a common centre point (at 45° intervals).</li>
                <li>Have the participant stand at the centre on the designated stance leg while maintaining single-leg balance and reaching with the opposite leg.</li>
                <li>Perform reaches in the required directions: <strong>ANT, AM, M, PM, POST, PL, LAT, and AL</strong>.</li>
                <li>Return the reaching foot to the starting position at the centre after each reach without bearing weight through the reaching foot.</li>
                <li>Record separate trials for each direction. Follow your study's chosen protocol regarding repetition counts and which trials are eligible for scoring.</li>
                <li><strong>Safety first:</strong> Immediately stop testing if the participant experiences pain, dizziness, loss of balance, or feels unsafe. Ensure appropriate supervision or spotters are present.</li>
              </ol>
            </div>

            {/* Video Recording Requirements Checklist */}
            <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '16px 20px' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Video size={18} color="var(--primary-600)" />
                Video Recording Checklist
              </h3>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>The participant must face the camera directly, as required by the front-view tracking model.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>The camera must remain strictly stationary throughout the entire recording session.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>Position the camera so the participant's full body, both legs, both feet, and floor grid remain visible in every reach.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>Maintain suitable camera distance and elevation to avoid cropping feet or tape line endpoints.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>Ensure stable indoor lighting; avoid deep shadows, extreme backlighting, and motion blur.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>Ensure the eight tape lines and their common centre intersection are clearly visible.</span>
                </li>
                <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                  <CheckCircle2 size={16} color="var(--primary-600)" style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>Format must be standard MP4 video.</span>
                </li>
              </ul>
            </div>
          </div>

          {/* 8 Directions Diagram & Grid */}
          <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-subtle)', paddingTop: '18px' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Compass size={18} color="var(--primary-600)" />
              The Eight SEBT Directions (45° Sectors)
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '10px' }}>
              {SEBT_DIRECTIONS.map((dir) => (
                <div
                  key={dir.code}
                  style={{
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-subtle)',
                    background: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px'
                  }}
                >
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      fontSize: '0.85rem',
                      background: 'var(--primary-50)',
                      color: 'var(--primary-700)',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      minWidth: '50px',
                      textAlign: 'center'
                    }}
                  >
                    {dir.code}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      {dir.name} <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>({dir.angle})</span>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      {dir.desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* Mandatory Acknowledgement Checkbox */}
      <div
        className="checkbox-card"
        onClick={() => onToggleAcknowledge(!isAcknowledged)}
        role="checkbox"
        aria-checked={isAcknowledged}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            onToggleAcknowledge(!isAcknowledged);
          }
        }}
      >
        <input
          id="protocol-acknowledgement-checkbox"
          type="checkbox"
          checked={isAcknowledged}
          onChange={(e) => {
            e.stopPropagation();
            onToggleAcknowledge(e.target.checked);
          }}
        />
        <span>
          <strong>Protocol & Safety Acknowledgement:</strong> I confirm that I have reviewed the SEBT testing procedure, verified that the camera captures full body and floor markings throughout the trial, and confirmed that the recording adheres to safety and supervision protocols.
        </span>
      </div>
    </section>
  );
}
