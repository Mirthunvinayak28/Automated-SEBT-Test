import React from 'react';
import { AlertCircle, ShieldAlert, CheckCircle, Info, FileText } from 'lucide-react';

export default function ClinicalLimitations({ summary = {} }) {
  const limitations = summary.limitations || [
    {
      item: '1. Ankle vs Toe Tracking',
      description: 'MediaPipe tracks lateral ankle landmarks (27/28), whereas standard clinical SEBT measures reach to the furthest point reached by the distal toe touching the floor grid lightly without bearing body weight.'
    },
    {
      item: '2. Starting Baseline Reference',
      description: "Directional reach displacement is measured relative to the moving ankle's resting baseline position prior to movement onset, rather than from the exact center of the floor grid."
    },
    {
      item: '3. Initial Depth Borrowing',
      description: 'The starting 3D ankle coordinate uses the stance ankle metric depth at peak reach to maintain geometric stability, rather than sampling noisy monocular depth during resting foot contact.'
    },
    {
      item: '4. 3D Vector vs 1D Axis',
      description: 'Displacement is calculated as 3D Euclidean distance (incorporating vertical ankle lift and out-of-plane variation) rather than a pure 1D scalar distance projected along the taped floor direction line.'
    }
  ];

  return (
    <section className="dashboard-card" id="clinical-limitations-section">
      <div className="card-header">
        <div className="card-header-title">
          <span className="card-step-badge">Clinical Context</span>
          <h2>Interpretation and Clinical Limitations</h2>
        </div>
      </div>

      {/* Critical Scientific Interpretation Guidance */}
      <div style={{ background: '#f8fafc', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '16px 20px', marginBottom: '20px' }}>
        <h3 style={{ fontSize: '0.96rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Info size={18} color="var(--primary-600)" />
          Score Interpretation Guidelines
        </h3>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '8px' }}>
          In accordance with clinical biomechanics and sports medicine standards, <strong>normalized SEBT percentages should not be reduced to arbitrary cutoffs such as "good balance" or "poor balance"</strong>. Functional dynamic balance performance is multifactorial and depends on participant demographics (age, sex, athletic discipline, prior injury history), the specific study protocol adopted, and validated normative reference values established for that target cohort.
        </p>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <strong>Partial Composite Note:</strong> If fewer than eight directions were performed or validated, the resulting score is a <em>Partial Composite</em>. A partial composite score reflects performance only across the specific evaluated reach directions and cannot be clinically compared directly to a full eight-direction SEBT composite.
        </p>
      </div>

      {/* Four Methodological Disclosures */}
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={16} color="var(--primary-600)" />
          Methodological Disclosures (Automated Vision Pipeline)
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
          {limitations.map((lim, idx) => (
            <div
              key={idx}
              style={{
                background: '#ffffff',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '12px 16px'
              }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.86rem', color: 'var(--primary-700)', marginBottom: '4px' }}>
                {lim.item}
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                {lim.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Medical Disclaimer Alert */}
      <div style={{ background: '#fef2f2', border: '1px solid var(--danger-border)', borderRadius: 'var(--radius-md)', padding: '14px 18px', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
        <ShieldAlert size={20} color="var(--danger-dot)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.84rem', color: 'var(--danger-text)', lineHeight: 1.5 }}>
          <strong>Experimental Research Disclaimer:</strong> The Automated SEBT Test web application and computer vision pipeline are designed strictly for biomechanical research and exploratory assessment. This system is <strong>not a medical device, diagnostic instrument, or clinical injury prediction tool</strong>. It does not replace physical examination, clinical balance testing, or diagnostic appraisal by a licensed physical therapist, athletic trainer, physician, or other qualified healthcare professional.
        </div>
      </div>
    </section>
  );
}
