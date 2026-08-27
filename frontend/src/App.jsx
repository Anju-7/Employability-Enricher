import React, { useState } from 'react';
import './Dashboard.css';

const EmployabilityDashboard = () => {
  const [selectedTab, setSelectedTab] = useState('LANDING');

  const telemetryData = [
    { timestamp: '2024-05-20_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2024-05-20_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2024-05-20_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2024-05-20_11:59:12', filename: '▲ TIMEOUT_ERROR_R_S8', score: '92.4', latency: '30.00s', isError: true },
    { timestamp: '2024-05-20_11:59:02', filename: '▲ TIMEOUT_ERROR_R_S8', score: 'N/A', latency: '30.00s', isError: true },
    { timestamp: '2024-05-20_11:59:02', filename: '▲ TIMEOUT_ERROR_R_S8', score: 'N/A', latency: '30.00s', isError: true },
  ];

  return (
    <div className="dashboard-container">
      {/* SECTION 1: LANDING */}
      <div className="dashboard-card">
        <div className="card-header">
          <span>EMPLOYABILITY_ENRICHER</span>
          <span>
            <a href="#landing" style={{ color: 'white', marginRight: 15 }}>LANDING</a>
            <a href="#dashboard" style={{ color: 'var(--text-muted)' }}>DASHBOARD</a>
          </span>
          <span className="title-glow">(01) SYSTEM_ONLINE</span>
        </div>

        <h2 className="main-title">
          SCORE YOUR RESUME <span className="title-glow">AGAINST WHAT THE MARKET IS ACTUALLY HIRING FOR RIGHT NOW.</span>
        </h2>

        <div style={{ textAlign: 'center' }}>
          <button className="cyan-btn">INITIALIZE_UPLOAD ↑</button>
        </div>

        <div className="steps-grid">
          <div className="step-card">
            <div>📄</div>
            <div className="step-title">01 UPLOAD_RESUME</div>
            <div className="step-desc">Upload your resume vector or profile session data stream.</div>
          </div>
          <div className="step-card">
            <div>🎯</div>
            <div className="step-title">02 ANALYZE_MARKET_SIGNAL</div>
            <div className="step-desc">Analyze market signal vectors against real-time recruitment streams.</div>
          </div>
          <div className="step-card">
            <div>⏱️</div>
            <div className="step-title">03 OUTPUT_ALIGNMENT_SCORE</div>
            <div className="step-desc">Output alignment score aligns your assets to baseline targets.</div>
          </div>
        </div>

        <div className="metrics-banner">
          <div>
            <div className="metric-label">PRECISION AUDITING FOR THE MODERN CAREER</div>
            <div style={{ display: 'flex', gap: '30px', marginTop: '5px' }}>
              <div>
                <span className="metric-value">14.2ms</span>
                <div className="metric-label">MEAN ANALYSIS LATENCY</div>
              </div>
              <div>
                <span className="metric-value">99.8%</span>
                <div className="metric-label">PARSER ACCURACY</div>
              </div>
            </div>
          </div>
          <div>
            <div className="metric-label" style={{ marginBottom: 5 }}>READY TO AUDIT YOUR VALUE?</div>
            <button className="cyan-btn">START_ANALYSIS_SEQ</button>
          </div>
        </div>
      </div>

      {/* SECTION 2: UPLOAD & RESULTS */}
      <div className="dashboard-card">
        <div className="card-header">
          <span>EMPLOYABILITY_ENRICHER</span>
          <span>UPLOAD & RESULTS</span>
          <span className="title-glow">(02) SYSTEM_ONLINE</span>
        </div>

        <h3 className="title-glow" style={{ margin: '0 0 10px 0' }}>ALIGNMENT_ENGINE [v4.2]</h3>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 20 }}>
          Upload professional profile (PDF) for deep-layer cross-referencing against sector-specific technical benchmarks and employability heuristics.
        </p>

        <div className="upload-grid">
          <div>
            <div className="dropzone">
              <span style={{ fontSize: 24, color: 'var(--cyan-primary)' }}>↑</span>
              <div style={{ color: 'var(--cyan-primary)', fontWeight: 'bold', margin: '10px 0' }}>
                DRAG_DROP_OR_BROWSE
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>[SUPPORTED_FORMATS: .PDF]</div>
            </div>
            <button className="cyan-btn" style={{ width: '100%', marginTop: 15 }}>
              RUN_ALIGNMENT_CHECK
            </button>
          </div>

          <div className="awaiting-panel">
            <span style={{ fontSize: 30, marginBottom: 10 }}>🔍</span>
            <span>AWAITING_INPUT_STREAM...</span>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 25 }}>
          <span>[S75_000E_774]</span>
          <div>
            <span style={{ marginRight: 15 }}>DOCUMENTATION</span>
            <span style={{ marginRight: 15 }}>TERMS</span>
            <span>PRIVACY</span>
          </div>
          <span>V1.0.4-BETA | SYSTEM_TIME: 2024-05-20_T12:00:00Z</span>
        </div>
      </div>

      {/* SECTION 3: MONITORING DASHBOARD */}
      <div className="dashboard-card">
        <div className="card-header">
          <span>EMPLOYABILITY_ENRICHER</span>
          <span>MONITORING DASHBOARD</span>
          <span className="title-glow">(03) SYSTEM_ONLINE</span>
        </div>

        <div className="telemetry-layout">
          <div className="sidebar-section">
            <div style={{ fontSize: 11, fontWeight: 'bold', marginBottom: 15 }}>CORE_METRICS</div>
            <div className="metric-row">
              <span style={{ color: 'var(--text-muted)' }}>QUEUE_STATUS:</span>
              <span className="status-active">ACTIVE</span>
            </div>
            <div className="metric-row">
              <span style={{ color: 'var(--text-muted)' }}>PROCESSING_LOAD:</span>
              <span>42.8%</span>
            </div>
            <div className="metric-row">
              <span style={{ color: 'var(--text-muted)' }}>AVG_LATENCY:</span>
              <span>1.24s</span>
            </div>

            <div style={{ fontSize: 11, fontWeight: 'bold', margin: '20px 0 10px 0' }}>DISTRIBUTION_MAP</div>
            <div className="distribution-grid">
              {[...Array(24)].map((_, i) => (
                <div key={i} className={`dist-cell ${[3, 8, 14, 15, 20].includes(i) ? 'active' : ''}`} />
              ))}
            </div>

            <div style={{ fontSize: 10, marginTop: 40, color: 'var(--cyan-primary)' }}>
              TERMINAL_READY_<br />
              &gt; <span style={{ animation: 'blink 1s infinite' }}>■</span>
            </div>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <h3 className="title-glow" style={{ margin: 0 }}>SYSTEM_TELEMETRY</h3>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                MEAN_SCORE <span style={{ color: 'var(--cyan-primary)' }}>84.2</span> | CONFIDENCE <span style={{ color: 'var(--cyan-primary)' }}>0.99</span>
              </div>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', margin: '5px 0 15px 0' }}>
              REAL-TIME SCORE DISTRIBUTION (RUN_N=50)
            </div>

            {/* SVG Line Chart Placeholder */}
            <svg className="line-chart-placeholder" viewBox="0 0 500 100">
              <path
                d="M 0,80 Q 50,20 100,70 T 200,40 T 300,90 T 400,20 T 500,60"
                fill="none"
                stroke="var(--cyan-primary)"
                strokeWidth="2"
              />
            </svg>

            <table className="telemetry-table">
              <thead>
                <tr>
                  <th>TIMESTAMP</th>
                  <th>FILENAME</th>
                  <th>SCORE</th>
                  <th>LATENCY</th>
                </tr>
              </thead>
              <tbody>
                {telemetryData.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.timestamp}</td>
                    <td style={{ color: row.isError ? 'var(--danger-red)' : 'inherit' }}>{row.filename}</td>
                    <td>
                      <span className={`score-badge ${row.isError ? 'error' : ''}`}>
                        {row.score}
                      </span>
                    </td>
                    <td>{row.latency}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmployabilityDashboard;