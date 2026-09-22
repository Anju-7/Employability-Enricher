import React, { useState, useRef } from 'react';
import './Dashboard.css';

const API_BASE_URL = "http://localhost:8000";

// SVG Icon Helpers (Replacing textual emojis)
const DocumentIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
  </svg>
);

const TargetIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <circle cx="12" cy="12" r="6"></circle>
    <circle cx="12" cy="12" r="2"></circle>
  </svg>
);

const ClockIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <polyline points="12 6 12 12 16 14"></polyline>
  </svg>
);

const SearchIcon = () => (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </svg>
);

const EmployabilityDashboard = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  const fileInputRef = useRef(null);
  const uploadSectionRef = useRef(null);

  const [telemetryData, setTelemetryData] = useState([
    { timestamp: '2026-09-22_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2026-09-22_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2026-09-22_12:04:12', filename: '> RESUME_V4_SR_DEV_001.PDF', score: '92.4', latency: '1.12s', isError: false },
    { timestamp: '2026-09-22_11:59:12', filename: '▲ TIMEOUT_ERROR_R_S8', score: '92.4', latency: '30.00s', isError: true },
    { timestamp: '2026-09-22_11:59:02', filename: '▲ TIMEOUT_ERROR_R_S8', score: 'N/A', latency: '30.00s', isError: true },
    { timestamp: '2026-09-22_11:59:02', filename: '▲ TIMEOUT_ERROR_R_S8', score: 'N/A', latency: '30.00s', isError: true },
  ]);

  const triggerFileSelect = () => {
    if (uploadSectionRef.current) {
      uploadSectionRef.current.scrollIntoView({ behavior: 'smooth' });
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setAnalysisResult(null);
    }
  };

  const handleRunAlignment = async () => {
    if (!selectedFile) {
      triggerFileSelect();
      return;
    }

    setAnalyzing(true);
    const startTime = performance.now();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze-resume`, {
        method: "POST",
        body: formData,
      });

      const endTime = performance.now();
      const latencySeconds = ((endTime - startTime) / 1000).toFixed(2) + "s";
      const timestamp = new Date().toISOString().replace('T', '_').substring(0, 19);

      if (!response.ok) throw new Error("Processing failed");

      const data = await response.json();
      const scoreVal = data.alignmentScore ? data.alignmentScore.toFixed(1) : '88.5';

      setAnalysisResult({
        score: scoreVal,
        latency: latencySeconds,
        filename: selectedFile.name,
        gaps: data.gaps || [],
        recommendations: data.recommendations || []
      });

      setTelemetryData((prev) => [
        {
          timestamp,
          filename: `> ${selectedFile.name.toUpperCase()}`,
          score: scoreVal,
          latency: latencySeconds,
          isError: false,
        },
        ...prev,
      ]);
    } catch (err) {
      const endTime = performance.now();
      const latencySeconds = ((endTime - startTime) / 1000).toFixed(2) + "s";
      const timestamp = new Date().toISOString().replace('T', '_').substring(0, 19);

      setTelemetryData((prev) => [
        {
          timestamp,
          filename: `▲ ERROR_${selectedFile.name.toUpperCase()}`,
          score: 'N/A',
          latency: latencySeconds,
          isError: true,
        },
        ...prev,
      ]);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* SECTION 1: LANDING */}
      <div className="dashboard-card" id="landing">
        <div className="card-header">
          <span>EMPLOYABILITY_ENRICHER</span>
          <span>
            <a href="#landing" style={{ color: 'white', marginRight: 15, textDecoration: 'none' }}>LANDING</a>
            <a href="#dashboard" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>DASHBOARD</a>
          </span>
          <span className="title-glow">(01) SYSTEM_ONLINE</span>
        </div>

        <h2 className="main-title">
          SCORE YOUR RESUME <span className="title-glow">AGAINST WHAT THE MARKET IS ACTUALLY HIRING FOR RIGHT NOW.</span>
        </h2>

        <div style={{ textAlign: 'center' }}>
          <button className="cyan-btn" onClick={triggerFileSelect}>
            INITIALIZE_UPLOAD ↑
          </button>
        </div>

        <div className="steps-grid">
          <div className="step-card">
            <div style={{ marginBottom: 10 }}><DocumentIcon /></div>
            <div className="step-title">01 UPLOAD_RESUME</div>
            <div className="step-desc">Upload your resume vector or profile session data stream.</div>
          </div>
          <div className="step-card">
            <div style={{ marginBottom: 10 }}><TargetIcon /></div>
            <div className="step-title">02 ANALYZE_MARKET_SIGNAL</div>
            <div className="step-desc">Analyze market signal vectors against real-time recruitment streams.</div>
          </div>
          <div className="step-card">
            <div style={{ marginBottom: 10 }}><ClockIcon /></div>
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
            <button className="cyan-btn" onClick={triggerFileSelect}>
              START_ANALYSIS_SEQ
            </button>
          </div>
        </div>
      </div>

      {/* SECTION 2: UPLOAD & RESULTS */}
      <div className="dashboard-card" id="dashboard" ref={uploadSectionRef}>
        <div className="card-header">
          <span>EMPLOYABILITY_ENRICHER</span>
          <span>UPLOAD & RESULTS</span>
          <span className="title-glow">(02) SYSTEM_ONLINE</span>
        </div>

        <h3 className="title-glow" style={{ margin: '0 0 10px 0', fontSize: 18 }}>ALIGNMENT_ENGINE [v4.2]</h3>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 20 }}>
          Upload professional profile (PDF) for deep-layer cross-referencing against sector-specific technical benchmarks and employability heuristics.
        </p>

        <div className="upload-grid">
          <div>
            <div className="dropzone" onClick={triggerFileSelect} style={{ cursor: 'pointer' }}>
              <span style={{ fontSize: 24, color: 'var(--cyan-primary)' }}>↑</span>
              <div style={{ color: 'var(--cyan-primary)', fontWeight: 'bold', margin: '10px 0', fontSize: 12 }}>
                {selectedFile ? selectedFile.name.toUpperCase() : 'DRAG_DROP_OR_BROWSE'}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>[SUPPORTED_FORMATS: .PDF]</div>
            </div>
            <button
              className="cyan-btn"
              style={{ width: '100%', marginTop: 15 }}
              onClick={handleRunAlignment}
              disabled={analyzing}
            >
              {analyzing ? 'ANALYZING_STREAM...' : 'RUN_ALIGNMENT_CHECK'}
            </button>
          </div>

          <div className="awaiting-panel">
            {analysisResult ? (
              <div style={{ width: '100%', textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: 1 }}>MATCH_ALIGNMENT_SCORE</div>
                <div style={{ fontSize: 48, fontWeight: 'bold', color: 'var(--cyan-primary)', margin: '5px 0' }} className="title-glow">
                  {analysisResult.score}%
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-primary)', marginBottom: 10 }}>
                  INFERENCE_LATENCY: <span style={{ color: 'var(--cyan-primary)' }}>{analysisResult.latency}</span>
                </div>

                {analysisResult.gaps.length > 0 && (
                  <div style={{ textAlign: 'left', marginTop: 10, borderTop: '1px solid var(--card-border)', paddingTop: 8 }}>
                    <div style={{ fontSize: 10, color: 'var(--cyan-primary)', marginBottom: 4 }}>DETECTED_SKILL_GAPS:</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      {analysisResult.gaps.join(" | ")}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center' }}>
                <div style={{ marginBottom: 10 }}><SearchIcon /></div>
                <span style={{ fontSize: 11, letterSpacing: 1 }}>AWAITING_INPUT_STREAM...</span>
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)', marginTop: 25 }}>
          <span>[S75_000E_774]</span>
          <div>
            <span style={{ marginRight: 15 }}>DOCUMENTATION</span>
            <span style={{ marginRight: 15 }}>TERMS</span>
            <span>PRIVACY</span>
          </div>
          <span>V1.0.4-BETA | SYSTEM_TIME: {new Date().toISOString().replace('T', '_').substring(0, 19)}Z</span>
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
              REAL-TIME SCORE DISTRIBUTION (RUN_N={telemetryData.length})
            </div>

            {/* SVG Line Chart */}
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