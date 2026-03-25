import React, { useState, useEffect, useRef } from 'react';

/**
 * FusionPanel — Sensor fusion info panel with:
 *   - Confidence gauges (camera / RF / fusion)
 *   - Live detection counts 2×2 grid
 *   - Camera visibility mode buttons
 *   - Intensity slider
 *   - Detection log with slide-in rows
 */

const MODES = [
  { key: 'clear', label: 'Clear', cls: 'clear' },
  { key: 'fog', label: 'Fog', cls: 'fog' },
  { key: 'night', label: 'Night', cls: 'night' },
  { key: 'occlusion', label: 'Occlude', cls: 'occ' },
  { key: 'rain', label: 'Rain', cls: 'rain' },
];

function FusionPanel({
  detections = [],
  cameraConfidence = 1.0,
  rfConfidence = 1.0,
  degradeMode = 'clear',
  degradeIntensity = 0.7,
  fusedCount = 0,
  visionCount = 0,
  rfCount = 0,
  totalCount = 0,
  onSetDegrade,
  onSetIntensity,
}) {
  const [log, setLog] = useState([]);
  const logCounterRef = useRef(0);

  // Build detection log (max 60 entries)
  useEffect(() => {
    if (!detections || detections.length === 0) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-GB', { hour12: false });

    const newEntries = detections.map((d) => {
      logCounterRef.current += 1;
      return {
        id: logCounterRef.current,
        trackId: d.track_id || '—',
        source: d.source || 'none',
        range: (d.fused_range_m || d.range_m || 0).toFixed(1),
        velocity: (d.fused_velocity_mps || d.velocity_mps || 0).toFixed(1),
        confidence: (d.fused_confidence || d.confidence || 0).toFixed(2),
        time: timeStr,
      };
    });

    setLog((prev) => {
      const combined = [...newEntries, ...prev];
      return combined.slice(0, 60);
    });
  }, [detections]);

  const fusionConf = Math.max(cameraConfidence, rfConfidence);

  return (
    <div className="fusion-scroll">
      {/* ── Confidence Gauges ─────────────────────────────────── */}
      <div className="gauge-group">
        <div className="gauge-row">
          <span className="gauge-label">Camera</span>
          <div className="gauge-track">
            <div
              className="gauge-fill camera"
              style={{ width: `${cameraConfidence * 100}%` }}
            />
          </div>
          <span className="gauge-value">{(cameraConfidence * 100).toFixed(0)}%</span>
        </div>
        <div className="gauge-row">
          <span className="gauge-label">RF</span>
          <div className="gauge-track">
            <div
              className="gauge-fill rf"
              style={{ width: `${rfConfidence * 100}%` }}
            />
          </div>
          <span className="gauge-value">{(rfConfidence * 100).toFixed(0)}%</span>
        </div>
        <div className="gauge-row">
          <span className="gauge-label">Fusion</span>
          <div className="gauge-track">
            <div
              className="gauge-fill fusion"
              style={{ width: `${fusionConf * 100}%` }}
            />
          </div>
          <span className="gauge-value">{(fusionConf * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* ── Live Counts ──────────────────────────────────────── */}
      <div className="counts-grid">
        <div className="count-card fused">
          <div className="count-value">{fusedCount}</div>
          <div className="count-label">Fused</div>
        </div>
        <div className="count-card vision">
          <div className="count-value">{visionCount}</div>
          <div className="count-label">Vision</div>
        </div>
        <div className="count-card rf">
          <div className="count-value">{rfCount}</div>
          <div className="count-label">RF Only</div>
        </div>
        <div className="count-card total">
          <div className="count-value">{totalCount}</div>
          <div className="count-label">Total</div>
        </div>
      </div>

      {/* ── Camera Visibility ────────────────────────────────── */}
      <div className="section-label">Camera Visibility</div>
      <div className="vis-buttons">
        {MODES.map((m) => (
          <button
            key={m.key}
            className={`vis-btn ${m.cls} ${degradeMode === m.key ? 'active' : ''}`}
            onClick={() => onSetDegrade && onSetDegrade(m.key)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* ── Intensity Slider ─────────────────────────────────── */}
      <div className="slider-row">
        <span className="gauge-label" style={{ width: 'auto' }}>Intensity</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={degradeIntensity}
          onChange={(e) => onSetIntensity && onSetIntensity(parseFloat(e.target.value))}
        />
        <span className="slider-value">{(degradeIntensity * 100).toFixed(0)}%</span>
      </div>

      {/* ── Detection Log ────────────────────────────────────── */}
      <div className="section-label" style={{ marginTop: 6 }}>Detection Log</div>
      <div style={{ maxHeight: 200, overflow: 'auto' }}>
        <table className="log-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Source</th>
              <th>Range</th>
              <th>Vel</th>
              <th>Conf</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {log.map((entry) => (
              <tr key={entry.id} className="log-row">
                <td>{entry.trackId}</td>
                <td>
                  <span className={`source-badge ${entry.source}`}>
                    {entry.source === 'fused' ? 'FUS' :
                     entry.source === 'rf_only' ? 'RF' :
                     entry.source === 'vision_only' ? 'VIS' : '—'}
                  </span>
                </td>
                <td>{entry.range}m</td>
                <td>{entry.velocity}</td>
                <td>{entry.confidence}</td>
                <td>{entry.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default FusionPanel;
