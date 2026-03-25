import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import RadarHeatmap from './RadarHeatmap';
import CameraFeed from './CameraFeed';
import FusionPanel from './FusionPanel';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const WS_BASE = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

function App() {
  // ── State ────────────────────────────────────────────────────
  const [frameB64, setFrameB64] = useState(null);
  const [rdMatrix, setRdMatrix] = useState(null);
  const [radarDetections, setRadarDetections] = useState([]);
  const [detections, setDetections] = useState([]);
  const [degradeMode, setDegradeMode] = useState('clear');
  const [degradeIntensity, setDegradeIntensity] = useState(0.7);
  const [cameraConfidence, setCameraConfidence] = useState(1.0);
  const [rfConfidence, setRfConfidence] = useState(1.0);
  const [backendOk, setBackendOk] = useState(false);
  const [nTargets, setNTargets] = useState(2);
  const [uptime, setUptime] = useState(0);
  const [activeTab, setActiveTab] = useState('predictor');

  const wsVideoRef = useRef(null);
  const wsRadarRef = useRef(null);
  const wsDetRef = useRef(null);
  const uptimeRef = useRef(0);

  // ── Health polling ───────────────────────────────────────────
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const res = await axios.get(`${BACKEND}/health`);
        setBackendOk(res.data.status === 'online');
        if (res.data.uptime_s) uptimeRef.current = res.data.uptime_s;
      } catch {
        setBackendOk(false);
      }
    }, 5000);
    return () => clearInterval(poll);
  }, []);

  // ── Uptime counter ──────────────────────────────────────────
  useEffect(() => {
    const timer = setInterval(() => {
      uptimeRef.current += 1;
      setUptime(uptimeRef.current);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // ── WebSocket connector with auto-reconnect ─────────────────
  const connectWS = useCallback((path, onMessage) => {
    let ws;
    let reconnectTimer;
    let alive = true;

    const connect = () => {
      if (!alive) return;
      ws = new WebSocket(`${WS_BASE}${path}`);

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          onMessage(data);
        } catch { /* ignore parse errors */ }
      };

      ws.onclose = () => {
        if (alive) reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      alive = false;
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  // ── Connect WebSockets ──────────────────────────────────────
  useEffect(() => {
    const cleanupVideo = connectWS('/ws/video', (data) => {
      if (data.frame) setFrameB64(data.frame);
    });
    const cleanupRadar = connectWS('/ws/radar', (data) => {
      if (data.rd_matrix) setRdMatrix(data.rd_matrix);
      if (data.detections) setRadarDetections(data.detections);
    });
    const cleanupDet = connectWS('/ws/detections', (data) => {
      if (data.detections) setDetections(data.detections);
      if (data.mode) setDegradeMode(data.mode);
      if (data.camera_confidence !== undefined) setCameraConfidence(data.camera_confidence);
      if (data.rf_confidence !== undefined) setRfConfidence(data.rf_confidence);
    });

    return () => {
      cleanupVideo();
      cleanupRadar();
      cleanupDet();
    };
  }, [connectWS]);

  // ── Actions ─────────────────────────────────────────────────
  const setTargets = async (n) => {
    setNTargets(n);
    try {
      await axios.post(`${BACKEND}/scenario`, { n_targets: n, seed: 42 });
    } catch { /* backend may be offline */ }
  };

  const setDegrade = async (mode) => {
    setDegradeMode(mode);
    try {
      const res = await axios.post(`${BACKEND}/degrade`, {
        mode,
        intensity: degradeIntensity,
      });
      if (res.data.camera_confidence !== undefined) {
        setCameraConfidence(res.data.camera_confidence);
      }
    } catch { /* ignore */ }
  };

  const setIntensity = async (val) => {
    setDegradeIntensity(val);
    try {
      const res = await axios.post(`${BACKEND}/degrade`, {
        mode: degradeMode,
        intensity: val,
      });
      if (res.data.camera_confidence !== undefined) {
        setCameraConfidence(res.data.camera_confidence);
      }
    } catch { /* ignore */ }
  };

  // ── Derived counts ──────────────────────────────────────────
  const fusedCount = detections.filter(d => d.source === 'fused').length;
  const visionCount = detections.filter(d => d.source === 'vision_only').length;
  const rfCount = detections.filter(d => d.source === 'rf_only').length;
  const totalCount = detections.length;

  const formatUptime = (s) => {
    const total = Math.floor(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  // ── Render ──────────────────────────────────────────────────
  return (
    <div className="app-container">
      {/* ── HEADER ──────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">SF</div>
          <div>
            <div className="header-title">SenseForge</div>
            <div className="header-subtitle">Multimodal ISAC Console</div>
          </div>
        </div>

        <div className="header-stats">
          <div className="stat-pill">
            <div className={`status-dot ${backendOk ? '' : 'offline'}`} />
            <span className="stat-value">{backendOk ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
          <div className="stat-pill">
            <span className="stat-label">Uptime</span>
            <span className="stat-value">{formatUptime(uptime)}</span>
          </div>
          <div className="stat-pill">
            <span className="stat-label">Targets</span>
            <span className="stat-value">{nTargets}</span>
          </div>
          <div className="stat-pill">
            <span className="stat-label">Detections</span>
            <span className="stat-value">{totalCount}</span>
          </div>
          <div className="stat-pill">
            <span className="stat-label">Mode</span>
            <span className="stat-value" style={{
              color: degradeMode === 'clear' ? 'var(--accent-green)' : 'var(--accent-red)'
            }}>
              {degradeMode.toUpperCase()}
            </span>
          </div>
        </div>
      </header>

      {/* ── CONTROLS ────────────────────────────────────────── */}
      <div className="controls-row">
        <span style={{ fontSize: '9px', color: '#4a5a70', marginRight: 8, textTransform: 'uppercase', letterSpacing: '1px' }}>
          Targets
        </span>
        <div className="target-buttons">
          {[1, 2, 3, 4].map((n) => (
            <button
              key={n}
              className={`target-btn ${nTargets === n ? 'active' : ''}`}
              onClick={() => setTargets(n)}
            >
              {n}
            </button>
          ))}
        </div>

        <div className="tab-bar">
          <button
            className={`tab-btn ${activeTab === 'predictor' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictor')}
          >
            Predictor
          </button>
          <button
            className={`tab-btn ${activeTab === 'monitor' ? 'active' : ''}`}
            onClick={() => setActiveTab('monitor')}
          >
            Live Monitor
          </button>
        </div>
      </div>

      {/* ── MAIN GRID ───────────────────────────────────────── */}
      <div className="main-grid">
        {/* Column 1: Camera Feed */}
        <div className="panel" id="camera-panel">
          <div className="panel-header">
            <span className="panel-title">Camera Feed</span>
            <span className="panel-badge live">● Live</span>
          </div>
          <div className="panel-body">
            <CameraFeed
              frameB64={frameB64}
              detections={detections}
              degradeMode={degradeMode}
              nTargets={nTargets}
            />
          </div>
        </div>

        {/* Column 2: Radar */}
        <div className="panel" id="radar-panel">
          <div className="panel-header">
            <span className="panel-title">Range-Doppler</span>
            <span className="panel-badge live">● RF</span>
          </div>
          <div className="panel-body">
            <RadarHeatmap
              rdMatrix={rdMatrix}
              detections={radarDetections}
            />
          </div>
        </div>

        {/* Column 3: Fusion */}
        <div className="panel" id="fusion-panel">
          <div className="panel-header">
            <span className="panel-title">Fusion</span>
            <span className="panel-badge live">● AI</span>
          </div>
          <div className="panel-body">
            <FusionPanel
              detections={detections}
              cameraConfidence={cameraConfidence}
              rfConfidence={rfConfidence}
              degradeMode={degradeMode}
              degradeIntensity={degradeIntensity}
              fusedCount={fusedCount}
              visionCount={visionCount}
              rfCount={rfCount}
              totalCount={totalCount}
              onSetDegrade={setDegrade}
              onSetIntensity={setIntensity}
            />
          </div>
        </div>
      </div>

      {/* ── FOOTER ──────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-left">
          <span className="nvidia-badge">NVIDIA AI AERIAL</span>
          <span>5G NR FR1 n78 · μ=1 · 30 kHz SCS</span>
        </div>
        <div className="footer-right">
          <span>SenseForge v1.0 · ISAC Sensor Fusion</span>
          <span>Sionna + cuPHY + YOLOv8</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
