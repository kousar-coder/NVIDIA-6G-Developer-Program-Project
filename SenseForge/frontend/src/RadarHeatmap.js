import React, { useRef, useEffect, useCallback } from 'react';

/**
 * RadarHeatmap — INFERNO-colormap range-Doppler map rendered via Canvas.
 * White crosshair markers at detection positions.
 * Pulsing amber glow on detection cells.
 */

// INFERNO colormap (12-stop array)
const INFERNO = [
  [0, 0, 4],
  [22, 11, 57],
  [52, 10, 95],
  [85, 15, 109],
  [120, 28, 109],
  [155, 44, 96],
  [188, 63, 73],
  [214, 91, 47],
  [234, 127, 22],
  [245, 168, 12],
  [249, 210, 45],
  [252, 255, 164],
];

function interpolateColor(val) {
  // val in [0, 1]
  const t = Math.max(0, Math.min(1, val)) * (INFERNO.length - 1);
  const idx = Math.floor(t);
  const frac = t - idx;
  const c0 = INFERNO[Math.min(idx, INFERNO.length - 1)];
  const c1 = INFERNO[Math.min(idx + 1, INFERNO.length - 1)];
  return [
    Math.round(c0[0] + (c1[0] - c0[0]) * frac),
    Math.round(c0[1] + (c1[1] - c0[1]) * frac),
    Math.round(c0[2] + (c1[2] - c0[2]) * frac),
  ];
}

function RadarHeatmap({ rdMatrix, detections = [] }) {
  const canvasRef = useRef(null);
  const prevDetRef = useRef([]);
  const glowTimeRef = useRef({});

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#010305';
    ctx.fillRect(0, 0, w, h);

    if (!rdMatrix || !rdMatrix.length) {
      // No data placeholder
      ctx.fillStyle = '#1a2438';
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting RF data...', w / 2, h / 2);
      return;
    }

    const rows = rdMatrix.length;
    const cols = rdMatrix[0] ? rdMatrix[0].length : 0;
    if (rows === 0 || cols === 0) return;

    // Find min/max for normalisation
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = rdMatrix[r][c];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const range = maxVal - minVal || 1;

    // Margins for axis labels
    const mx = 35;
    const my = 20;
    const pw = w - mx - 10;
    const ph = h - my - 25;

    // Render heatmap using ImageData
    const cellW = pw / cols;
    const cellH = ph / rows;

    const imgData = ctx.createImageData(w, h);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = (rdMatrix[r][c] - minVal) / range;
        const [cr, cg, cb] = interpolateColor(val);

        const px0 = Math.floor(mx + c * cellW);
        const py0 = Math.floor(my + r * cellH);
        const px1 = Math.floor(mx + (c + 1) * cellW);
        const py1 = Math.floor(my + (r + 1) * cellH);

        for (let py = py0; py < py1 && py < h; py++) {
          for (let px = px0; px < px1 && px < w; px++) {
            const idx = (py * w + px) * 4;
            imgData.data[idx] = cr;
            imgData.data[idx + 1] = cg;
            imgData.data[idx + 2] = cb;
            imgData.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Detection markers (white crosshairs)
    const now = Date.now();
    detections.forEach((det, i) => {
      const r = det.range_bin || Math.floor((det.range_m / 200) * rows);
      const d = det.doppler_bin || Math.floor(cols / 2 + (det.velocity_mps || 0) * 2);

      const cx = mx + (d + 0.5) * cellW;
      const cy = my + (r + 0.5) * cellH;

      // Check if new detection → glow
      const key = `${Math.round(det.range_m)}_${Math.round((det.velocity_mps || 0) * 10)}`;
      if (!glowTimeRef.current[key]) {
        glowTimeRef.current[key] = now;
      }
      const age = now - glowTimeRef.current[key];

      // Amber glow if recent (< 400ms)
      if (age < 400) {
        const alpha = 1.0 - age / 400;
        ctx.save();
        ctx.globalAlpha = alpha * 0.6;
        ctx.fillStyle = '#f5a518';
        ctx.beginPath();
        ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // White crosshair +
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      const cs = 6;
      ctx.beginPath();
      ctx.moveTo(cx - cs, cy);
      ctx.lineTo(cx + cs, cy);
      ctx.moveTo(cx, cy - cs);
      ctx.lineTo(cx, cy + cs);
      ctx.stroke();
    });

    // Update glow times — remove stale entries
    const keys = Object.keys(glowTimeRef.current);
    if (keys.length > 50) {
      const cutoff = now - 2000;
      keys.forEach(k => {
        if (glowTimeRef.current[k] < cutoff) delete glowTimeRef.current[k];
      });
    }

    // Axis labels
    ctx.fillStyle = '#4a5a70';
    ctx.font = '9px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';

    // Range axis (bottom)
    ctx.fillText('0m', mx, h - 5);
    ctx.fillText('100m', mx + pw / 2, h - 5);
    ctx.fillText('200m', mx + pw, h - 5);

    // Doppler axis label (left, rotated)
    ctx.save();
    ctx.translate(10, my + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Doppler (m/s)', 0, 0);
    ctx.restore();

    // Range label bottom
    ctx.fillText('Range →', mx + pw / 2, h - 14);

  }, [rdMatrix, detections]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Re-draw periodically for glow animations
  useEffect(() => {
    const interval = setInterval(draw, 100);
    return () => clearInterval(interval);
  }, [draw]);

  return (
    <div className="radar-canvas-wrapper">
      <canvas
        ref={canvasRef}
        width={320}
        height={280}
      />

      {/* Legend */}
      <div className="radar-legend">
        <div className="legend-chip">
          <div className="legend-swatch" style={{ background: '#000004' }} />
          <span>Noise</span>
        </div>
        <div className="legend-chip">
          <div className="legend-swatch" style={{ background: '#550f6d' }} />
          <span>Clutter</span>
        </div>
        <div className="legend-chip">
          <div className="legend-swatch" style={{ background: '#d64e12' }} />
          <span>Target</span>
        </div>
        <div className="legend-chip">
          <div className="legend-swatch" style={{ background: '#fcffa4' }} />
          <span>Strong</span>
        </div>
      </div>

      <div className="radar-scale">0 ——— 100m ——— 200m</div>

      {/* RF Callout */}
      <div className={`rf-callout ${detections.length > 0 ? 'active' : ''}`}>
        RF DETECTIONS: {detections.length} · SNR: {
          detections.length > 0
            ? `${Math.max(...detections.map(d => d.snr_db || 0)).toFixed(1)} dB`
            : '—'
        }
      </div>
    </div>
  );
}

export default RadarHeatmap;
