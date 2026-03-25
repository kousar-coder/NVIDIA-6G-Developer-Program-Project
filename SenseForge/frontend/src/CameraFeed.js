import React, { useRef, useEffect, useCallback } from 'react';

/**
 * CameraFeed — Canvas-rendered camera feed with corner-bracket detection boxes.
 * Draws animated stroke L-brackets, scan overlay, and source-colour labels.
 */

function CameraFeed({ frameB64, detections = [], degradeMode = 'clear', nTargets = 2 }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const frameLoadedRef = useRef(false);

  // Source colors
  const SOURCE_COLORS = {
    fused: '#00ff88',
    rf_only: '#38b6ff',
    vision_only: '#f5c518',
    none: '#666666',
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    if (!frameLoadedRef.current) {
      // Loading skeleton
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, '#05070d');
      grad.addColorStop(0.5, '#0a0f1a');
      grad.addColorStop(1, '#05070d');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#1a2438';
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting camera feed...', w / 2, h / 2);
      return;
    }

    // Draw video frame
    ctx.drawImage(imgRef.current, 0, 0, w, h);

    // Scanline overlay
    ctx.fillStyle = 'rgba(0, 255, 136, 0.012)';
    for (let y = 0; y < h; y += 4) {
      ctx.fillRect(0, y, w, 2);
    }

    // Scale factor (image might be different size than canvas)
    const imgW = imgRef.current.naturalWidth || 640;
    const imgH = imgRef.current.naturalHeight || 480;
    const sx = w / imgW;
    const sy = h / imgH;

    // Corner-bracket detection boxes
    detections.forEach((det) => {
      if (!det.bbox) return;

      const x1 = det.bbox[0] * sx;
      const y1 = det.bbox[1] * sy;
      const x2 = det.bbox[2] * sx;
      const y2 = det.bbox[3] * sy;

      const color = SOURCE_COLORS[det.source] || '#666';
      const L = Math.min(18, (x2 - x1) / 3, (y2 - y1) / 3);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = 'square';

      // Top-left corner
      ctx.beginPath();
      ctx.moveTo(x1 + L, y1);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1, y1 + L);
      ctx.stroke();

      // Top-right corner
      ctx.beginPath();
      ctx.moveTo(x2 - L, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + L);
      ctx.stroke();

      // Bottom-left corner
      ctx.beginPath();
      ctx.moveTo(x1 + L, y2);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1, y2 - L);
      ctx.stroke();

      // Bottom-right corner
      ctx.beginPath();
      ctx.moveTo(x2 - L, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - L);
      ctx.stroke();

      // Label pill
      const label = `${(det.source || '').toUpperCase()} ${(det.fused_range_m || det.range_m || 0).toFixed(0)}m`;
      ctx.font = '9px "IBM Plex Mono", monospace';
      const tm = ctx.measureText(label);
      const pw = tm.width + 8;
      const ph = 14;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x1, y1 - ph - 3, pw, ph);
      ctx.globalAlpha = 1.0;

      ctx.fillStyle = '#020509';
      ctx.fillText(label, x1 + 4, y1 - 6);
    });

    // Mode badge (bottom-left)
    const modeText = degradeMode.toUpperCase();
    const isDegraded = degradeMode !== 'clear';

    ctx.font = '10px "IBM Plex Mono", monospace';
    const badgeW = ctx.measureText(modeText).width + 16;
    const badgeH = 20;
    const badgeX = 10;
    const badgeY = h - badgeH - 10;

    ctx.globalAlpha = 0.85;
    ctx.fillStyle = isDegraded ? 'rgba(255, 68, 68, 0.2)' : 'rgba(0, 255, 136, 0.15)';
    ctx.strokeStyle = isDegraded ? '#ff4444' : '#00ff88';
    ctx.lineWidth = 1;
    ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
    ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);
    ctx.globalAlpha = 1.0;

    ctx.fillStyle = isDegraded ? '#ff4444' : '#00ff88';
    ctx.fillText(modeText, badgeX + 8, badgeY + 14);

    // Target count (top-right)
    const countText = `TGT: ${nTargets}`;
    ctx.font = '10px "IBM Plex Mono", monospace';
    const countW = ctx.measureText(countText).width + 12;
    ctx.fillStyle = 'rgba(5, 7, 13, 0.8)';
    ctx.fillRect(w - countW - 10, 8, countW, 18);
    ctx.fillStyle = '#8090a8';
    ctx.fillText(countText, w - countW - 4, 21);

  }, [frameB64, detections, degradeMode, nTargets]);

  // Load image when frameB64 changes
  useEffect(() => {
    if (!frameB64) {
      frameLoadedRef.current = false;
      draw();
      return;
    }

    const img = imgRef.current;
    img.onload = () => {
      frameLoadedRef.current = true;
      draw();
    };
    img.src = `data:image/jpeg;base64,${frameB64}`;
  }, [frameB64, draw]);

  // Redraw on detection updates
  useEffect(() => {
    if (frameLoadedRef.current) draw();
  }, [detections, draw]);

  return (
    <div className="feed-wrapper">
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
      />
      {/* CSS scanline overlay is applied via ::after in CSS */}
    </div>
  );
}

export default CameraFeed;
