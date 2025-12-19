function byId(id) {
  return document.getElementById(id);
}

function parseOptionalInt(value) {
  if (value === null || value === undefined) return null;
  const s = String(value).trim();
  if (s.length === 0) return null;
  const n = Number.parseInt(s, 10);
  return Number.isFinite(n) ? n : null;
}

function parseOptionalFloat(value) {
  if (value === null || value === undefined) return null;
  const s = String(value).trim();
  if (s.length === 0) return null;
  const n = Number.parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

function buildRequest() {
  return {
    impl: "python",
    preset: "bench",
    k: parseOptionalInt(byId("k").value),
  };
}

function setStatus(which, text) {
  byId(which === "python" ? "statusPy" : "statusCpp").textContent = text;
}

function setMetrics(which, html) {
  byId(which === "python" ? "metricsPy" : "metricsCpp").innerHTML = html;
}

function setComparison(text) {
  const el = byId("comparison");
  if (!el) return;
  el.textContent = text;
}

function clearPlot(canvasId) {
  const canvas = byId(canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawScatter(canvasId, X2, labels, centers2) {
  const canvas = byId(canvasId);
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!X2 || X2.length === 0) return;

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;

  for (let i = 0; i < X2.length; i++) {
    const x = X2[i][0];
    const y = X2[i][1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const pad = 18;
  const w = canvas.width;
  const h = canvas.height;

  const sx = (v) => {
    if (maxX === minX) return w * 0.5;
    return pad + ((v - minX) / (maxX - minX)) * (w - 2 * pad);
  };

  const sy = (v) => {
    if (maxY === minY) return h * 0.5;
    return h - (pad + ((v - minY) / (maxY - minY)) * (h - 2 * pad));
  };

  function colorFor(label) {
    const hue = (label * 47) % 360;
    return `hsla(${hue}, 55%, 45%, 0.65)`;
  }

  for (let i = 0; i < X2.length; i++) {
    const x = sx(X2[i][0]);
    const y = sy(X2[i][1]);
    const lab = labels[i];
    ctx.fillStyle = colorFor(lab);
    ctx.beginPath();
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  if (centers2 && centers2.length > 0) {
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 2;
    for (let c = 0; c < centers2.length; c++) {
      const x = sx(centers2[c][0]);
      const y = sy(centers2[c][1]);
      ctx.beginPath();
      ctx.moveTo(x - 6, y - 6);
      ctx.lineTo(x + 6, y + 6);
      ctx.moveTo(x - 6, y + 6);
      ctx.lineTo(x + 6, y - 6);
      ctx.stroke();
    }
  }
}

async function runBoth() {
  const runBtn = byId("run");
  runBtn.disabled = true;

  clearPlot("plotPy");
  clearPlot("plotCpp");
  setStatus("python", "Running…");
  setStatus("cpp", "Running…");
  setMetrics("python", "runtime: —");
  setMetrics("cpp", "runtime: —");
  setComparison("—");

  const state = {
    python: { startMs: null, elapsed: null, running: false, ok: null },
    cpp: { startMs: null, elapsed: null, running: false, ok: null },
  };

  function pctFaster(fastS, slowS) {
    return (((slowS - fastS) / slowS) * 100).toFixed(1);
  }

  function updateComparison(nowMs = performance.now()) {
    const py = state.python;
    const cpp = state.cpp;

    const pyNow =
      py.elapsed !== null
        ? py.elapsed
        : py.running && py.startMs !== null
          ? (nowMs - py.startMs) / 1000
          : null;
    const cppNow =
      cpp.elapsed !== null
        ? cpp.elapsed
        : cpp.running && cpp.startMs !== null
          ? (nowMs - cpp.startMs) / 1000
          : null;

    if (py.elapsed !== null && cpp.elapsed !== null) {
      if (!Number.isFinite(py.elapsed) || !Number.isFinite(cpp.elapsed) || py.elapsed <= 0 || cpp.elapsed <= 0) {
        setComparison("—");
        return;
      }

      if (cpp.elapsed < py.elapsed) {
        setComparison(`C++ responding ${pctFaster(cpp.elapsed, py.elapsed)}% faster than Python implementation`);
      } else if (py.elapsed < cpp.elapsed) {
        setComparison(`Python responding ${pctFaster(py.elapsed, cpp.elapsed)}% faster than C++ implementation`);
      } else {
        setComparison("Same runtime");
      }
      return;
    }

    if (cpp.elapsed !== null && py.elapsed === null && py.running && pyNow !== null) {
      if (pyNow > cpp.elapsed && Number.isFinite(pyNow) && pyNow > 0 && cpp.elapsed > 0) {
        setComparison(`C++ responding ${pctFaster(cpp.elapsed, pyNow)}% faster than Python so far (Python still running…)`);
      } else {
        setComparison("C++ finished; Python still running…");
      }
      return;
    }

    if (py.elapsed !== null && cpp.elapsed === null && cpp.running && cppNow !== null) {
      if (cppNow > py.elapsed && Number.isFinite(cppNow) && cppNow > 0 && py.elapsed > 0) {
        setComparison(`Python responding ${pctFaster(py.elapsed, cppNow)}% faster than C++ so far (C++ still running…)`);
      } else {
        setComparison("Python finished; C++ still running…");
      }
      return;
    }

    setComparison("—");
  }

  function startTicker(which) {
    const start = performance.now();
    state[which].startMs = start;
    state[which].running = true;
    let cancelled = false;
    function tick() {
      if (cancelled) return;
      const sec = (performance.now() - start) / 1000;
      setStatus(which, `${sec.toFixed(2)} s`);
      updateComparison();
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    return () => {
      cancelled = true;
    };
  }

  async function runImpl(which) {
    const cancel = startTicker(which);
    try {
      const req = buildRequest();
      req.impl = which;

      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });

      const json = await resp.json();
      if (!resp.ok) {
        throw new Error(json.detail || "Request failed");
      }

      setStatus(which, `${json.elapsed_s.toFixed(4)} s`);
      setMetrics(which, `runtime: <code>${json.elapsed_s.toFixed(6)} s</code>`);
      state[which].elapsed = json.elapsed_s;
      state[which].running = false;
      state[which].ok = true;
      updateComparison();
      const canvasId = which === "python" ? "plotPy" : "plotCpp";
      drawScatter(canvasId, json.plot.X2, json.plot.labels, json.plot.centers2);
      return { ok: true };
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      setStatus(which, "Error");
      setMetrics(which, `<span class="error">${msg}</span>`);
      state[which].elapsed = null;
      state[which].running = false;
      state[which].ok = false;
      updateComparison();
      return { ok: false, error: msg };
    } finally {
      cancel();
    }
  }

  await Promise.all([runImpl("python"), runImpl("cpp")]);
  runBtn.disabled = false;
}

byId("run").addEventListener("click", () => {
  runBoth();
});

clearPlot("plotPy");
clearPlot("plotCpp");
