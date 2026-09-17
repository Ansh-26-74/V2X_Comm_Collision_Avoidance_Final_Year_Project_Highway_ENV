/**
 * Phase 10: V2I Smart Intersection Results Dashboard Logic
 * Standalone, zero-dependency browser application.
 * Reads and visualizes benchmark_results.json dynamically.
 */

(function () {
  'use strict';

  // Global state
  let benchmarkData = null;
  let chartInstances = {};
  let currentSort = { column: 'trial_id', ascending: true };

  // DOM Elements
  const container = document.getElementById('dashboardContainer');
  const fileInput = document.getElementById('jsonFileInput');
  const reloadBtn = document.getElementById('reloadBtn');
  const presentationBtn = document.getElementById('presentationBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const fileNoticeBanner = document.getElementById('fileNoticeBanner');
  const errorPanel = document.getElementById('errorPanel');
  const mainContent = document.getElementById('mainDashboardContent');
  const tableBody = document.getElementById('trialsTableBody');
  const tableFilterInput = document.getElementById('tableFilterInput');

  // ───────────────────────────────────────────────────────────────────────────
  // 1. Data Loading & Local File Handling
  // ───────────────────────────────────────────────────────────────────────────

  async function loadData() {
    hideError();
    fileNoticeBanner.style.display = 'none';

    // Attempt 1: Fetch ../benchmark_results.json
    try {
      const resp = await fetch('../benchmark_results.json', { cache: 'no-cache' });
      if (resp.ok) {
        const json = await resp.json();
        handleDataLoaded(json, 'Loaded from ../benchmark_results.json');
        return;
      }
    } catch (e) {
      console.warn('[Dashboard] Relative fetch (../) restricted or failed:', e);
    }

    // Attempt 2: Fetch ./benchmark_results.json (in current folder if copied)
    try {
      const resp = await fetch('benchmark_results.json', { cache: 'no-cache' });
      if (resp.ok) {
        const json = await resp.json();
        handleDataLoaded(json, 'Loaded from ./benchmark_results.json');
        return;
      }
    } catch (e) {
      console.warn('[Dashboard] Local fetch (./) restricted or failed:', e);
    }

    // If fetch failed due to file:// origin restrictions, prompt user to select the file
    fileNoticeBanner.style.display = 'flex';
  }

  function handleDataLoaded(data, sourceLabel) {
    if (!validateSchema(data)) {
      showError('Invalid benchmark data format', 'The loaded file does not match the expected Phase 10 benchmark schema.');
      return;
    }

    benchmarkData = data;
    hideError();
    fileNoticeBanner.style.display = 'none';

    const sourceBadge = document.getElementById('dataSourceBadge');
    if (sourceBadge) {
      sourceBadge.textContent = sourceLabel || 'Source: benchmark_results.json';
    }

    renderDashboard(data);
  }

  function validateSchema(data) {
    return (
      data &&
      data.summary &&
      Array.isArray(data.baseline_trials) &&
      Array.isArray(data.v2i_trials) &&
      data.summary.travel_time &&
      data.summary.signal_wait_time
    );
  }

  // File picker handler
  if (fileInput) {
    fileInput.addEventListener('change', function (e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function (event) {
        try {
          const json = JSON.parse(event.target.result);
          handleDataLoaded(json, `File: ${file.name}`);
        } catch (err) {
          showError('Invalid JSON file', `Error parsing ${file.name}: ${err.message}`);
        }
      };
      reader.onerror = function () {
        showError('File Read Error', 'Could not read the selected file.');
      };
      reader.readAsText(file);
    });
  }

  // Drag and drop JSON file support
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.json')) {
        const reader = new FileReader();
        reader.onload = function (event) {
          try {
            const json = JSON.parse(event.target.result);
            handleDataLoaded(json, `Dropped: ${file.name}`);
          } catch (err) {
            showError('Invalid JSON', err.message);
          }
        };
        reader.readAsText(file);
      }
    }
  });

  function showError(title, msg) {
    if (errorPanel) {
      document.getElementById('errorTitle').textContent = title;
      document.getElementById('errorMessage').textContent = msg;
      errorPanel.style.display = 'flex';
      if (mainContent) mainContent.style.opacity = '0.3';
    }
  }

  function hideError() {
    if (errorPanel) {
      errorPanel.style.display = 'none';
      if (mainContent) mainContent.style.opacity = '1';
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 2. Dashboard Rendering
  // ───────────────────────────────────────────────────────────────────────────

  function renderDashboard(data) {
    const s = data.summary;

    // Header metadata
    setText('metaTotalTrials', `${s.total_trials || 10} Paired`);
    setText('metaSimRuns', `${(s.total_trials || 10) * 2} Total`);

    // 1. KPI Cards
    const tt = s.travel_time;
    setText('kpiBaseTravelTime', formatNum(tt.baseline.mean) + ' s');
    setText('kpiV2iTravelTime', formatNum(tt.v2i.mean) + ' s');
    const ttDiff = tt.difference_mean;
    setText('kpiTravelTimeDelta', `${ttDiff >= 0 ? '↓' : '↑'} ${Math.abs(ttDiff).toFixed(2)} s travel time reduction`);
    setText('kpiTravelTimeReduction', `-${Math.abs(tt.reduction_percent).toFixed(1)}%`);

    const sw = s.signal_wait_time;
    setText('kpiBaseSignalWait', formatNum(sw.baseline.mean) + ' s');
    setText('kpiV2iSignalWait', formatNum(sw.v2i.mean) + ' s');
    const swDiff = sw.difference_mean;
    setText('kpiSignalWaitDelta', `${swDiff >= 0 ? '↓' : '↑'} ${Math.abs(swDiff).toFixed(2)} s wait time reduction`);
    setText('kpiSignalWaitReduction', `-${Math.abs(sw.reduction_percent).toFixed(1)}%`);

    const cl = s.clearance_time;
    setText('kpiBaseClearance', formatNum(cl.baseline.mean) + ' s');
    setText('kpiV2iClearance', formatNum(cl.v2i.mean) + ' s');

    const st = s.total_stops;
    setText('kpiBaseStops', formatNum(st.baseline.mean) + ' stops');
    setText('kpiV2iStops', formatNum(st.v2i.mean) + ' stops');

    const pd = s.preemption_delay;
    if (pd && typeof pd.mean === 'number') {
      setText('kpiPreemptionDelay', pd.mean.toFixed(2));
    }

    // Effect Box Summary
    setText('effTravelVal', `↓ ${Math.abs(ttDiff).toFixed(2)} s`);
    setText('effTravelPct', `${Math.abs(tt.reduction_percent).toFixed(1)}% reduction`);
    setText('effWaitVal', `↓ ${Math.abs(swDiff).toFixed(2)} s`);
    setText('effWaitPct', `${Math.abs(sw.reduction_percent).toFixed(1)}% reduction`);
    setText('effClearVal', `${Math.abs(cl.baseline.mean - cl.v2i.mean).toFixed(2)} s`);
    setText('effStopsVal', `${Math.abs(st.baseline.mean - st.v2i.mean).toFixed(2)}`);

    // 2. Render Charts
    renderCharts(data);

    // 3. Render Table
    renderTable(data);
  }

  function formatNum(val) {
    return (val !== undefined && val !== null) ? Number(val).toFixed(2) : 'N/A';
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 3. Chart Rendering with Chart.js & SVG Fallback
  // ───────────────────────────────────────────────────────────────────────────

  function renderCharts(data) {
    if (window.Chart) {
      renderChartJS(data);
    } else {
      renderSVGFallback(data);
    }
  }

  function renderChartJS(data) {
    const s = data.summary;
    const baseTrials = data.baseline_trials;
    const v2iTrials = data.v2i_trials;

    // Chart 1: Main Comparison Bar Chart
    const ctxMain = document.getElementById('mainComparisonChart')?.getContext('2d');
    if (ctxMain) {
      if (chartInstances.main) chartInstances.main.destroy();

      chartInstances.main = new Chart(ctxMain, {
        type: 'bar',
        data: {
          labels: ['Total Travel Time', 'Signal Waiting Time', 'Intersection Clearance'],
          datasets: [
            {
              label: 'Baseline (No V2I)',
              data: [s.travel_time.baseline.mean, s.signal_wait_time.baseline.mean, s.clearance_time.baseline.mean],
              backgroundColor: 'rgba(148, 163, 184, 0.65)',
              borderColor: '#94a3b8',
              borderWidth: 1.5,
              borderRadius: 4,
            },
            {
              label: 'V2I Enabled',
              data: [s.travel_time.v2i.mean, s.signal_wait_time.v2i.mean, s.clearance_time.v2i.mean],
              backgroundColor: 'rgba(6, 182, 212, 0.75)',
              borderColor: '#06b6d4',
              borderWidth: 1.5,
              borderRadius: 4,
            },
          ],
        },
        options: getChartOptions('Seconds (s)'),
      });
    }

    // Trial Labels (Trial 01 .. Trial 10)
    const trialLabels = baseTrials.map((t, idx) => `Trial ${String(idx + 1).padStart(2, '0')}`);

    // Chart 2: Travel Time Across All Trials
    const ctxTrial = document.getElementById('trialTravelTimeChart')?.getContext('2d');
    if (ctxTrial) {
      if (chartInstances.trial) chartInstances.trial.destroy();

      chartInstances.trial = new Chart(ctxTrial, {
        type: 'bar',
        data: {
          labels: trialLabels,
          datasets: [
            {
              label: 'Baseline Travel Time',
              data: baseTrials.map(t => t.total_travel_time),
              backgroundColor: 'rgba(148, 163, 184, 0.6)',
              borderColor: '#94a3b8',
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: 'V2I Travel Time',
              data: v2iTrials.map(t => t.total_travel_time),
              backgroundColor: 'rgba(6, 182, 212, 0.75)',
              borderColor: '#06b6d4',
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: getChartOptions('Seconds (s)'),
      });
    }

    // Chart 3: Signal Waiting Time Across Trials
    const ctxWait = document.getElementById('trialSignalWaitChart')?.getContext('2d');
    if (ctxWait) {
      if (chartInstances.wait) chartInstances.wait.destroy();

      chartInstances.wait = new Chart(ctxWait, {
        type: 'bar',
        data: {
          labels: trialLabels,
          datasets: [
            {
              label: 'Baseline Signal Wait',
              data: baseTrials.map(t => t.signal_wait_time),
              backgroundColor: 'rgba(148, 163, 184, 0.6)',
              borderColor: '#94a3b8',
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: 'V2I Signal Wait',
              data: v2iTrials.map(t => t.signal_wait_time),
              backgroundColor: 'rgba(56, 189, 248, 0.75)',
              borderColor: '#38bdf8',
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: getChartOptions('Seconds (s)'),
      });
    }

    // Chart 4: Advisory AI Predicted Conflict vs. Kinematic TTC Critical Timeline (Seed 42)
    const ctxAiTtc = document.getElementById('aiVsTtcChart')?.getContext('2d');
    if (ctxAiTtc) {
      if (chartInstances.aiTtc) chartInstances.aiTtc.destroy();

      chartInstances.aiTtc = new Chart(ctxAiTtc, {
        type: 'bar',
        data: {
          labels: [
            '1. AI Predicted Conflict',
            '2. Kinematic TTC Critical',
            '3. Ambulance Evasive Start',
            '4. Lane Maneuver Complete',
          ],
          datasets: [
            {
              label: 'Event Epoch (Seconds in Seed-42 Run)',
              data: [10.93, 11.43, 11.45, 12.25],
              backgroundColor: [
                'rgba(245, 158, 11, 0.85)',  // Amber for AI Early Forecast
                'rgba(239, 68, 68, 0.85)',   // Red for Kinematic TTC Critical
                'rgba(59, 130, 246, 0.85)',  // Blue for Evasive Action Initiated
                'rgba(16, 185, 129, 0.85)',  // Green for Maneuver Secured
              ],
              borderColor: [
                '#f59e0b',
                '#ef4444',
                '#3b82f6',
                '#10b981',
              ],
              borderWidth: 1.5,
              borderRadius: 6,
              barThickness: 24,
            },
          ],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(15, 23, 42, 0.95)',
              titleColor: '#f1f5f9',
              bodyColor: '#cbd5e1',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1,
              padding: 10,
              callbacks: {
                label: function (ctx) {
                  return `Timestamp: ${ctx.raw.toFixed(2)} s`;
                },
                afterLabel: function (ctx) {
                  const descs = [
                    'Advisory conflict forecast (Clearance: 72.8px <= 75px buffer) | +0.50s lead',
                    'Kinematic TTC threshold crossed (TTC = 1.99s <= 2.0s)',
                    'Safety Fusion confirms Lane 2 clear -> shift to x=628.0 px',
                    'Lateral evasion secured -> collision avoided at emergency speed',
                  ];
                  return descs[ctx.dataIndex] || '';
                },
              },
            },
          },
          scales: {
            x: {
              min: 10.0,
              max: 13.0,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8', font: { size: 11 }, stepSize: 0.5 },
              title: { display: true, text: 'Simulation Time (Seconds)', color: '#64748b', font: { size: 11 } },
            },
            y: {
              grid: { display: false },
              ticks: { color: '#cbd5e1', font: { size: 11, weight: 'bold' } },
            },
          },
        },
      });
    }

    // Render interactive trajectory canvas
    renderTrajectoryCanvas();
  }

  function getChartOptions(yAxisTitle) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: '#94a3b8',
            font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto', size: 11 },
            boxWidth: 12,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#94a3b8', font: { size: 11 } },
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8', font: { size: 11 } },
          title: { display: true, text: yAxisTitle, color: '#64748b', font: { size: 11 } },
          beginAtZero: true,
        },
      },
    };
  }

  // Graceful SVG fallback if Chart.js is not loaded (e.g. offline)
  function renderSVGFallback(data) {
    console.info('[Dashboard] Chart.js not loaded. Rendering responsive SVG fallback.');
    const s = data.summary;

    const mainContainer = document.getElementById('mainChartContainer');
    if (mainContainer) {
      mainContainer.innerHTML = `
        <div style="padding: 20px; font-family: var(--font-mono); font-size: 12px; color: #94a3b8;">
          <div style="margin-bottom: 8px;"><strong>Total Travel Time:</strong> Baseline: ${s.travel_time.baseline.mean.toFixed(2)}s | V2I: ${s.travel_time.v2i.mean.toFixed(2)}s (-${s.travel_time.reduction_percent.toFixed(1)}%)</div>
          <div style="margin-bottom: 8px;"><strong>Signal Waiting Time:</strong> Baseline: ${s.signal_wait_time.baseline.mean.toFixed(2)}s | V2I: ${s.signal_wait_time.v2i.mean.toFixed(2)}s (-${s.signal_wait_time.reduction_percent.toFixed(1)}%)</div>
          <div><strong>Intersection Clearance:</strong> Baseline: ${s.clearance_time.baseline.mean.toFixed(2)}s | V2I: ${s.clearance_time.v2i.mean.toFixed(2)}s</div>
        </div>
      `;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 4. Interactive Trial Table
  // ───────────────────────────────────────────────────────────────────────────

  function renderTable(data) {
    if (!tableBody) return;
    tableBody.innerHTML = '';

    const baseTrials = data.baseline_trials;
    const v2iTrials = data.v2i_trials;

    let rowsData = baseTrials.map((b, idx) => {
      const v = v2iTrials[idx] || {};
      const diff_tt = (b.total_travel_time !== null && v.total_travel_time !== null)
        ? (b.total_travel_time - v.total_travel_time)
        : null;
      const diff_sw = (b.signal_wait_time !== null && v.signal_wait_time !== null)
        ? (b.signal_wait_time - v.signal_wait_time)
        : null;

      return {
        trial_id: b.trial_id || (idx + 1),
        seed: b.seed,
        base_tt: b.total_travel_time,
        v2i_tt: v.total_travel_time,
        diff_tt: diff_tt,
        base_sw: b.signal_wait_time,
        v2i_sw: v.signal_wait_time,
        diff_sw: diff_sw,
        stops: `${b.total_stops} / ${v.total_stops || 0}`,
        status: b.status,
      };
    });

    // Filtering
    const query = (tableFilterInput?.value || '').toLowerCase().trim();
    if (query) {
      rowsData = rowsData.filter(r =>
        `trial ${r.trial_id}`.includes(query) ||
        String(r.seed).includes(query) ||
        String(r.status).toLowerCase().includes(query)
      );
    }

    // Sorting
    rowsData.sort((a, b) => {
      let vA = a[currentSort.column];
      let vB = b[currentSort.column];
      if (vA === null) vA = -9999;
      if (vB === null) vB = -9999;
      if (vA < vB) return currentSort.ascending ? -1 : 1;
      if (vA > vB) return currentSort.ascending ? 1 : -1;
      return 0;
    });

    // Populate rows
    rowsData.forEach(r => {
      const tr = document.createElement('tr');

      const ttDiffClass = r.diff_tt > 0.01 ? 'col-diff-pos' : (r.diff_tt < -0.01 ? 'col-diff-neg' : 'col-diff-neutral');
      const swDiffClass = r.diff_sw > 0.01 ? 'col-diff-pos' : (r.diff_sw < -0.01 ? 'col-diff-neg' : 'col-diff-neutral');

      const ttDiffStr = r.diff_tt !== null
        ? (r.diff_tt > 0 ? `-${r.diff_tt.toFixed(2)} s` : (r.diff_tt < 0 ? `+${Math.abs(r.diff_tt).toFixed(2)} s` : '0.00 s'))
        : 'N/A';

      const swDiffStr = r.diff_sw !== null
        ? (r.diff_sw > 0 ? `-${r.diff_sw.toFixed(2)} s` : (r.diff_sw < 0 ? `+${Math.abs(r.diff_sw).toFixed(2)} s` : '0.00 s'))
        : 'N/A';

      tr.innerHTML = `
        <td><strong>#${String(r.trial_id).padStart(2, '0')}</strong></td>
        <td style="color: #64748b;">${r.seed}</td>
        <td>${r.base_tt !== null ? r.base_tt.toFixed(2) + ' s' : 'TIMEOUT'}</td>
        <td style="color: #38bdf8;">${r.v2i_tt !== null ? r.v2i_tt.toFixed(2) + ' s' : 'TIMEOUT'}</td>
        <td class="${ttDiffClass}">${ttDiffStr}</td>
        <td>${r.base_sw !== null ? r.base_sw.toFixed(2) + ' s' : 'N/A'}</td>
        <td style="color: #38bdf8;">${r.v2i_sw !== null ? r.v2i_sw.toFixed(2) + ' s' : 'N/A'}</td>
        <td class="${swDiffClass}">${swDiffStr}</td>
        <td>${r.stops}</td>
        <td><span class="table-badge-completed">${r.status}</span></td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // Setup table sorting clicks
  document.querySelectorAll('#trialsTable th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-sort');
      if (currentSort.column === col) {
        currentSort.ascending = !currentSort.ascending;
      } else {
        currentSort.column = col;
        currentSort.ascending = true;
      }
      if (benchmarkData) renderTable(benchmarkData);
    });
  });

  if (tableFilterInput) {
    tableFilterInput.addEventListener('input', () => {
      if (benchmarkData) renderTable(benchmarkData);
    });
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 5. Presentation Mode & Fullscreen Controls
  // ───────────────────────────────────────────────────────────────────────────

  if (reloadBtn) {
    reloadBtn.addEventListener('click', () => loadData());
  }

  if (presentationBtn) {
    presentationBtn.addEventListener('click', () => {
      const isPres = document.body.classList.toggle('presentation-mode');
      presentationBtn.innerHTML = isPres
        ? '<span class="btn-icon">✖</span> Exit Presentation'
        : '<span class="btn-icon">📺</span> Presentation Mode';

      // Re-render charts so Chart.js resizes cleanly
      setTimeout(() => {
        if (benchmarkData) renderCharts(benchmarkData);
      }, 100);
    });
  }

  if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
      }
    });
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 6. Interactive Trajectory Visualizer Canvas
  // ───────────────────────────────────────────────────────────────────────────

  const waypointsData = [
    { num: 1, dt: 0.25, simX: 592.0, simY: 737.0, canvasY: 104, deltaY: 26.0 },
    { num: 2, dt: 0.50, simX: 592.0, simY: 710.5, canvasY: 86,  deltaY: 52.5 },
    { num: 3, dt: 0.75, simX: 591.7, simY: 684.4, canvasY: 68,  deltaY: 78.6 },
    { num: 4, dt: 1.00, simX: 591.8, simY: 658.7, canvasY: 50,  deltaY: 104.3 },
    { num: 5, dt: 1.25, simX: 591.9, simY: 633.2, canvasY: 32,  deltaY: 129.8 },
    { num: 6, dt: 1.50, simX: 591.8, simY: 608.1, canvasY: 15,  deltaY: 154.9, isFinal: true },
  ];

  let hoveredWaypoint = null;

  function renderTrajectoryCanvas() {
    const canvas = document.getElementById('trajectoryCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = 460;
    const h = 240;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    // Background: dark asphalt
    ctx.fillStyle = '#080c16';
    ctx.fillRect(0, 0, w, h);

    // Roadway boundaries (Southbound Corridor)
    const roadLeft = 40;
    const roadRight = 420;
    const roadMid = (roadLeft + roadRight) / 2; // 230px
    const lane1X = 145; // Primary Lane (C-01 & AMB initial)
    const lane2X = 315; // Passing / Evasive Lane

    // Asphalt surface
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(roadLeft, 0, roadRight - roadLeft, h);

    // Road edge curbs
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(roadLeft, 0); ctx.lineTo(roadLeft, h);
    ctx.moveTo(roadRight, 0); ctx.lineTo(roadRight, h);
    ctx.stroke();

    // Center dashed lane divider
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(roadMid, 0);
    ctx.lineTo(roadMid, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Lane Labels
    ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('LANE 1 (PRIMARY x=592)', lane1X, h - 8);
    ctx.fillText('LANE 2 (PASSING x=628)', lane2X, h - 8);

    // Intersection entrance line at top (y = 8)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(roadLeft, 8); ctx.lineTo(roadRight, 8);
    ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.font = '9px monospace';
    ctx.fillText('STOP LINE / INTERSECTION BOX ENTRANCE (y=490 px)', roadMid, 6);

    // Projected Ambulance Evasive Path (Lane 1 -> Lane 2)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(lane1X, 205);
    ctx.bezierCurveTo(lane1X, 160, lane2X, 150, lane2X, 70);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#60a5fa';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('↗ Evasive Shift to Lane 2 (x=628.0)', lane2X - 55, 115);

    // Draw Ambulance AMB-01 (Approaching Behind)
    const ambY = 195;
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    roundRect(ctx, lane1X - 11, ambY, 22, 34, 4, true, true);

    // Red side stripe on ambulance
    ctx.fillStyle = '#ef4444';
    ctx.fillRect(lane1X - 11, ambY + 6, 2, 22);
    ctx.fillRect(lane1X + 9, ambY + 6, 2, 22);

    // Lightbar (flashing strobe)
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(lane1X - 6, ambY + 4, 12, 3);

    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('AMB-01', lane1X, ambY + 20);

    // Safety Bumper Clearance Buffer (75px threshold)
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(lane1X - 16, 120, 32, ambY - 120);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.7)';
    ctx.font = '9px monospace';
    ctx.fillText('Buffer 75px', lane1X - 35, 160);

    // Draw Lead Civilian Vehicle C-01
    const c01Y = 120;
    ctx.fillStyle = '#334155';
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    roundRect(ctx, lane1X - 10, c01Y, 20, 30, 4, true, true);

    // C-01 Windshield & Lights
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(lane1X - 7, c01Y + 4, 14, 6);
    ctx.fillStyle = '#cbd5e1';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('C-01', lane1X, c01Y + 20);

    // AI Predicted Trajectory Line connecting 6 waypoints
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(lane1X, c01Y);
    waypointsData.forEach(wp => {
      ctx.lineTo(lane1X, wp.canvasY);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw the 6 Waypoint Markers
    waypointsData.forEach(wp => {
      const isHov = hoveredWaypoint === wp.num;
      const r = wp.isFinal ? 5 : (isHov ? 5 : 3.5);

      // Outer glow halo
      ctx.fillStyle = wp.isFinal
        ? 'rgba(168, 85, 247, 0.35)'
        : (isHov ? 'rgba(6, 182, 212, 0.5)' : 'rgba(6, 182, 212, 0.2)');
      ctx.beginPath();
      ctx.arc(lane1X, wp.canvasY, r + 4, 0, Math.PI * 2);
      ctx.fill();

      // Core waypoint circle
      ctx.fillStyle = wp.isFinal ? '#c084fc' : '#38bdf8';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(lane1X, wp.canvasY, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Text labels beside waypoint
      ctx.fillStyle = wp.isFinal ? '#c084fc' : (isHov ? '#ffffff' : '#94a3b8');
      ctx.font = isHov ? 'bold 10px monospace' : '9px monospace';
      ctx.textAlign = 'left';
      const label = wp.isFinal
        ? `WP#${wp.num} (+${wp.dt.toFixed(2)}s AI HORIZON)`
        : `WP#${wp.num} (+${wp.dt.toFixed(2)}s)`;
      ctx.fillText(label, lane1X + 10, wp.canvasY + 3);
    });

    // Tooltip if hovering over a waypoint
    if (hoveredWaypoint !== null) {
      const activeWp = waypointsData.find(w => w.num === hoveredWaypoint);
      if (activeWp) {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1;
        const boxX = lane1X + 130;
        const boxY = Math.max(10, activeWp.canvasY - 20);
        roundRect(ctx, boxX, boxY, 150, 42, 4, true, true);

        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`WAYPOINT #${activeWp.num} (+${activeWp.dt.toFixed(2)}s)`, boxX + 8, boxY + 14);

        ctx.fillStyle = '#f1f5f9';
        ctx.font = '9px monospace';
        ctx.fillText(`Sim: (${activeWp.simX}, ${activeWp.simY}) px`, boxX + 8, boxY + 26);
        ctx.fillText(`Forward Delta: +${activeWp.deltaY} px`, boxX + 8, boxY + 37);
      }
    }
  }

  function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  // Interactive hover inspection on Trajectory Canvas
  const trajCanvas = document.getElementById('trajectoryCanvas');
  if (trajCanvas) {
    trajCanvas.addEventListener('mousemove', (e) => {
      const rect = trajCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let found = null;
      waypointsData.forEach(wp => {
        // Distance check to waypoint (lane1X = 145)
        const dx = mouseX - 145;
        const dy = mouseY - wp.canvasY;
        if (Math.hypot(dx, dy) <= 16) {
          found = wp.num;
        }
      });

      if (found !== hoveredWaypoint) {
        hoveredWaypoint = found;
        renderTrajectoryCanvas();
      }
    });

    trajCanvas.addEventListener('mouseleave', () => {
      if (hoveredWaypoint !== null) {
        hoveredWaypoint = null;
        renderTrajectoryCanvas();
      }
    });
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 7. Sticky Sub-Navigation & Scroll Spy
  // ───────────────────────────────────────────────────────────────────────────

  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const targetId = link.getAttribute('href');
      if (targetId && targetId.startsWith('#')) {
        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          navLinks.forEach(l => l.classList.remove('active'));
          link.classList.add('active');
        }
      }
    });
  });

  window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('.section-container, #section-trials-table');
    const scrollPos = window.scrollY + 160;
    let currentId = '';

    sections.forEach(sec => {
      if (sec.offsetTop <= scrollPos) {
        currentId = sec.getAttribute('id');
      }
    });

    if (currentId) {
      navLinks.forEach(link => {
        if (link.getAttribute('href') === `#${currentId}`) {
          link.classList.add('active');
        } else {
          link.classList.remove('active');
        }
      });
    }
  });

  // Window resize handler for canvas & charts
  window.addEventListener('resize', () => {
    renderTrajectoryCanvas();
  });

  // Initialize on load
  document.addEventListener('DOMContentLoaded', () => {
    loadData();
    renderTrajectoryCanvas();
  });
})();

