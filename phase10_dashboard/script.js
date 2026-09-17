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

  // Initialize on load
  document.addEventListener('DOMContentLoaded', loadData);
})();
