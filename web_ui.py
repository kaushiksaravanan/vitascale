"""Shared HTML dashboard for VitaScale visualization."""

from __future__ import annotations

import json
from typing import Sequence

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VitaScale Dashboard</title>
  <style>
    :root {
      --bg: #08111f;
      --panel: #0f1b2d;
      --panel-2: #14243b;
      --text: #e5eefb;
      --muted: #9cb0ca;
      --accent: #60a5fa;
      --accent-2: #34d399;
      --warning: #fbbf24;
      --danger: #f87171;
      --border: rgba(148, 163, 184, 0.18);
      --shadow: 0 18px 42px rgba(8, 17, 31, 0.35);
      --radius: 18px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.15), transparent 25%),
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 30%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }

    .panel {
      background: linear-gradient(180deg, rgba(20, 36, 59, 0.96), rgba(15, 27, 45, 0.96));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
    }

    .title {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }

    .title h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 700;
    }

    .subtitle {
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }

    .badge-row, .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }

    .badge, .status-pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.04);
    }

    .status-pill strong { color: var(--text); }
    .status-pill.success { color: #bbf7d0; border-color: rgba(52, 211, 153, 0.35); }
    .status-pill.warn { color: #fde68a; border-color: rgba(251, 191, 36, 0.35); }
    .status-pill.error { color: #fecaca; border-color: rgba(248, 113, 113, 0.35); }

    .controls {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      align-items: end;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
    }

    select, input {
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(8, 17, 31, 0.7);
      color: var(--text);
      font-size: 14px;
      outline: none;
    }

    .button-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    button {
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid transparent;
      background: linear-gradient(135deg, rgba(96, 165, 250, 0.2), rgba(52, 211, 153, 0.18));
      color: var(--text);
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.15s ease, border-color 0.15s ease;
    }

    button:hover { transform: translateY(-1px); border-color: rgba(96, 165, 250, 0.35); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .secondary { background: rgba(255,255,255,0.04); border-color: var(--border); }
    .danger { background: rgba(248, 113, 113, 0.14); border-color: rgba(248, 113, 113, 0.28); }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }

    .metric {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
    }

    .metric .label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .metric .value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }

    .metric .hint {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .layout {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
    }

    .chart-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }

    .chart-card h3, .panel h3 {
      margin: 0 0 12px 0;
      font-size: 16px;
    }

    .chart {
      min-height: 240px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(8, 17, 31, 0.45);
      padding: 12px;
    }

    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: middle;
      background: currentColor;
    }

    .list, .log-table {
      width: 100%;
      border-collapse: collapse;
    }

    .list td {
      padding: 10px 0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      vertical-align: top;
    }

    .list td:first-child { color: var(--muted); width: 38%; }

    .progress-shell {
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
      margin-top: 12px;
    }

    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 0.25s ease;
    }

    .log-shell {
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: rgba(8, 17, 31, 0.4);
    }

    .log-table th, .log-table td {
      padding: 10px 12px;
      font-size: 13px;
      text-align: left;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      white-space: nowrap;
    }

    .log-table th {
      position: sticky;
      top: 0;
      background: rgba(8, 17, 31, 0.92);
      color: var(--muted);
      z-index: 1;
    }

    .empty {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 200px;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      padding: 18px;
    }

    .json-box {
      background: rgba(8, 17, 31, 0.52);
      border-radius: 16px;
      border: 1px solid var(--border);
      padding: 14px;
      color: #cbd5e1;
      font-size: 12px;
      line-height: 1.55;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    @media (max-width: 1100px) {
      .hero, .layout { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 720px) {
      .wrap { padding: 18px; }
      .controls, .button-grid, .metrics { grid-template-columns: 1fr; }
      .title h1 { font-size: 22px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="panel">
        <div class="title">
          <div style="font-size: 30px;">⚡</div>
          <div>
            <h1>VitaScale Dashboard</h1>
            <p class="subtitle">Interactive visualization for the OpenEnv autoscaling environment. Reset a task, send scaling actions, and watch load, capacity, cost, reward, and SLA pressure evolve step by step.</p>
          </div>
        </div>
        <div class="badge-row">
          <div class="badge">OpenEnv-compatible HTTP API</div>
          <div class="badge">24-hour / 720-step episodes</div>
          <div class="badge">Tasks: easy · medium · hard</div>
        </div>
        <div class="status-row" id="statusRow">
          <div class="status-pill"><strong>Status:</strong> Ready</div>
        </div>
      </div>
      <div class="panel">
        <h3>Task Controls</h3>
        <div class="controls">
          <div class="field">
            <label for="taskSelect">Task</label>
            <select id="taskSelect"></select>
          </div>
          <div class="field">
            <label for="numInstances">Instances to add/remove</label>
            <input id="numInstances" type="number" min="0" max="20" value="2" />
          </div>
        </div>
        <div class="button-grid">
          <button id="resetBtn">Reset task</button>
          <button class="secondary" id="refreshBtn">Refresh state</button>
          <button class="secondary" id="doNothingBtn">Do nothing</button>
          <button id="scaleUpBtn">Scale up</button>
          <button id="scaleDownBtn">Scale down</button>
          <button class="danger" id="migrateBtn">Migrate load</button>
        </div>
      </div>
    </section>

    <section class="metrics" id="metricsGrid">
      <div class="metric"><div class="label">Current Load</div><div class="value" id="loadValue">—</div><div class="hint">req/min</div></div>
      <div class="metric"><div class="label">Instances</div><div class="value" id="instancesValue">—</div><div class="hint" id="capacityHint">capacity —</div></div>
      <div class="metric"><div class="label">Cost So Far</div><div class="value" id="costValue">—</div><div class="hint">USD cumulative</div></div>
      <div class="metric"><div class="label">Last Reward</div><div class="value" id="rewardValue">—</div><div class="hint">per-step reward</div></div>
      <div class="metric"><div class="label">CPU / Memory</div><div class="value" id="cpuValue">—</div><div class="hint" id="memoryHint">memory —</div></div>
      <div class="metric"><div class="label">SLA Violations</div><div class="value" id="slaValue">—</div><div class="hint">minutes so far</div></div>
      <div class="metric"><div class="label">Pending Queue</div><div class="value" id="queueValue">—</div><div class="hint">queued requests</div></div>
      <div class="metric"><div class="label">Response Time</div><div class="value" id="respValue">—</div><div class="hint">avg ms</div></div>
    </section>

    <section class="layout">
      <div class="chart-grid">
        <div class="panel chart-card">
          <h3>Load vs Capacity</h3>
          <div class="legend">
            <span style="color:#34d399">Load</span>
            <span style="color:#60a5fa">Capacity</span>
          </div>
          <div class="chart" id="loadChart"></div>
        </div>
        <div class="panel chart-card">
          <h3>Reward & Cost Trajectory</h3>
          <div class="legend">
            <span style="color:#fbbf24">Reward</span>
            <span style="color:#f472b6">Cost</span>
            <span style="color:#a78bfa">Response time</span>
          </div>
          <div class="chart" id="rewardChart"></div>
        </div>
        <div class="panel">
          <h3>Action Log</h3>
          <div class="log-shell">
            <table class="log-table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Action</th>
                  <th>Reward</th>
                  <th>Load</th>
                  <th>Instances</th>
                  <th>Queue</th>
                  <th>Done</th>
                </tr>
              </thead>
              <tbody id="logBody"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div style="display:grid; gap:18px;">
        <div class="panel">
          <h3>Task Snapshot</h3>
          <table class="list">
            <tbody>
              <tr><td>Task</td><td id="taskValue">—</td></tr>
              <tr><td>Difficulty</td><td id="difficultyValue">—</td></tr>
              <tr><td>Progress</td><td id="progressValue">—</td></tr>
              <tr><td>Recent events</td><td id="eventsValue">None</td></tr>
            </tbody>
          </table>
          <div class="progress-shell"><div class="progress-bar" id="progressBar"></div></div>
        </div>
        <div class="panel">
          <h3>Selected Task Details</h3>
          <div id="taskMeta" class="subtitle">Choose a task to inspect its objective and difficulty.</div>
        </div>
        <div class="panel">
          <h3>Latest API Payload</h3>
          <div class="json-box" id="payloadBox">No response captured yet.</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const FALLBACK_TASKS = __TASKS_JSON__;
    const CAPACITY_PER_INSTANCE = 175;
    const state = {
      taskMeta: {},
      currentTask: null,
      currentStep: 0,
      maxSteps: 720,
      lastReward: 0,
      lastObservation: null,
      lastPayload: null,
      history: [],
    };

    const el = {
      statusRow: document.getElementById('statusRow'),
      taskSelect: document.getElementById('taskSelect'),
      numInstances: document.getElementById('numInstances'),
      taskMeta: document.getElementById('taskMeta'),
      taskValue: document.getElementById('taskValue'),
      difficultyValue: document.getElementById('difficultyValue'),
      progressValue: document.getElementById('progressValue'),
      eventsValue: document.getElementById('eventsValue'),
      progressBar: document.getElementById('progressBar'),
      payloadBox: document.getElementById('payloadBox'),
      loadValue: document.getElementById('loadValue'),
      instancesValue: document.getElementById('instancesValue'),
      capacityHint: document.getElementById('capacityHint'),
      costValue: document.getElementById('costValue'),
      rewardValue: document.getElementById('rewardValue'),
      cpuValue: document.getElementById('cpuValue'),
      memoryHint: document.getElementById('memoryHint'),
      slaValue: document.getElementById('slaValue'),
      queueValue: document.getElementById('queueValue'),
      respValue: document.getElementById('respValue'),
      loadChart: document.getElementById('loadChart'),
      rewardChart: document.getElementById('rewardChart'),
      logBody: document.getElementById('logBody'),
      resetBtn: document.getElementById('resetBtn'),
      refreshBtn: document.getElementById('refreshBtn'),
      doNothingBtn: document.getElementById('doNothingBtn'),
      scaleUpBtn: document.getElementById('scaleUpBtn'),
      scaleDownBtn: document.getElementById('scaleDownBtn'),
      migrateBtn: document.getElementById('migrateBtn'),
    };

    function setStatus(text, kind = '') {
      el.statusRow.innerHTML = `<div class="status-pill ${kind}"><strong>Status:</strong> ${text}</div>`;
    }

    function setBusy(isBusy) {
      [el.resetBtn, el.refreshBtn, el.doNothingBtn, el.scaleUpBtn, el.scaleDownBtn, el.migrateBtn].forEach((button) => {
        button.disabled = isBusy;
      });
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
      });
      const text = await response.text();
      let payload = {};
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { raw: text };
      }
      if (!response.ok) {
        const detail = payload.detail || payload.raw || `${response.status} ${response.statusText}`;
        throw new Error(detail);
      }
      return payload;
    }

    function clampHistory() {
      if (state.history.length > 80) {
        state.history = state.history.slice(-80);
      }
    }

    function formatNumber(value, digits = 0, suffix = '') {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return '—';
      }
      return `${Number(value).toFixed(digits)}${suffix}`;
    }

    function renderTaskMeta() {
      const taskId = el.taskSelect.value;
      const meta = state.taskMeta[taskId];
      if (!meta) {
        el.taskMeta.textContent = 'Choose a task to inspect its objective and difficulty.';
        return;
      }
      el.taskMeta.innerHTML = `
        <strong style="display:block; margin-bottom:6px; color: var(--text);">${meta.name}</strong>
        <div style="margin-bottom:8px; color: var(--muted);">Difficulty: ${meta.difficulty} · Max steps: ${meta.max_steps}</div>
        <div>${meta.description}</div>
      `;
    }

    function updateProgress(step, maxSteps) {
      const percent = maxSteps > 0 ? Math.min(100, (step / maxSteps) * 100) : 0;
      el.progressValue.textContent = `${step} / ${maxSteps}`;
      el.progressBar.style.width = `${percent.toFixed(1)}%`;
    }

    function updateObservation(observation = {}, info = {}, done = false) {
      state.lastObservation = observation;
      const instances = observation.instance_count;
      const capacity = typeof instances === 'number' ? instances * CAPACITY_PER_INSTANCE : null;
      const recentEvents = Array.isArray(observation.recent_events) ? observation.recent_events : [];

      el.loadValue.textContent = formatNumber(observation.current_load, 0);
      el.instancesValue.textContent = formatNumber(instances, 0);
      el.capacityHint.textContent = capacity === null ? 'capacity —' : `capacity ${capacity.toFixed(0)} req/min`;
      el.costValue.textContent = formatNumber(observation.cost_so_far, 2, ' $');
      el.rewardValue.textContent = formatNumber(state.lastReward, 2);
      el.cpuValue.textContent = observation.cpu_util === undefined ? '—' : `${(Number(observation.cpu_util) * 100).toFixed(1)}%`;
      el.memoryHint.textContent = observation.memory_util === undefined ? 'memory —' : `memory ${(Number(observation.memory_util) * 100).toFixed(1)}%`;
      el.slaValue.textContent = formatNumber(observation.sla_violation_minutes, 0);
      el.queueValue.textContent = formatNumber(observation.pending_requests, 0);
      el.respValue.textContent = formatNumber(observation.avg_response_time_ms, 1, ' ms');
      el.taskValue.textContent = state.currentTask || '—';
      el.difficultyValue.textContent = observation.difficulty_level ? `Level ${observation.difficulty_level}` : '—';
      el.eventsValue.textContent = recentEvents.length ? recentEvents.join(', ') : 'None';
      updateProgress(state.currentStep, state.maxSteps);

      const statusText = done ? 'Episode complete' : `Live — step ${state.currentStep}`;
      setStatus(statusText, done ? 'success' : '');
    }

    function renderPayload(payload) {
      state.lastPayload = payload;
      el.payloadBox.textContent = JSON.stringify(payload, null, 2);
    }

    function renderLog() {
      if (!state.history.length) {
        el.logBody.innerHTML = `<tr><td colspan="7"><div class="empty">Reset a task and start stepping to see the trajectory.</div></td></tr>`;
        return;
      }
      el.logBody.innerHTML = state.history.slice().reverse().map((row) => `
        <tr>
          <td>${row.step}</td>
          <td>${row.action}</td>
          <td>${row.reward.toFixed(2)}</td>
          <td>${row.load.toFixed(0)}</td>
          <td>${row.instances}</td>
          <td>${row.queue.toFixed(0)}</td>
          <td>${row.done ? 'true' : 'false'}</td>
        </tr>
      `).join('');
    }

    function renderChart(container, series) {
      const width = 760;
      const height = 260;
      const padding = 24;
      const allValues = series.flatMap((line) => line.values).filter((value) => Number.isFinite(value));
      const maxLen = Math.max(0, ...series.map((line) => line.values.length));

      if (!allValues.length || maxLen < 1) {
        container.innerHTML = `<div class="empty">No trajectory yet — reset a task and take a few steps.</div>`;
        return;
      }

      const min = Math.min(...allValues);
      const max = Math.max(...allValues);
      const range = max - min || 1;
      const chartHeight = height - padding * 2;
      const chartWidth = width - padding * 2;

      const grid = Array.from({ length: 4 }, (_, idx) => {
        const y = padding + (chartHeight / 3) * idx;
        return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(148,163,184,0.18)" stroke-width="1" />`;
      }).join('');

      const lines = series.map((line) => {
        const points = line.values.map((value, index) => {
          const x = maxLen === 1 ? width / 2 : padding + (index / Math.max(1, maxLen - 1)) * chartWidth;
          const y = height - padding - ((value - min) / range) * chartHeight;
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        }).join(' ');
        return `<polyline fill="none" stroke="${line.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${points}" />`;
      }).join('');

      container.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" width="100%" height="230" role="img" aria-label="Time series chart">
          ${grid}
          ${lines}
          <text x="${padding}" y="20" fill="rgba(229,238,251,0.9)" font-size="12">max ${max.toFixed(2)}</text>
          <text x="${padding}" y="${height - 6}" fill="rgba(156,176,202,0.9)" font-size="12">min ${min.toFixed(2)}</text>
        </svg>
      `;
    }

    function renderCharts() {
      renderChart(el.loadChart, [
        { color: '#34d399', values: state.history.map((row) => row.load) },
        { color: '#60a5fa', values: state.history.map((row) => row.capacity) },
      ]);

      renderChart(el.rewardChart, [
        { color: '#fbbf24', values: state.history.map((row) => row.reward) },
        { color: '#f472b6', values: state.history.map((row) => row.cost) },
        { color: '#a78bfa', values: state.history.map((row) => row.response) },
      ]);
    }

    function captureStep(result, actionLabel) {
      const observation = result.observation || {};
      const reward = Number(result.reward || 0);
      const done = Boolean(result.done);
      state.lastReward = reward;
      state.currentStep = Number(result.info?.step || observation.timestamp || state.history.length + 1);
      state.maxSteps = Number(result.info?.max_steps || state.maxSteps || 720);
      state.history.push({
        step: state.currentStep,
        action: actionLabel,
        reward,
        load: Number(observation.current_load || 0),
        capacity: Number(observation.instance_count || 0) * CAPACITY_PER_INSTANCE,
        cost: Number(observation.cost_so_far || 0),
        response: Number(observation.avg_response_time_ms || 0),
        queue: Number(observation.pending_requests || 0),
        instances: Number(observation.instance_count || 0),
        done,
      });
      clampHistory();
      updateObservation(observation, result.info || {}, done);
      renderPayload(result);
      renderLog();
      renderCharts();
    }

    async function loadTasks() {
      try {
        const payload = await api('/tasks');
        state.taskMeta = payload;
      } catch {
        state.taskMeta = Object.fromEntries(FALLBACK_TASKS.map((task) => [task, {
          name: task,
          description: 'Task metadata unavailable — endpoint fallback in use.',
          difficulty: task.replace('_bench', ''),
          max_steps: 720,
        }]));
      }

      const taskIds = Object.keys(state.taskMeta).length ? Object.keys(state.taskMeta) : FALLBACK_TASKS;
      el.taskSelect.innerHTML = taskIds.map((task) => `<option value="${task}">${task}</option>`).join('');
      if (!state.currentTask) {
        state.currentTask = taskIds[0] || null;
      }
      el.taskSelect.value = state.currentTask || taskIds[0] || '';
      renderTaskMeta();
    }

    async function refreshState() {
      setBusy(true);
      try {
        const payload = await api('/state');
        state.currentTask = payload.task || el.taskSelect.value || state.currentTask;
        state.currentStep = Number(payload.step || 0);
        state.maxSteps = Number(payload.max_steps || 720);
        el.taskValue.textContent = state.currentTask || '—';
        el.progressValue.textContent = `${state.currentStep} / ${state.maxSteps}`;
        el.progressBar.style.width = `${state.maxSteps > 0 ? (state.currentStep / state.maxSteps) * 100 : 0}%`;
        el.instancesValue.textContent = formatNumber(payload.instance_count, 0);
        el.capacityHint.textContent = payload.instance_count === undefined ? 'capacity —' : `capacity ${(payload.instance_count * CAPACITY_PER_INSTANCE).toFixed(0)} req/min`;
        el.costValue.textContent = formatNumber(payload.cost_so_far, 2, ' $');
        el.slaValue.textContent = formatNumber(payload.sla_violation_minutes, 0);
        el.rewardValue.textContent = formatNumber(payload.total_reward, 2);
        renderPayload(payload);
        setStatus(payload.done ? 'Waiting for reset' : `State synced — step ${state.currentStep}`, payload.done ? 'warn' : '');
      } catch (error) {
        setStatus(error.message, 'error');
      } finally {
        setBusy(false);
      }
    }

    async function resetTask() {
      setBusy(true);
      try {
        state.currentTask = el.taskSelect.value;
        state.history = [];
        state.lastReward = 0;
        const payload = await api(`/reset?task_id=${encodeURIComponent(state.currentTask)}`, { method: 'POST' });
        state.currentStep = 0;
        state.maxSteps = 720;
        updateObservation(payload.observation || {}, payload.info || {}, false);
        renderPayload(payload);
        renderLog();
        renderCharts();
        setStatus(`Reset ${state.currentTask}`, 'success');
      } catch (error) {
        setStatus(error.message, 'error');
      } finally {
        setBusy(false);
      }
    }

    async function takeAction(actionType) {
      setBusy(true);
      try {
        if (!state.currentTask) {
          state.currentTask = el.taskSelect.value;
        }
        const requested = Math.max(0, Math.min(20, Number(el.numInstances.value || 0)));
        const action = {
          action_type: actionType,
          num_instances: actionType === 'do_nothing' ? 0 : requested,
        };
        const actionLabel = `${action.action_type}(${action.num_instances})`;
        const payload = await api('/step', {
          method: 'POST',
          body: JSON.stringify(action),
        });
        captureStep(payload, actionLabel);
      } catch (error) {
        setStatus(error.message, 'error');
      } finally {
        setBusy(false);
      }
    }

    el.taskSelect.addEventListener('change', () => {
      state.currentTask = el.taskSelect.value;
      renderTaskMeta();
    });
    el.resetBtn.addEventListener('click', resetTask);
    el.refreshBtn.addEventListener('click', refreshState);
    el.doNothingBtn.addEventListener('click', () => takeAction('do_nothing'));
    el.scaleUpBtn.addEventListener('click', () => takeAction('scale_up'));
    el.scaleDownBtn.addEventListener('click', () => takeAction('scale_down'));
    el.migrateBtn.addEventListener('click', () => takeAction('migrate_load'));

    (async function init() {
      await loadTasks();
      renderLog();
      renderCharts();
      await refreshState();
      setStatus('Dashboard ready — reset a task to begin.', 'success');
    })();
  </script>
</body>
</html>
"""


def render_dashboard_html(tasks: Sequence[str]) -> str:
    """Render the interactive VitaScale dashboard HTML."""
    return _TEMPLATE.replace("__TASKS_JSON__", json.dumps(list(tasks)))
