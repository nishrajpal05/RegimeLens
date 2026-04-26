/* app.js — NSE Regime Predictor frontend logic */

const API = '';  // same origin

let priceChart = null;
let r2Chart = null;
let pollTimer = null;

// ─── Startup ───────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  checkStatus();
});

// ─── Status polling ─────────────────────────────────────────────────────────

async function checkStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    updateStatusUI(data);

    if (data.status === 'training') {
      showOverlay();
      pollTimer = setTimeout(checkStatus, 3000);
    } else if (data.status === 'ready' || data.status === 'ready_cached' || data.status === 'timeout') {
      hideOverlay();
      clearTimeout(pollTimer);
      loadAll();
      
      // Show warning if using cached/timeout mode
      if (data.status === 'ready_cached') {
        console.warn('⚠ Using cached model (training timed out)');
      } else if (data.status === 'timeout') {
        console.warn('⚠ Training timeout - model may be outdated');
      }
    } else {
      hideOverlay();
    }
  } catch (e) {
    setStatus('error', 'Connection error');
  }
}

function updateStatusUI(data) {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  
  // Map status to display class
  const displayStatus = data.status === 'ready_cached' ? 'cached' : 
                        data.status === 'timeout' ? 'warning' : 
                        data.status;
  
  dot.className = 'status-dot ' + displayStatus;
  
  // Show user-friendly labels
  const labels = {
    'training': 'Training...',
    'ready': 'Ready',
    'ready_cached': 'Using Cache',
    'timeout': '⚠ Timeout',
    'idle': 'Idle',
    'error': 'Error'
  };
  
  label.textContent = labels[data.status] || data.status;
}

// ─── Training ───────────────────────────────────────────────────────────────

async function startTraining() {
  document.getElementById('trainBtn').disabled = true;
  showOverlay();
  animateSteps();

  await fetch(`${API}/api/train`, { method: 'POST' });
  pollTimer = setTimeout(checkStatus, 4000);
}

function showOverlay() {
  document.getElementById('trainingOverlay').classList.add('active');
}
function hideOverlay() {
  document.getElementById('trainingOverlay').classList.remove('active');
  document.getElementById('trainBtn').disabled = false;
}

function animateSteps() {
  const steps = document.querySelectorAll('.step');
  steps.forEach(s => s.className = 'step');
  let i = 0;
  function next() {
    if (i > 0) steps[i - 1].className = 'step done';
    if (i < steps.length) {
      steps[i].className = 'step active';
      i++;
      setTimeout(next, 28000 / steps.length);  // spread over ~28s
    }
  }
  next();
}

// ─── Load all panels ────────────────────────────────────────────────────────

async function loadAll() {
  await Promise.all([
    loadPredict(),
    loadHistory(),
    loadRegime(),
    loadBlend(),
  ]);
}

// ─── Prediction ─────────────────────────────────────────────────────────────

async function loadPredict() {
  try {
    const res = await fetch(`${API}/api/predict`);
    if (!res.ok) return;
    const d = await res.json();

    const dir = document.getElementById('signalDirection');
    const val = document.getElementById('signalValue');
    const meta = document.getElementById('signalMeta');

    dir.textContent = d.direction === 'UP' ? '↑ UP' : '↓ DOWN';
    dir.className = 'signal-direction ' + d.direction.toLowerCase();
    val.textContent = `${(d.prediction * 100).toFixed(4)}% predicted return`;
    meta.textContent = `Confidence: ${(d.confidence * 100).toFixed(1)}%  ·  As of ${d.as_of_date}`;

    // Proxy score
    const ps = d.proxy_stats;
    document.getElementById('proxyScore').textContent = ps.proxy_score?.toFixed(6) ?? '—';
    document.getElementById('fullR2').textContent = ps.full_r2?.toFixed(6) ?? '—';
    document.getElementById('batchR2').textContent = ps.mean_batch_r2?.toFixed(6) ?? '—';
    document.getElementById('leakageGap').textContent =
      ps.leakage_gap !== undefined
        ? `Leakage gap: ${ps.leakage_gap.toFixed(6)} (lower = more consistent across batches)`
        : '';
  } catch (e) {
    console.error('Predict error:', e);
  }
}

// ─── History chart ──────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await fetch(`${API}/api/history`);
    if (!res.ok) return;
    const d = await res.json();

    const ctx = document.getElementById('priceChart').getContext('2d');
    if (priceChart) priceChart.destroy();

    priceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.dates.map(dt => {
          const date = new Date(dt);
          return date.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
        }),
        datasets: [
          {
            label: 'Nifty 50 Close',
            data: d.closes,
            borderColor: '#1B4F72',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            yAxisID: 'y',
          },
          {
            label: 'Predicted Return %',
            data: d.predicted_returns_pct,
            borderColor: '#1E6B45',
            borderWidth: 1,
            borderDash: [4, 3],
            pointRadius: 0,
            tension: 0.3,
            yAxisID: 'y2',
          },
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            labels: {
              font: { family: "'DM Mono', monospace", size: 11 },
              color: '#6B6860',
              boxWidth: 12,
            }
          },
          tooltip: {
            backgroundColor: '#fff',
            borderColor: '#E8E6DF',
            borderWidth: 1,
            titleColor: '#1A1A1A',
            bodyColor: '#6B6860',
            titleFont: { family: "'DM Mono', monospace", size: 11 },
            bodyFont: { family: "'DM Mono', monospace", size: 11 },
          }
        },
        scales: {
          x: {
            grid: { color: '#F0EEE8' },
            ticks: {
              font: { family: "'DM Mono', monospace", size: 10 },
              color: '#6B6860',
              maxTicksLimit: 10,
            }
          },
          y: {
            position: 'left',
            grid: { color: '#F0EEE8' },
            ticks: {
              font: { family: "'DM Mono', monospace", size: 10 },
              color: '#1B4F72',
              callback: v => v.toLocaleString('en-IN'),
            }
          },
          y2: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: {
              font: { family: "'DM Mono', monospace", size: 10 },
              color: '#1E6B45',
              callback: v => v.toFixed(3) + '%',
            }
          }
        }
      }
    });
  } catch (e) {
    console.error('History error:', e);
  }
}

// ─── Regime panel ───────────────────────────────────────────────────────────

async function loadRegime() {
  try {
    const res = await fetch(`${API}/api/regime`);
    if (!res.ok) return;
    const d = await res.json();

    // AUC gauge
    const auc = d.adversarial_auc ?? 0;
    document.getElementById('advAUC').textContent = auc.toFixed(4);
    document.getElementById('aucInterpret').textContent = d.auc_interpretation ?? '';

    // Bar: map auc from [0.5, 1.0] to [0%, 100%]
    const pct = Math.max(0, Math.min(100, (auc - 0.5) / 0.5 * 100));
    document.getElementById('aucBarFill').style.width = pct + '%';

    // Stability list
    const list = document.getElementById('stabilityList');
    list.innerHTML = '';
    const maxScore = d.top_stable_features[0]?.stability_score ?? 1;
    d.top_stable_features.forEach((item, i) => {
      const barPct = (item.stability_score / maxScore * 100).toFixed(1);
      list.innerHTML += `
        <div class="stability-row">
          <span class="stability-rank">${i + 1}</span>
          <span class="stability-name" title="${item.feature}">${item.feature}</span>
          <div class="stability-bar-track">
            <div class="stability-bar-fill" style="width:${barPct}%"></div>
          </div>
          <span class="stability-score">${item.stability_score.toFixed(3)}</span>
        </div>`;
    });
  } catch (e) {
    console.error('Regime error:', e);
  }
}

// ─── Blend panel ────────────────────────────────────────────────────────────

async function loadBlend() {
  try {
    const res = await fetch(`${API}/api/blend`);
    if (!res.ok) return;
    const d = await res.json();

    // Blend weight bars
    const blendList = document.getElementById('blendList');
    blendList.innerHTML = '';
    for (const [name, w] of Object.entries(d.blend_weights)) {
      const pct = (w * 100).toFixed(0);
      blendList.innerHTML += `
        <div class="blend-row">
          <span class="blend-name">${name}</span>
          <div class="blend-bar-track">
            <div class="blend-bar-fill" style="width:${pct}%"></div>
          </div>
          <span class="blend-pct">${pct}%</span>
        </div>`;
    }

    // R² bar chart
    const ctx = document.getElementById('r2Chart').getContext('2d');
    if (r2Chart) r2Chart.destroy();

    const names = Object.keys(d.model_r2);
    const r2vals = Object.values(d.model_r2);
    const colors = names.map(n => n === 'Ridge_Meta' ? '#1B4F72' : '#1E6B45');

    r2Chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          data: r2vals,
          backgroundColor: colors,
          borderRadius: 4,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: ctx => ` R²: ${ctx.raw.toFixed(6)}` },
            titleFont: { family: "'DM Mono', monospace", size: 11 },
            bodyFont: { family: "'DM Mono', monospace", size: 11 },
            backgroundColor: '#fff',
            borderColor: '#E8E6DF',
            borderWidth: 1,
            titleColor: '#1A1A1A',
            bodyColor: '#6B6860',
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { family: "'DM Mono', monospace", size: 10 }, color: '#6B6860' }
          },
          y: {
            grid: { color: '#F0EEE8' },
            ticks: {
              font: { family: "'DM Mono', monospace", size: 10 },
              color: '#6B6860',
              callback: v => v.toFixed(5),
            }
          }
        }
      }
    });
  } catch (e) {
    console.error('Blend error:', e);
  }
}
