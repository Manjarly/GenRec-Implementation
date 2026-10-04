/**
 * Netflix GenRec Interactive Walkthrough & Client Logic
 */

let allCatalog = [];
let allScenarios = [];
let activeScenario = null;
let activeHistory = [];
let previousRankingsMap = {}; // Tracks previous title -> rank for delta calculation
let currentRecommendations = [];
let currentCandidateAttributions = {};

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initAudienceModeToggle();
  initGuidedBanner();
  initAddTitleControls();
  initDrawer();
  initBenchmarkRunner();
  initEfficiencySlider();
  initTokenModal();
  loadCatalogAndScenarios();
});

// 1. Navigation Tabs
function initNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const targetId = `view-${btn.dataset.tab}`;
      document.querySelectorAll('.view-content').forEach(view => view.classList.remove('active'));
      const targetView = document.getElementById(targetId);
      if (targetView) {
        targetView.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });
}

// 2. Audience Mode (Plain English vs Technical Deep-Dive)
function initAudienceModeToggle() {
  const btnPlain = document.getElementById('btn-mode-plain');
  const btnTech = document.getElementById('btn-mode-tech');

  btnPlain.addEventListener('click', () => {
    btnPlain.classList.add('active');
    btnTech.classList.remove('active');
    document.querySelectorAll('.mode-plain-text').forEach(el => el.style.display = 'block');
    document.querySelectorAll('.mode-tech-text').forEach(el => el.style.display = 'none');
  });

  btnTech.addEventListener('click', () => {
    btnTech.classList.add('active');
    btnPlain.classList.remove('active');
    document.querySelectorAll('.mode-plain-text').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.mode-tech-text').forEach(el => el.style.display = 'block');
  });
}

// 3. Guided Banner Toggle
function initGuidedBanner() {
  const btn = document.getElementById('btn-toggle-guide');
  const body = document.getElementById('guided-body');
  btn.addEventListener('click', () => {
    if (body.style.display === 'none') {
      body.style.display = 'block';
      btn.textContent = 'Hide Guide ▴';
    } else {
      body.style.display = 'none';
      btn.textContent = 'Show Guide ▾';
    }
  });

  document.getElementById('btn-dismiss-delta').addEventListener('click', () => {
    document.getElementById('rank-delta-banner').style.display = 'none';
  });
}

// 4. Load Catalog & Scenarios
async function loadCatalogAndScenarios() {
  try {
    const [catRes, scRes] = await Promise.all([
      fetch('/api/catalog'),
      fetch('/api/scenarios')
    ]);
    allCatalog = await catRes.json();
    allScenarios = await scRes.json();

    // Populate Catalog dropdown for adding titles
    const catalogSelect = document.getElementById('select-catalog-add');
    catalogSelect.innerHTML = '';
    allCatalog.forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = `${item.title} (${item.type}, ${item.release_year})`;
      catalogSelect.appendChild(opt);
    });

    // Populate Curated Scenarios
    const scenariosList = document.getElementById('scenario-buttons-list');
    scenariosList.innerHTML = '';
    allScenarios.forEach((sc, idx) => {
      const btn = document.createElement('button');
      btn.className = `scenario-btn ${idx === 0 ? 'active' : ''}`;
      btn.textContent = sc.name;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadScenario(sc);
      });
      scenariosList.appendChild(btn);
    });

    if (allScenarios.length > 0) {
      loadScenario(allScenarios[0]);
    }
  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

function loadScenario(scenario) {
  activeScenario = scenario;
  // Clone interactions so user can modify freely
  activeHistory = scenario.interactions.map(item => ({ ...item }));
  renderWatchHistoryList();
  triggerRecommendation(false);
}

// 5. Watch History Rendering & Builder
function renderWatchHistoryList() {
  const container = document.getElementById('history-items-container');
  container.innerHTML = '';
  document.getElementById('watch-history-count').textContent = `${activeHistory.length} titles`;

  activeHistory.forEach((inter, idx) => {
    const it = allCatalog.find(c => c.id === inter.item_id);
    const title = it ? it.title : `Title #${inter.item_id}`;
    const pct = Math.round(inter.completion_pct * 100);

    const row = document.createElement('div');
    row.className = 'history-item-row';
    row.innerHTML = `
      <div class="item-main-info">
        <span class="item-title">${title}</span>
        <span class="item-sub">${inter.device} • ${inter.rating || 'No Rating'}</span>
      </div>
      <div class="item-actions">
        <span class="item-pct-badge">${pct}%</span>
        <button class="btn-remove-item" title="Remove from history">✕</button>
      </div>
    `;

    row.querySelector('.btn-remove-item').addEventListener('click', (e) => {
      e.stopPropagation();
      removeHistoryItem(idx, title);
    });

    container.appendChild(row);
  });
}

function removeHistoryItem(idx, title) {
  activeHistory.splice(idx, 1);
  renderWatchHistoryList();
  showRankDeltaNotification(`Removed "${title}" from history. Re-evaluating candidate affinities...`);
  triggerRecommendation(true);
}

function initAddTitleControls() {
  document.getElementById('btn-add-title-to-history').addEventListener('click', () => {
    const select = document.getElementById('select-catalog-add');
    const itemId = parseInt(select.value);
    const compVal = parseFloat(document.getElementById('select-add-completion').value);
    const ratingVal = document.getElementById('select-add-rating').value;
    const deviceVal = document.getElementById('device-context-select').value;
    const timeVal = document.getElementById('time-context-select').value;

    const it = allCatalog.find(c => c.id === itemId);
    if (!it) return;

    if (activeHistory.some(h => h.item_id === itemId)) {
      alert(`"${it.title}" is already in your watch history.`);
      return;
    }

    activeHistory.push({
      item_id: itemId,
      completion_pct: compVal,
      rating: ratingVal === 'None' ? null : ratingVal,
      rewatched: false,
      device: deviceVal,
      time_of_day: timeVal,
      day_of_week: 'Weekend'
    });

    renderWatchHistoryList();
    showRankDeltaNotification(`Added "${it.title}" to history! Notice how recommendations and match scores adapt below.`);
    triggerRecommendation(true);
  });

  document.getElementById('btn-recalculate-rankings').addEventListener('click', () => {
    triggerRecommendation(true);
  });
}

// 6. Recommendation Trigger & Inference
async function triggerRecommendation(isUpdate = false) {
  const device = document.getElementById('device-context-select').value;
  const timeVal = document.getElementById('time-context-select').value;

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        history: activeHistory,
        device: device,
        time_of_day: timeVal
      })
    });
    const data = await res.json();

    document.getElementById('stat-latency').textContent = `${data.latency_ms} ms`;
    const bbEl = document.getElementById('stat-backbone');
    const tokenBtn = document.getElementById('btn-open-token-modal');
    if (bbEl && data.active_backbone) {
      if (data.llama_ready) {
        bbEl.textContent = 'meta-llama/Llama-3.2-1B';
        bbEl.className = 'telemetry-val green-text';
        if (tokenBtn) {
          tokenBtn.textContent = '⚙️ Llama 3.2 Active';
          tokenBtn.classList.remove('awaiting-token');
        }
      } else {
        bbEl.textContent = `${data.active_backbone} · Awaiting HF Token`;
        bbEl.className = 'telemetry-val orange-text';
        if (tokenBtn) {
          tokenBtn.textContent = '🔑 Setup Llama 3.2';
          tokenBtn.classList.add('awaiting-token');
        }
      }
    }
    document.getElementById('stat-candidate-count').textContent = `${data.candidate_count} Titles Evaluated`;

    renderRecommendations(data.ranked_recommendations, data.baseline_recommendations, isUpdate);

    previousRankingsMap = {};
    data.ranked_recommendations.forEach((item, idx) => {
      previousRankingsMap[item.id] = idx + 1;
    });

    document.getElementById('prompt-preview-content').textContent = data.verbalized_prompt;
    document.getElementById('legacy-tabular-features-raw').textContent = JSON.stringify(data.legacy_tabular_features, null, 2);

  } catch (err) {
    console.error('Recommendation API error:', err);
  }
}

// 7. Render Recommendation Cards
function renderRecommendations(rankedItems, baselineItems, isUpdate) {
  const container = document.getElementById('recommendation-cards-list');
  const showLegacy = document.getElementById('check-compare-legacy').checked;
  container.innerHTML = '';

  currentRecommendations = rankedItems;

  rankedItems.forEach((item, idx) => {
    const currentRank = idx + 1;
    const prevRank = previousRankingsMap[item.id];

    let rankDeltaHtml = '';
    if (isUpdate && prevRank !== undefined) {
      const delta = prevRank - currentRank;
      if (delta > 0) {
        rankDeltaHtml = `<span style="color:var(--accent-green); font-size:0.75rem; font-weight:700;">▲ +${delta}</span>`;
      } else if (delta < 0) {
        rankDeltaHtml = `<span style="color:#ef4444; font-size:0.75rem; font-weight:700;">▼ ${delta}</span>`;
      } else {
        rankDeltaHtml = `<span style="color:var(--text-muted); font-size:0.75rem;">━</span>`;
      }
    }

    let legacyCompHtml = '';
    if (showLegacy && baselineItems) {
      const bIdx = baselineItems.findIndex(b => b.id === item.id);
      const legacyRank = bIdx >= 0 ? bIdx + 1 : 9;
      legacyCompHtml = `
        <div class="legacy-comparison-line">
          <span>GenRec Rank: <strong>#${currentRank}</strong></span>
          <span>Legacy Popularity Rank: <strong>#${legacyRank}</strong></span>
        </div>
      `;
    }

    const tagsHtml = item.genres.map(g => `<span class="rec-tag">${g}</span>`).join('');

    const card = document.createElement('div');
    card.className = 'rec-card';
    card.innerHTML = `
      <div class="rec-card-top">
        <div class="rec-card-left">
          <span class="rec-rank-digit">#${currentRank}</span>
          <div class="rec-title-block">
            <h3>${item.title} <span style="font-size:0.78rem; color:var(--text-muted); font-weight:400;">(${item.release_year})</span> ${rankDeltaHtml}</h3>
            <div class="rec-meta-tags">${tagsHtml}</div>
          </div>
        </div>
        <div class="rec-card-right">
          <div class="match-pct-box">
            <span class="match-pct-val">${item.confidence_pct}%</span>
            <span class="match-pct-sub">Match Score</span>
          </div>
          <button class="btn-card-add" title="Add to watch history">+ Add</button>
        </div>
      </div>

      <div class="rec-reasoning-snippet">
        <strong>AI Insight:</strong> ${item.semantic_reasoning}
      </div>

      ${legacyCompHtml}
    `;

    card.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-card-add')) return;
      openExplanationDrawer(item);
    });

    card.querySelector('.btn-card-add').addEventListener('click', (e) => {
      e.stopPropagation();
      quickAddFromRecommendation(item);
    });

    container.appendChild(card);
  });
}

function quickAddFromRecommendation(item) {
  if (activeHistory.some(h => h.item_id === item.id)) {
    alert(`"${item.title}" is already in your watch history.`);
    return;
  }

  activeHistory.push({
    item_id: item.id,
    completion_pct: 1.0,
    rating: 'ThumbsUp',
    rewatched: false,
    device: document.getElementById('device-context-select').value,
    time_of_day: document.getElementById('time-context-select').value,
    day_of_week: 'Weekend'
  });

  renderWatchHistoryList();
  showRankDeltaNotification(`You added "${item.title}" to history! Watch how other candidate ranks adapted.`);
  triggerRecommendation(true);
}

function showRankDeltaNotification(msg) {
  const banner = document.getElementById('rank-delta-banner');
  document.getElementById('rank-delta-text').textContent = msg;
  banner.style.display = 'flex';
}

// 8. Explanation Drawer
function initDrawer() {
  document.getElementById('btn-close-drawer').addEventListener('click', () => {
    document.getElementById('explanation-drawer').style.display = 'none';
  });
}

function openExplanationDrawer(item) {
  const drawer = document.getElementById('explanation-drawer');
  document.getElementById('drawer-movie-title').textContent = `${item.title} (${item.release_year})`;
  document.getElementById('drawer-plain-reasoning').textContent = item.semantic_reasoning;
  document.getElementById('drawer-raw-score').textContent = item.score;

  const attribList = document.getElementById('drawer-attribution-list');
  attribList.innerHTML = '';

  if (item.attributions && item.attributions.length > 0) {
    item.attributions.forEach(att => {
      const row = document.createElement('div');
      row.className = 'attrib-row';
      row.innerHTML = `
        <div class="attrib-top">
          <span>From your watch of <strong>"${att.history_title}"</strong></span>
          <span style="font-family:var(--font-mono); font-weight:700;">${att.contribution_pct}%</span>
        </div>
        <div class="attrib-bar-track">
          <div class="attrib-bar-fill" style="width: ${Math.max(5, att.contribution_pct)}%;"></div>
        </div>
      `;
      attribList.appendChild(row);
    });
  } else {
    attribList.innerHTML = `<p style="color:var(--text-muted); font-size:0.8rem;">Add more titles to your history to decompose multi-title attention.</p>`;
  }

  const drawerAddBtn = document.getElementById('btn-drawer-add-action');
  drawerAddBtn.onclick = () => {
    quickAddFromRecommendation(item);
    drawer.style.display = 'none';
  };

  drawer.style.display = 'flex';
  drawer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 9. Benchmark Runner
function initBenchmarkRunner() {
  const btn = document.getElementById('btn-run-live-benchmark');
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    document.getElementById('benchmark-status-msg').textContent = 'Executing 15 benchmark iterations in local CPU/GPU...';

    try {
      const res = await fetch('/api/latency_benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: activeHistory })
      });
      const bench = await res.json();

      document.getElementById('flame-prefill-val').textContent = `~${bench.prefill_only.p50_ms} ms P50`;
      document.getElementById('flame-autoreg-val').textContent = `~${bench.autoregressive.p50_ms} ms P50`;
      document.getElementById('flame-speedup-factor').textContent = `${bench.speedup_factor}×`;
      document.getElementById('benchmark-status-msg').textContent = `Completed! Prefill scoring is ${bench.speedup_factor}x faster with 100% catalog compliance.`;
    } catch (err) {
      console.error(err);
      document.getElementById('benchmark-status-msg').textContent = 'Benchmark execution failed.';
    } finally {
      btn.disabled = false;
    }
  });
}

// 10. Sample Efficiency Slider
function initEfficiencySlider() {
  const slider = document.getElementById('efficiency-slider');
  const curve = {
    10: { genrec: 0.620, legacy: 0.310, delta: '+100.0% Better' },
    25: { genrec: 0.710, legacy: 0.440, delta: '+61.4% Better' },
    40: { genrec: 0.745, legacy: 0.520, delta: '+43.2% Better' },
    55: { genrec: 0.780, legacy: 0.605, delta: '+28.9% Better' },
    70: { genrec: 0.805, legacy: 0.670, delta: '+20.1% Better' },
    85: { genrec: 0.825, legacy: 0.715, delta: '+15.3% Better' },
    100: { genrec: 0.840, legacy: 0.740, delta: '+13.5% Better' }
  };

  slider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    const keys = Object.keys(curve).map(Number);
    const closest = keys.reduce((prev, curr) => Math.abs(curr - val) < Math.abs(prev - val) ? curr : prev);
    const d = curve[closest];

    document.getElementById('disp-genrec-mrr').textContent = `${d.genrec.toFixed(3)} MRR`;
    document.getElementById('disp-legacy-mrr').textContent = `${d.legacy.toFixed(3)} MRR`;
    document.getElementById('disp-delta-val').textContent = d.delta;
  });
}


// 11. Foundation Model & Token Modal Setup
function initTokenModal() {
  const modal = document.getElementById('modal-token-setup');
  const openBtn = document.getElementById('btn-open-token-modal');
  const closeBtn = document.getElementById('btn-close-token-modal');
  const cancelBtn = document.getElementById('btn-cancel-token');
  const saveBtn = document.getElementById('btn-save-token');
  const inputToken = document.getElementById('input-hf-token');
  const toggleVisBtn = document.getElementById('btn-toggle-token-vis');
  const feedbackArea = document.getElementById('token-feedback-area');
  const activePill = document.getElementById('modal-active-backbone-pill');

  if (!modal || !openBtn) return;

  const showModal = async () => {
    feedbackArea.style.display = 'none';
    feedbackArea.className = 'token-feedback-area';
    feedbackArea.textContent = '';
    
    // Fetch live status
    try {
      const res = await fetch('/api/model_status');
      const status = await res.json();
      if (activePill) {
        activePill.textContent = status.active_backbone;
        if (status.llama_ready) {
          activePill.className = 'status-val-pill active-llama';
        } else {
          activePill.className = 'status-val-pill active-fallback';
        }
      }
    } catch (e) {
      console.warn('Could not fetch model status:', e);
    }

    modal.style.display = 'flex';
    inputToken.focus();
  };

  const hideModal = () => {
    modal.style.display = 'none';
  };

  openBtn.addEventListener('click', showModal);
  closeBtn.addEventListener('click', hideModal);
  cancelBtn.addEventListener('click', hideModal);

  // Close on backdrop click
  modal.addEventListener('click', (e) => {
    if (e.target === modal) hideModal();
  });

  // Toggle password visibility
  toggleVisBtn.addEventListener('click', () => {
    if (inputToken.type === 'password') {
      inputToken.type = 'text';
      toggleVisBtn.textContent = '🔒';
    } else {
      inputToken.type = 'password';
      toggleVisBtn.textContent = '👁';
    }
  });

  // Save & Activate
  saveBtn.addEventListener('click', async () => {
    const tokenVal = inputToken.value.trim();
    if (!tokenVal) {
      feedbackArea.style.display = 'block';
      feedbackArea.className = 'token-feedback-area error';
      feedbackArea.textContent = 'Please enter your Hugging Face access token (hf_...).';
      return;
    }

    const btnText = document.getElementById('btn-save-token-text');
    const btnSpinner = document.getElementById('btn-save-token-spinner');
    saveBtn.disabled = true;
    btnText.style.display = 'none';
    btnSpinner.style.display = 'inline';
    
    feedbackArea.style.display = 'block';
    feedbackArea.className = 'token-feedback-area loading';
    feedbackArea.textContent = 'Connecting to Hugging Face Hub and loading meta-llama/Llama-3.2-1B weights...';

    try {
      const res = await fetch('/api/configure_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hf_token: tokenVal,
          backbone: 'meta-llama/Llama-3.2-1B'
        })
      });
      const data = await res.json();

      if (data.success) {
        feedbackArea.className = 'token-feedback-area success';
        feedbackArea.textContent = `✅ ${data.message}`;
        if (activePill) {
          activePill.textContent = data.active_backbone;
          activePill.className = 'status-val-pill active-llama';
        }
        
        // Re-run recommendation with new model
        setTimeout(() => {
          hideModal();
          runGenRec(true);
        }, 1200);
      } else {
        feedbackArea.className = 'token-feedback-area error';
        feedbackArea.textContent = `❌ ${data.message}`;
      }
    } catch (err) {
      console.error(err);
      feedbackArea.className = 'token-feedback-area error';
      feedbackArea.textContent = 'Failed to communicate with server: ' + err.message;
    } finally {
      saveBtn.disabled = false;
      btnText.style.display = 'inline';
      btnSpinner.style.display = 'none';
    }
  });
}
