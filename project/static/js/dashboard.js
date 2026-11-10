/* Adaptive IDS dashboard - plain JS + Bootstrap + Chart.js */
const $ = (s, r = document) => r.querySelector(s);
const fmt = n => Number(n || 0).toLocaleString();
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (x, d = 2) => (x == null ? '-' : (x * 100).toFixed(d) + '%');
const localTime = ts => ts ? new Date(ts.length === 16 ? ts + ':00Z' : ts + 'Z').toLocaleTimeString() : '';
const SEV_CLASS = {Critical: 'danger', High: 'danger', Medium: 'warning', Low: 'info'};

async function api(path, opts = {}) {
  const init = {method: opts.method || 'GET', headers: {'Content-Type': 'application/json'}};
  if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
  const r = await fetch(path, init);
  if (r.status === 401) { location.href = '/login'; throw new Error('auth'); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), {data, status: r.status});
  return data;
}

function toast(msg) {
  $('#toastBody').textContent = msg;
  bootstrap.Toast.getOrCreateInstance($('#toast'), {delay: 3500}).show();
}

const labelBadge = (label, isAttack) =>
  `<span class="badge text-bg-${isAttack ? 'danger' : 'success'}">${esc(label)}</span>`;

/* ----------------------------------------------------------------- tabs */
let currentTab = 'overview';
document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
  currentTab = b.dataset.tab;
  document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('active', x === b));
  ['overview', 'review', 'models'].forEach(t => $('#tab-' + t).classList.toggle('d-none', t !== currentTab));
  refresh();
}));

/* ----------------------------------------------------------------- charts */
const charts = {};
function chart(id, config) {
  if (charts[id]) { return charts[id]; }
  charts[id] = new Chart($('#' + id), config);
  return charts[id];
}
Chart.defaults.color = '#adb5bd';
Chart.defaults.borderColor = 'rgba(255,255,255,.08)';

/* ----------------------------------------------------------------- overview */
async function refreshOverview() {
  const [s, recent] = await Promise.all([api('/api/stats'), api('/api/predictions?view=all&limit=15')]);
  const t = s.totals;
  $('#kFlows').textContent = fmt(t.flows);
  $('#kAttacks').textContent = fmt(t.attacks);
  $('#kAlerts').textContent = fmt(t.open_alerts);
  $('#kReviewed').textContent = fmt(t.reviewed);
  $('#kLatency').textContent = t.avg_latency_ms;
  $('#kModel').textContent = s.model.version;
  $('#kAlgo').textContent = s.model.algorithm;
  $('#navModel').textContent = 'model: ' + s.model.version;
  $('#badgeAlerts').textContent = t.open_alerts;
  $('#badgePending').textContent = t.pending_feedback;

  const tl = chart('chTimeline', {
    type: 'bar',
    data: {labels: [], datasets: [
      {label: 'Benign', data: [], backgroundColor: '#198754'},
      {label: 'Attack', data: [], backgroundColor: '#dc3545'}]},
    options: {maintainAspectRatio: false, animation: false, scales: {x: {stacked: true}, y: {stacked: true, beginAtZero: true}}}
  });
  tl.data.labels = s.timeline.map(p => localTime(p.minute).replace(/:00(?=\s|$)/, ''));
  tl.data.datasets[0].data = s.timeline.map(p => p.benign);
  tl.data.datasets[1].data = s.timeline.map(p => p.attack);
  tl.update();

  const ty = chart('chTypes', {
    type: 'doughnut', data: {labels: [], datasets: [{data: []}]},
    options: {maintainAspectRatio: false, animation: false, plugins: {legend: {position: 'right', labels: {boxWidth: 12}}}}
  });
  ty.data.labels = s.by_label.map(x => x.label);
  ty.data.datasets[0].data = s.by_label.map(x => x.count);
  ty.update();

  $('#feedBody').innerHTML = recent.map(p => `
    <tr><td>${localTime(p.ts)}</td><td>${labelBadge(p.predicted_label, p.is_attack)}</td>
    <td>${pct(p.confidence, 1)}</td>
    <td>${p.severity ? `<span class="badge text-bg-${SEV_CLASS[p.severity] || 'secondary'}">${p.severity}</span>` : ''}</td>
    <td>${p.latency_ms} ms</td><td>${esc(p.model_version)}</td></tr>`).join('');
}

/* ----------------------------------------------------------------- simulator */
let simTimer = null, simBusy = false, simCount = 0;
$('#simRatio').addEventListener('input', e => $('#simRatioVal').textContent = e.target.value + '%');
async function simTick() {
  if (simBusy) return;
  simBusy = true;
  try {
    await api('/api/simulate/tick', {method: 'POST', body: {attack_ratio: $('#simRatio').value / 100}});
    simCount++;
    $('#simInfo').textContent = `Running - ${fmt(simCount)} flows sent this session`;
  } catch (e) { stopSim(); toast('Simulator stopped: ' + e.message); }
  simBusy = false;
}
function stopSim() {
  clearInterval(simTimer); simTimer = null;
  $('#simBtn').textContent = '\u25B6 Start'; $('#simBtn').className = 'btn btn-sm btn-success';
}
$('#simBtn').addEventListener('click', () => {
  if (simTimer) return stopSim();
  simTimer = setInterval(simTick, Number($('#simRate').value));
  $('#simBtn').textContent = '\u23F9 Stop'; $('#simBtn').className = 'btn btn-sm btn-danger';
});
$('#simRate').addEventListener('change', () => { if (simTimer) { stopSim(); $('#simBtn').click(); } });

/* ----------------------------------------------------------------- review */
let classes = [];
async function refreshReview() {
  const view = $('#reviewView').value;
  const rows = await api(`/api/predictions?view=${view}&limit=100`);
  $('#reviewBody').innerHTML = rows.map(p => `
    <tr><td>${p.id}</td><td>${localTime(p.ts)}</td><td>${labelBadge(p.predicted_label, p.is_attack)}</td>
    <td>${pct(p.confidence, 1)}</td>
    <td>${p.severity ? `<span class="badge text-bg-${SEV_CLASS[p.severity] || 'secondary'}">${p.severity}</span>` : ''}</td>
    <td>${p.alert_status ? esc(p.alert_status.replace('_', ' ')) : ''}</td>
    <td>${p.review_state === 'reviewed' ? '<span class="text-success">&#10003; reviewed</span>' : '<span class="text-secondary">pending</span>'}</td>
    <td class="text-nowrap">
      ${p.alert_id && p.alert_status === 'open' ? `<button class="btn btn-sm btn-outline-secondary" data-ack="${p.alert_id}">Ack</button>` : ''}
      <button class="btn btn-sm btn-outline-primary" data-review="${p.id}">${p.review_state === 'reviewed' ? 'Edit' : 'Review'}</button></td></tr>`).join('')
    || '<tr><td colspan="8" class="text-secondary">Nothing here yet - start the simulator on the Overview tab.</td></tr>';
}
$('#reviewView').addEventListener('change', refreshReview);
$('#reviewBody').addEventListener('click', async e => {
  const ack = e.target.closest('[data-ack]'), rev = e.target.closest('[data-review]');
  if (ack) { await api('/api/alerts/' + ack.dataset.ack, {method: 'POST', body: {status: 'acknowledged'}}); refreshReview(); }
  if (rev) openReview(Number(rev.dataset.review));
});

const modal = new bootstrap.Modal($('#reviewModal'));
let reviewing = null;
async function openReview(id) {
  if (!classes.length) classes = await api('/api/classes');
  const p = await api('/api/predictions/' + id);
  reviewing = p;
  $('#mId').textContent = '#' + p.id;
  $('#mLabel').textContent = p.predicted_label;
  $('#mLabel').className = 'badge text-bg-' + (p.is_attack ? 'danger' : 'success');
  $('#mConf').textContent = 'confidence ' + pct(p.confidence, 1);
  $('#mTop3').textContent = 'Top candidates: ' + p.top3.map(t => `${t.label} ${pct(t.probability, 1)}`).join(' | ');
  $('#mHint').classList.toggle('d-none', !p.sim_true_label);
  $('#mHintLabel').textContent = p.sim_true_label || '';
  $('#mFinal').innerHTML = classes.filter(c => c !== p.predicted_label).map(c => `<option>${esc(c)}</option>`).join('');
  const fb = p.feedback;
  const wrong = fb && fb.verdict === 'incorrect';
  $('#vCorrect').checked = !wrong; $('#vWrong').checked = wrong;
  $('#mFinal').disabled = !wrong;
  if (wrong) $('#mFinal').value = fb.final_label;
  else if (p.sim_true_label && p.sim_true_label !== p.predicted_label) {   // pre-select the simulated truth
    $('#vWrong').checked = true; $('#vCorrect').checked = false; $('#mFinal').disabled = false;
    $('#mFinal').value = p.sim_true_label;
  }
  $('#mNotes').value = fb ? (fb.notes || '') : '';
  $('#mErr').textContent = '';
  modal.show();
}
document.querySelectorAll('input[name=verdict]').forEach(r =>
  r.addEventListener('change', () => $('#mFinal').disabled = $('#vCorrect').checked));
$('#mSubmit').addEventListener('click', async () => {
  const verdict = document.querySelector('input[name=verdict]:checked').value;
  try {
    const r = await api('/api/feedback', {method: 'POST', body: {
      prediction_id: reviewing.id, verdict,
      final_label: verdict === 'incorrect' ? $('#mFinal').value : null, notes: $('#mNotes').value}});
    modal.hide();
    toast(`Saved. ${r.pending_feedback} new labels waiting for the next model update.`);
    refresh();
  } catch (e) { $('#mErr').textContent = e.message; }
});

/* ----------------------------------------------------------------- models & adaptation */
let pollTimer = null;
async function refreshModels() {
  const [vers, runs, st] = await Promise.all([api('/api/models'), api('/api/adaptive/runs'), api('/api/adaptive/status')]);

  $('#adPending').textContent = st.pending;
  $('#adMin').textContent = st.min_required;
  $('#adBar').style.width = Math.min(100, st.pending / st.min_required * 100) + '%';
  $('#gF1').textContent = st.gate.max_macro_f1_drop;
  $('#gDR').textContent = st.gate.max_detection_drop;
  $('#gFPR').textContent = st.gate.max_fpr_increase;
  $('#badgePending').textContent = st.pending;
  $('#retrainBtn').disabled = st.running || st.pending < st.min_required;
  $('#retrainForceBtn').disabled = st.running || st.pending === 0;

  if (st.running) {
    $('#adStatus').innerHTML = '<span class="spinner-border spinner-border-sm"></span> Retraining and evaluating a candidate model...';
    if (!pollTimer) pollTimer = setInterval(refreshModels, 3000);
  } else {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    const l = st.last;
    $('#adStatus').innerHTML = !l ? '' : !l.ok ? `<span class="text-danger">Failed: ${esc(l.error)}</span>`
      : l.promoted ? `<span class="text-success">&#10003; ${esc(l.candidate_version)} promoted: ${esc(l.reasons.join('; '))}</span>`
      : `<span class="text-warning">${esc(l.candidate_version)} rejected: ${esc(l.reasons.join('; '))}</span>`;
  }

  const statusBadge = s => `<span class="badge text-bg-${{active: 'success', rejected: 'danger', retired: 'secondary'}[s] || 'secondary'}">${s}</span>`;
  $('#verBody').innerHTML = vers.map(v => `
    <tr><td><b>${esc(v.version)}</b> <span class="text-secondary small">${esc(v.algorithm)}</span></td>
    <td>${esc(v.parent_version || '-')}</td><td>${esc(v.trigger_type || '')}</td><td>${statusBadge(v.status)}</td>
    <td>${(v.metrics.macro_f1 ?? 0).toFixed(4)}</td><td>${pct(v.metrics.detection_rate)}</td><td>${pct(v.metrics.false_positive_rate, 3)}</td>
    <td class="small text-secondary">${esc(v.metrics.basis || '')}</td><td class="small">${esc(v.created_at || '')}</td>
    <td>${v.status === 'active' ? '' : `<button class="btn btn-sm btn-outline-info" data-promote="${v.version}" data-status="${v.status}">${v.status === 'rejected' ? 'Force activate' : 'Activate'}</button>`}</td></tr>`).join('');

  const asc = vers.slice().reverse();
  const cv = chart('chVersions', {
    type: 'bar',
    data: {labels: [], datasets: [
      {label: 'Macro-F1', data: [], backgroundColor: '#0d6efd', yAxisID: 'y'},
      {label: 'False-positive rate (%)', data: [], backgroundColor: '#fd7e14', yAxisID: 'y2'}]},
    options: {maintainAspectRatio: false, animation: false, scales: {
      y: {beginAtZero: true, max: 1, position: 'left'},
      y2: {beginAtZero: true, position: 'right', grid: {drawOnChartArea: false}}}}
  });
  cv.data.labels = asc.map(v => v.version + (v.status === 'rejected' ? ' (rejected)' : ''));
  cv.data.datasets[0].data = asc.map(v => v.metrics.macro_f1 ?? 0);
  cv.data.datasets[1].data = asc.map(v => +((v.metrics.false_positive_rate ?? 0) * 100).toFixed(3));
  cv.update();

  $('#runBody').innerHTML = runs.map(r => {
    const mp = r.metrics_parent || {}, mc = r.metrics_candidate || {};
    const d = (a, b, f) => (a == null || b == null) ? '-' : `${f(a)} &rarr; ${f(b)}`;
    return `<tr><td>${r.id}</td><td class="small">${esc(r.started_at || '')}</td>
      <td>${esc(r.parent_version)} &rarr; ${esc(r.candidate_version || '-')}</td>
      <td>${r.n_feedback_new} (${r.n_train_new ?? '-'}/${r.n_val ?? '-'})</td>
      <td>${d(mp.macro_f1, mc.macro_f1, x => x.toFixed(4))}</td>
      <td>${d(mp.false_positive_rate, mc.false_positive_rate, x => pct(x, 3))}</td>
      <td>${d(mp.val_accuracy, mc.val_accuracy, x => pct(x, 1))}</td>
      <td>${r.status === 'failed' ? '<span class="badge text-bg-danger">failed</span>'
        : r.status === 'running' ? '<span class="badge text-bg-warning">running</span>'
        : `<span class="badge text-bg-${r.decision === 'promoted' ? 'success' : 'secondary'}">${esc(r.decision)}</span>`}</td>
      <td class="small">${esc(r.error || (r.reasons || []).join('; '))}</td></tr>`;
  }).join('') || '<tr><td colspan="9" class="text-secondary">No retraining runs yet.</td></tr>';
}

$('#verBody').addEventListener('click', async e => {
  const b = e.target.closest('[data-promote]');
  if (!b) return;
  if (!confirm(`Activate ${b.dataset.promote}? Live predictions will use it immediately.`)) return;
  try {
    await api(`/api/models/${b.dataset.promote}/promote`, {method: 'POST', body: {force: b.dataset.status === 'rejected'}});
    toast(b.dataset.promote + ' is now active'); refresh();
  } catch (err) { toast(err.message); }
});

async function startRetrain(force) {
  try {
    await api('/api/adaptive/retrain', {method: 'POST', body: {force}});
    toast('Retraining started'); refreshModels();
  } catch (e) { toast(e.message); }
}
$('#retrainBtn').addEventListener('click', () => startRetrain(false));
$('#retrainForceBtn').addEventListener('click', () => {
  if (confirm('Retrain with fewer labels than recommended? Small batches are less likely to pass the gate.')) startRetrain(true);
});

/* ----------------------------------------------------------------- main loop */
async function refresh() {
  try {
    if (currentTab === 'overview') await refreshOverview();
    else if (currentTab === 'review') { await refreshReview(); await refreshOverview(); }
    else await Promise.all([refreshModels(), refreshOverview()]);
  } catch (e) { if (e.message !== 'auth') console.error(e); }
}
refresh();
setInterval(() => { if (!document.hidden && !$('#reviewModal').classList.contains('show')) refresh(); }, 3000);
