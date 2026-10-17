import { clusterSample, parseClusterCSV, distanceSquared } from './kmeans-model.js';
import { palette, classColor } from './palette.js';
import { attachPointBrush } from './brush.js';

const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./kmeans-worker.js', import.meta.url), { type: 'module' });
let config = { k: 3, seed: 42 }, preset = 'blobs', custom = false;
let points = clusterSample(preset), version = 0, state = null, mode = 'draw';
let selected = -1, inspected = null, brushStart = null;
const send = (type, extra = {}) => worker.postMessage({ type, version, ...extra });
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function safe(fn) { return (...args) => { try { return fn(...args); } catch (error) { message(error.message, true); } }; }
function readConfig() {
  const seed = Number($('seed').value), k = Number($('k').value);
  if (!$('seed').value || !Number.isInteger(seed) || seed < 0 || seed > 999999) throw Error('Choose a seed from 0 to 999999.');
  return { k, seed };
}
function reset(next = config) {
  config = next; version++; state = null; inspected = null;
  if (selected >= config.k) selected = -1;
  send('init', { ...config, points });
  send('speed', { delay: Number($('speed').value) });
  $('run').disabled = true; $('step').disabled = true; $('status').textContent = 'Initializing';
  renderButtons();
}
function describe() {
  document.querySelectorAll('[data-preset]').forEach(button => {
    const on = !custom && preset === button.dataset.preset;
    button.classList.toggle('selected', on); button.setAttribute('aria-pressed', String(on));
  });
  $('dataset-description').textContent = custom ? 'Custom unlabeled points. Drag to add more.' : {
    blobs: 'Four groups. Try different values of k on the same data.',
    uniform: 'Scattered points with no built-in groups.',
    rings: 'Concentric rings reveal the limitations of distance-based clusters.'
  }[preset];
}
function changed(text) { describe(); reset(); message(text + ' Centers and progress reset.'); }
function resample() { points = clusterSample(preset, Number($('count').value), config.seed); custom = false; changed('Sampled ' + points.length + ' points.'); }
worker.onmessage = ({ data: d }) => {
  if (d.version !== version) return;
  if (d.type === 'error') { message(d.message, true); if (state) state.running = false; $('status').textContent = 'Paused'; $('run').textContent = '▶ Start clustering'; return; }
  state = d;
  $('iteration').textContent = d.iteration;
  $('inertia').textContent = d.inertia == null ? '—' : d.inertia.toFixed(4);
  $('shift').textContent = d.iteration ? d.shift.toFixed(4) : '—';
  $('point-count').textContent = points.length + ' points';
  $('assigned').textContent = d.assignments.filter(a => a >= 0).length + ' assigned';
  $('status').textContent = !d.centers.length ? 'Need points' : d.converged ? 'Converged' : d.running ? 'Running' : 'Ready';
  $('run').textContent = d.running ? 'Ⅱ Pause clustering' : '▶ Start clustering';
  $('run').disabled = !d.centers.length || d.converged;
  $('step').disabled = !d.centers.length || d.converged;
  $('step').textContent = d.phase === 'update' ? 'Move centers' : 'Assign points';
  renderDetails(); drawMap(); drawProgress();
};
worker.onerror = () => { message('Clustering stopped. Reload the page to restart.', true); $('run').disabled = true; $('step').disabled = true; };
function selectCluster(label) { selected = label; renderButtons(); renderDetails(); drawMap(); }
function renderButtons() {
  $('cluster-buttons').replaceChildren();
  for (let label = -1; label < config.k; label++) {
    const button = document.createElement('button');
    button.textContent = label < 0 ? 'All' : String(label + 1);
    if (label >= 0) {
      const dot = document.createElement('i'); dot.className = 'dot'; dot.style.background = classColor(label); button.prepend(dot);
    }
    button.setAttribute('aria-label', label < 0 ? 'Highlight all clusters' : `Highlight cluster ${label + 1}, ${palette[label].name}`);
    button.setAttribute('aria-pressed', String(selected === label)); button.classList.toggle('selected', selected === label);
    button.onclick = () => selectCluster(label); $('cluster-buttons').append(button);
  }
}
function members(label) { return points.filter((_, i) => state.assignments[i] === label); }
function mean(set) { return { x: set.reduce((s, p) => s + p.x, 0) / set.length, y: set.reduce((s, p) => s + p.y, 0) / set.length }; }
const coords = p => `(${p.x.toFixed(3)}, ${p.y.toFixed(3)})`;
function renderDetails() {
  if (!state) return;
  const unassigned = state.assignments.every(a => a < 0);
  $('phase-title').textContent = !state.centers.length ? `Add at least ${config.k} points` : state.converged ? 'Converged · assignments unchanged' : state.phase === 'update' ? 'Next: move centers to their means' : 'Next: assign points to nearest centers';
  $('phase-detail').textContent = state.converged ? 'Every center is the mean of its members. Try another seed or k to compare solutions.' : state.phase === 'update' ? 'Colors and lines show the members used to calculate each new center.' : unassigned ? 'Gray points have no assignment yet. Compare each point with every center.' : 'Centers have moved. Membership will be recalculated in the next step; regions show the nearest current center.';
  $('centers').replaceChildren();
  state.centers.forEach((center, label) => {
    const button = document.createElement('button'); button.className = 'center-row';
    button.classList.toggle('selected', selected === label); button.setAttribute('aria-pressed', String(selected === label));
    const title = document.createElement('span'); title.textContent = `${label + 1} · ${palette[label].name}`;
    title.style.color = classColor(label);
    const count = document.createElement('span'); count.textContent = state.counts[label] + ' points';
    const info = document.createElement('small'); info.textContent = `Center ${coords(center)}` + (!unassigned && !state.counts[label] ? ' · empty; center retained' : '');
    button.append(title, count, info); button.onclick = () => selectCluster(label); $('centers').append(button);
  });
  $('selection-title').textContent = selected < 0 ? 'All clusters' : `Cluster ${selected + 1} · ${palette[selected].name}`;
  if (selected >= 0 && state.centers[selected]) {
    const set = members(selected);
    $('selection-detail').textContent = !set.length ? (unassigned ? 'Assign points first to see which points contribute to this center.' : 'No members. This center keeps its previous position.') :
      `${set.length} highlighted points contribute equally. Mean x = Σx / ${set.length} = ${mean(set).x.toFixed(4)}; mean y = Σy / ${set.length} = ${mean(set).y.toFixed(4)}. ` +
      (state.phase === 'update' ? 'This is the next center position.' : 'This is the center calculated in the last update.');
  } else $('selection-detail').textContent = 'Select a cluster to highlight all of its members and inspect their mean. Cluster colors are assigned by the algorithm.';
  $('distances').replaceChildren();
  if (inspected == null || !points[inspected]) { $('point-detail').textContent = 'In Inspect mode, click near a point to compare its distances to all centers.'; return; }
  const point = points[inspected], assigned = state.assignments[inspected];
  const distances = state.centers.map(c => Math.sqrt(distanceSquared(point, c)));
  const nearest = distances.indexOf(Math.min(...distances));
  $('point-detail').textContent = `Point ${inspected + 1} ${coords(point)} · ` + (assigned < 0 ? 'not assigned yet.' : `assigned to cluster ${assigned + 1}.`) + (nearest < 0 ? '' : ` Nearest current center: ${nearest + 1}.`);
  distances.forEach((distance, label) => {
    const row = document.createElement('div'); row.className = 'distance-row'; row.style.color = classColor(label);
    const name = document.createElement('span'); name.textContent = `${label + 1} · ${palette[label].name}`;
    const value = document.createElement('span'); value.textContent = distance.toFixed(4) + (label === nearest ? ' · nearest' : '');
    row.append(name, value); $('distances').append(row);
  });
}
function canvas(id) {
  const el = $(id), box = el.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
  el.width = Math.round(box.width * ratio); el.height = Math.round(box.height * ratio);
  const ctx = el.getContext('2d'); ctx.scale(ratio, ratio); return { ctx, w: box.width, h: box.height };
}
function drawMap() {
  if (!state) return;
  const { ctx, w, h } = canvas('map');
  ctx.fillStyle = '#121923'; ctx.fillRect(0, 0, w, h);
  if ($('regions').checked && state.centers.length) {
    for (let y = 0; y < 50; y++) for (let x = 0; x < 50; x++) {
      const distances = state.centers.map(c => distanceSquared({ x: (x + 0.5) / 50, y: (y + 0.5) / 50 }, c));
      const label = distances.indexOf(Math.min(...distances));
      ctx.fillStyle = classColor(label, selected < 0 || selected === label ? 0.17 : 0.035); ctx.fillRect(x * w / 50, y * h / 50, w / 50 + 1, h / 50 + 1);
    }
  }
  ctx.strokeStyle = '#ffffff12'; ctx.lineWidth = 1;
  for (let i = 1; i < 10; i++) { ctx.beginPath(); ctx.moveTo(i * w / 10, 0); ctx.lineTo(i * w / 10, h); ctx.moveTo(0, i * h / 10); ctx.lineTo(w, i * h / 10); ctx.stroke(); }
  points.forEach((point, i) => {
    const label = state.assignments[i] ?? -1, highlight = selected < 0 || selected === label;
    if ($('connections').checked && label >= 0 && highlight) {
      const center = state.centers[label]; ctx.strokeStyle = classColor(label, 0.24); ctx.beginPath(); ctx.moveTo(point.x * w, point.y * h); ctx.lineTo(center.x * w, center.y * h); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(point.x * w, point.y * h, inspected === i ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = label < 0 ? '#afbbc9' : classColor(label, highlight ? 1 : 0.17); ctx.fill();
    ctx.strokeStyle = inspected === i ? '#fff' : '#111722'; ctx.lineWidth = inspected === i ? 2 : 1; ctx.stroke();
  });
  state.centers.forEach((center, label) => {
    const x = center.x * w, y = center.y * h, previous = state.previousCenters[label];
    ctx.strokeStyle = classColor(label); ctx.lineWidth = 2;
    if (previous && distanceSquared(center, previous) > 1e-12) {
      ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(previous.x * w, previous.y * h); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(previous.x * w, previous.y * h, 7, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(x, y - 10); ctx.lineTo(x + 10, y); ctx.lineTo(x, y + 10); ctx.lineTo(x - 10, y); ctx.closePath();
    ctx.fillStyle = classColor(label); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.stroke();
    ctx.font = 'bold 11px monospace'; ctx.fillStyle = '#10131b'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(label + 1), x, y);
  });
}
function drawProgress() {
  if (!state) return;
  const { ctx, w, h } = canvas('curve'), left = 45, right = w - 16, top = 16, bottom = h - 28;
  const max = Math.max(0.001, ...state.history.map(row => row.inertia)) * 1.1;
  ctx.font = '10px monospace'; ctx.fillStyle = '#a0abbd';
  for (let i = 0; i < 4; i++) {
    const y = top + (bottom - top) * i / 3; ctx.strokeStyle = '#ffffff12'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText((max * (1 - i / 3)).toFixed(2), left - 6, y + 3);
  }
  ctx.strokeStyle = '#83e8ce'; ctx.lineWidth = 2; ctx.beginPath();
  state.history.forEach((row, i) => {
    const x = left + row.step / Math.max(1, state.history.at(-1).step) * (right - left), y = bottom - row.inertia / max * (bottom - top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }); ctx.stroke();
  ctx.textAlign = 'left'; ctx.fillText('0', left, h - 8); ctx.textAlign = 'right'; ctx.fillText((state.history.at(-1)?.step ?? 0) + ' phase steps', right, h - 8);
  if (state.history.length < 2) { ctx.textAlign = 'center'; ctx.fillText('Assign and update to trace inertia', (left + right) / 2, h / 2); }
}

$('k').onchange = safe(() => { reset(readConfig()); message('Changed k. Data preserved; centers reset.'); });
$('apply').onclick = safe(() => { reset(readConfig()); message('Settings applied. Resample to regenerate data with this seed.'); });
$('reset').onclick = () => { reset(); message('Centers reset with the same seed.'); };
$('run').onclick = () => { send(state?.running ? 'pause' : 'run'); message(state?.running ? 'Clustering paused.' : 'Alternating assignments and center updates.'); };
$('step').onclick = () => send('step');
$('speed').onchange = () => send('speed', { delay: Number($('speed').value) });
$('count').oninput = () => { $('count-value').value = $('count').value; };
$('resample').onclick = safe(resample);
document.querySelectorAll('[data-preset]').forEach(button => button.onclick = safe(() => { preset = button.dataset.preset; resample(); }));
function setMode(next) {
  mode = next; for (const name of ['draw', 'inspect']) { $(name).classList.toggle('selected', name === mode); $(name).setAttribute('aria-pressed', String(name === mode)); }
  $('point-action').textContent = mode === 'draw' ? 'Add point' : 'Inspect nearest';
}
$('draw').onclick = () => setMode('draw'); $('inspect').onclick = () => setMode('inspect');
$('connections').onchange = drawMap; $('regions').onchange = drawMap;
function inspect(x, y) {
  if (!points.length) { message('Add points first to inspect their distances.', true); return; }
  const distances = points.map(p => distanceSquared(p, { x, y })); inspected = distances.indexOf(Math.min(...distances));
  $('px').value = points[inspected].x.toFixed(3); $('py').value = points[inspected].y.toFixed(3);
  renderDetails(); drawMap();
}
$('point-action').onclick = safe(() => {
  const x = Number($('px').value), y = Number($('py').value);
  if (!$('px').value || !$('py').value || ![x, y].every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw Error('Coordinates must be between 0 and 1.');
  if (mode === 'inspect') { inspect(x, y); return; }
  if (points.length >= 1000) throw Error('The limit is 1,000 points.');
  points.push({ x, y }); custom = true; changed('Added a point.');
});
attachPointBrush($('map'), {
  isDrawing: () => mode === 'draw',
  onStart() {
    if (!state) return false;
    if (points.length >= 1000) { message('The limit is 1,000 points.', true); return false; }
    send('pause'); version++; state.running = false; brushStart = points.length;
    $('status').textContent = 'Drawing'; $('run').disabled = true; $('step').disabled = true; return true;
  },
  onPoint(x, y) { points.push({ x, y }); $('px').value = x.toFixed(3); $('py').value = y.toFixed(3); return points.length < 1000; },
  onUpdate() { $('point-count').textContent = points.length + ' points'; drawMap(); },
  onEnd() { const count = points.length - brushStart; brushStart = null; custom = true; changed(`Painted ${count} points.` + (points.length === 1000 ? ' Point limit reached.' : '')); },
  onInspect: inspect
});
$('clear').onclick = () => { points = []; custom = true; changed('Map cleared.'); };
$('about').onclick = () => $('about-dialog').showModal(); $('import-open').onclick = () => $('import-dialog').showModal();
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => $(button.dataset.close).close());
$('csv-file').onchange = async () => {
  try { const file = $('csv-file').files[0]; if (!file) return; if (file.size > 100000) throw Error('Choose a CSV smaller than 100 KB.'); $('csv').value = await file.text(); $('import-error').textContent = ''; }
  catch (error) { $('import-error').textContent = error.message; }
};
$('import-confirm').onclick = () => {
  try { const next = parseClusterCSV($('csv').value); points = next; custom = true; $('import-dialog').close(); $('import-error').textContent = ''; changed('Imported ' + points.length + ' points.'); }
  catch (error) { $('import-error').textContent = error.message; }
};
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawMap(); drawProgress(); }, 80); });
describe(); reset();
