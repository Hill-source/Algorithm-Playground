import { sampleRegression, parseRegressionCSV } from './regression-data.js';
import { attachPointBrush } from './brush.js';

const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./regression-worker.js', import.meta.url), { type: 'module' });
let config = { degree: 2, rate: 0.1, seed: 42 };
let preset = 'quadratic', points = sampleRegression(), custom = false;
let version = 0, state = null, epochTarget = 2000, mode = 'draw', probe = 0.3, brushStart = null;
const powers = ['1', 'x', 'x²', 'x³', 'x⁴', 'x⁵', 'x⁶'];
const send = (type, extra = {}) => worker.postMessage({ type, version, ...extra });
const number = value => value == null ? '—' : Math.abs(value) < 0.0001 && value !== 0 ? value.toExponential(2) : value.toFixed(4);
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function safe(fn) { return (...args) => { try { return fn(...args); } catch (error) { message(error.message, true); } }; }
function readConfig() {
  const degree = Number($('degree').value), rate = Number($('rate').value), seed = Number($('seed').value);
  if (!$('seed').value || !Number.isInteger(seed) || seed < 0 || seed > 999999) throw Error('Choose a seed from 0 to 999999.');
  return { degree, rate, seed };
}
function reset(next = config) {
  config = next; version++; state = null;
  send('init', { ...config, points, epochTarget });
  $('train').disabled = true; $('step').disabled = true; $('status').textContent = 'Resetting';
  $('architecture').textContent = `${config.degree + 1} coefficients`;
  $('features-label').textContent = powers.slice(0, config.degree + 1).join(', ');
}
function describe() {
  document.querySelectorAll('[data-preset]').forEach(button => {
    const selected = !custom && button.dataset.preset === preset;
    button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', String(selected));
  });
  $('dataset-description').textContent = custom ? 'Your own points. Hold and drag to paint more.' :
    { line: 'A straight trend with optional noise.', quadratic: 'A curved dataset. Try fitting it with different degrees.', cubic: 'An S-shaped trend. Try degree 3 or higher.' }[preset];
}
function dataChanged(text) { describe(); reset(); message(text + ' Model and history reset.'); }
function resample() {
  points = sampleRegression(preset, Number($('count').value), Number($('noise').value), config.seed);
  custom = false; dataChanged('Sampled ' + points.length + ' points.');
}

worker.onmessage = ({ data: d }) => {
  if (d.version !== version) return;
  if (d.type === 'error') {
    message(d.message, true); $('status').textContent = 'Paused';
    if (state) state.running = false;
    $('train').textContent = '▶ Start training'; return;
  }
  if (d.type === 'probe') {
    if (state) { state.inspection = d.inspection; renderModel(); drawMap(); }
    return;
  }
  state = d; epochTarget = d.epochTarget;
  $('epoch').textContent = d.epoch.toLocaleString();
  $('epoch-caption').textContent = 'of ' + epochTarget.toLocaleString() + ' target';
  $('loss').textContent = number(d.metrics?.loss);
  $('test-rmse').textContent = number(d.validation?.rmse);
  $('r2').textContent = 'Test R²: ' + (d.validation?.r2 == null ? '—' : d.validation.r2.toFixed(3));
  $('r2').title = !d.validation ? 'No held-out points.' : d.validation.r2 == null ? 'Undefined for constant test targets.' : '1 is a perfect fit; values below 0 are worse than predicting the test mean.';
  $('split-count').textContent = `${d.train.length} train / ${d.test.length} test`;
  $('point-count').textContent = points.length + ' points';
  $('status').textContent = d.running ? 'Training' : d.epoch >= epochTarget ? 'Target reached' : d.epoch ? 'Paused' : 'Ready';
  $('train').textContent = d.running ? 'Ⅱ Pause training' : '▶ Start training';
  $('train').disabled = d.train.length < 2 || d.epoch >= epochTarget;
  $('step').disabled = d.train.length < 2 || d.epoch >= epochTarget;
  renderModel(); drawMap(); drawLoss();
};
worker.onerror = () => { message('Model execution stopped. Reload the page to restart.', true); $('train').disabled = true; $('step').disabled = true; };

function canvas(id) {
  const element = $(id), box = element.getBoundingClientRect(), ratio = Math.min(devicePixelRatio || 1, 2);
  element.width = Math.round(box.width * ratio); element.height = Math.round(box.height * ratio);
  const ctx = element.getContext('2d'); ctx.scale(ratio, ratio);
  return { ctx, w: box.width, h: box.height };
}
function predict(x) { return state.inspection.weights.reduce((sum, weight, i) => sum + weight * x ** i, 0); }
function drawMap() {
  if (!state) return;
  const { ctx, w, h } = canvas('map');
  const X = x => (x + 1) * w / 2, Y = y => (1 - y) * h / 2;
  ctx.fillStyle = '#121923'; ctx.fillRect(0, 0, w, h);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
  for (let i = 1; i < 10; i++) {
    ctx.strokeStyle = i === 5 ? '#ffffff40' : '#ffffff0c'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(i * w / 10, 0); ctx.lineTo(i * w / 10, h);
    ctx.moveTo(0, i * h / 10); ctx.lineTo(w, i * h / 10); ctx.stroke();
  }
  if ($('residuals').checked) {
    ctx.strokeStyle = '#83e8ce35'; ctx.lineWidth = 1;
    for (const point of points) {
      ctx.beginPath(); ctx.moveTo(X(point.x), Y(point.y)); ctx.lineTo(X(point.x), Y(predict(point.x))); ctx.stroke();
    }
  }
  ctx.strokeStyle = '#ffcc85'; ctx.lineWidth = 2.5; ctx.beginPath();
  state.curve.forEach((y, i) => { if (i === 0) ctx.moveTo(i * w / 200, Y(y)); else ctx.lineTo(i * w / 200, Y(y)); });
  ctx.stroke();
  const pending = brushStart === null ? [] : points.slice(brushStart);
  for (const [set, held] of [[state.train, false], [state.test, true], [pending, false]]) {
    for (const point of set) {
      ctx.beginPath(); ctx.arc(X(point.x), Y(point.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = held ? '#191e29' : '#83e8ce'; ctx.fill();
      ctx.strokeStyle = held ? '#ac9bfa' : '#0d2224'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  if (mode === 'inspect') {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = '#ffffff70'; ctx.beginPath();
    ctx.moveTo(X(probe), 0); ctx.lineTo(X(probe), h); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(X(probe), Y(predict(probe)), 6, 0, Math.PI * 2); ctx.strokeStyle = '#fff'; ctx.stroke();
  }
  ctx.restore(); ctx.fillStyle = '#a0abbd'; ctx.font = '11px monospace';
  ctx.fillText('1', 6, 14); ctx.fillText('0', 6, h / 2 - 6); ctx.fillText('−1', 6, h - 6);
}
function renderModel() {
  if (!state) return;
  const n = state.inspection;
  $('probe-x').textContent = n.x.toFixed(2); $('prediction').textContent = number(n.output);
  $('equation').textContent = 'ŷ = ' + n.weights.map((weight, i) =>
    (i ? (weight < 0 ? ' − ' : ' + ') : weight < 0 ? '−' : '') + Math.abs(weight).toFixed(3) + (i ? powers[i] : '')).join('');
  $('coefficients').replaceChildren();
  n.weights.forEach((weight, i) => {
    const row = document.createElement('tr');
    for (const value of [i ? powers[i] : '1 (bias)', number(weight), number(n.gradients?.[i]), number(n.gradients ? -config.rate * n.gradients[i] : null)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    $('coefficients').append(row);
  });
}
function drawLoss() {
  if (!state) return;
  const { ctx, w, h } = canvas('curve'), left = 48, right = w - 15, top = 15, bottom = h - 28;
  const maximum = Math.max(0.001, ...state.history.flatMap(row => [row.loss ?? 0, row.testLoss ?? 0])) * 1.1;
  ctx.font = '10px monospace'; ctx.fillStyle = '#a0abbd';
  for (let i = 0; i < 4; i++) {
    const y = top + i * (bottom - top) / 3;
    ctx.strokeStyle = '#ffffff12'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText((maximum * (1 - i / 3)).toFixed(3), left - 7, y + 3);
  }
  for (const [key, color] of [['loss', '#83e8ce'], ['testLoss', '#ac9bfa']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); let first = true;
    for (const row of state.history) {
      if (row[key] == null) continue;
      const x = left + row.epoch / Math.max(1, state.epoch) * (right - left), y = bottom - row[key] / maximum * (bottom - top);
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.textAlign = 'left'; ctx.fillText('0', left, h - 8); ctx.textAlign = 'right'; ctx.fillText(state.epoch + ' epochs', right, h - 8);
  if (!state.epoch) { ctx.textAlign = 'center'; ctx.fillText('Start training to trace the loss', (left + right) / 2, h / 2); }
}

document.querySelectorAll('[data-preset]').forEach(button => button.onclick = safe(() => { preset = button.dataset.preset; resample(); }));
$('count').oninput = () => { $('count-value').value = $('count').value; };
$('noise').oninput = () => { $('noise-value').value = Number($('noise').value).toFixed(2); };
$('resample').onclick = safe(resample);
$('degree').onchange = safe(() => { reset(readConfig()); message('Degree changed. Data preserved; model reset.'); });
$('apply').onclick = safe(() => { reset(readConfig()); message('Settings applied. Model reset; data preserved. Resample to use the new seed for sample noise.'); });
$('reset').onclick = () => { reset(); message('Coefficients reset to zero.'); };
$('train').onclick = () => { send(state?.running ? 'pause' : 'run'); message(state?.running ? 'Training paused.' : 'Training the model.'); };
$('step').onclick = () => { send('step'); message('One batch gradient update.'); };
$('epoch-form').onsubmit = event => {
  event.preventDefault();
  const input = $('epoch-target'), value = Number(input.value);
  if (!Number.isSafeInteger(value) || value < 1) { input.setCustomValidity('Enter a positive whole number.'); input.reportValidity(); return; }
  input.setCustomValidity(''); send('setEpochTarget', { epochTarget: value }); message('Epoch target updated. Learned coefficients are preserved.');
};
$('epoch-target').oninput = () => $('epoch-target').setCustomValidity('');
function setMode(next) {
  mode = next;
  for (const name of ['draw', 'inspect']) { $(name).classList.toggle('selected', mode === name); $(name).setAttribute('aria-pressed', String(mode === name)); }
  $('point-action').textContent = mode === 'draw' ? 'Add point' : 'Predict y'; $('py').disabled = mode === 'inspect'; drawMap();
}
$('draw').onclick = () => setMode('draw'); $('inspect').onclick = () => setMode('inspect');
$('residuals').onchange = drawMap;
function inspect(x) { probe = x; $('px').value = x.toFixed(3); send('probe', { x }); }
$('point-action').onclick = safe(() => {
  const x = Number($('px').value), y = Number($('py').value);
  if (!$('px').value || !Number.isFinite(x) || Math.abs(x) > 1) throw Error('x must be between −1 and 1.');
  if (mode === 'inspect') { inspect(x); return; }
  if (!$('py').value || !Number.isFinite(y) || Math.abs(y) > 1) throw Error('y must be between −1 and 1.');
  if (points.length >= 1000) throw Error('The limit is 1,000 points.');
  points.push({ x, y }); custom = true; dataChanged('Added a point.');
});
attachPointBrush($('map'), {
  isDrawing: () => mode === 'draw',
  onStart() {
    if (!state) return false;
    if (points.length >= 1000) { message('The limit is 1,000 points.', true); return false; }
    send('pause'); version++; state.running = false; brushStart = points.length;
    $('status').textContent = 'Drawing'; $('train').disabled = true; $('step').disabled = true; return true;
  },
  onPoint(u, v) {
    const x = u * 2 - 1, y = 1 - v * 2;
    points.push({ x, y }); $('px').value = x.toFixed(3); $('py').value = y.toFixed(3);
    return points.length < 1000;
  },
  onUpdate() { $('point-count').textContent = points.length + ' points'; drawMap(); },
  onEnd() { const added = points.length - brushStart; brushStart = null; custom = true; dataChanged(`Painted ${added} points.` + (points.length === 1000 ? ' Point limit reached.' : '')); },
  onInspect(u) { inspect(u * 2 - 1); }
});
$('clear').onclick = () => { points = []; custom = true; dataChanged('Plot cleared. Add at least two points.'); };
$('about').onclick = () => $('about-dialog').showModal(); $('import-open').onclick = () => $('import-dialog').showModal();
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => $(button.dataset.close).close());
$('csv-file').onchange = async () => {
  try {
    const file = $('csv-file').files[0]; if (!file) return;
    if (file.size > 100000) throw Error('Choose a CSV smaller than 100 KB.');
    $('csv').value = await file.text(); $('import-error').textContent = '';
  } catch (error) { $('import-error').textContent = error.message; }
};
$('import-confirm').onclick = () => {
  try { const next = parseRegressionCSV($('csv').value); points = next; custom = true; $('import-dialog').close(); $('import-error').textContent = ''; dataChanged('Imported ' + points.length + ' points.'); }
  catch (error) { $('import-error').textContent = error.message; }
};
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { drawMap(); drawLoss(); }, 80); });
describe(); reset();
