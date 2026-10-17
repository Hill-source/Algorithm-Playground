import { KMeans } from './kmeans-model.js';
let model, version = 0, running = false, timer, delay = 500;
const stop = () => { running = false; clearTimeout(timer); };
const snapshot = () => postMessage({ type: 'snapshot', version, running, ...model.snapshot() });
function tick() {
  try {
    if (!running) return;
    model.step();
    if (model.converged) stop();
    snapshot();
    if (running) timer = setTimeout(tick, delay);
  } catch (error) { stop(); postMessage({ type: 'error', version, message: error.message }); }
}
onmessage = ({ data: d }) => {
  try {
    if (d.type === 'init') {
      stop(); version = d.version; model = new KMeans(d.points, d.k, d.seed); snapshot();
    } else if (d.version === version && model) {
      if (d.type === 'step') { stop(); model.step(); snapshot(); }
      else if (d.type === 'run' && !running && !model.converged) { running = true; tick(); }
      else if (d.type === 'pause') { stop(); snapshot(); }
      else if (d.type === 'speed') {
        if (!Number.isFinite(d.delay) || d.delay < 50 || d.delay > 2000) throw Error('Invalid animation speed.');
        delay = d.delay;
      }
    }
  } catch (error) { stop(); postMessage({ type: 'error', version, message: error.message }); }
};
