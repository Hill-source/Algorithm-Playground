import { PolynomialRegression } from './regression-model.js';
import { validateRegressionPoints, splitRegression } from './regression-data.js';

let model, train = [], test = [], version = 0, epoch = 0, epochTarget = 2000;
let running = false, timer, probe = 0.3, history = [];
function target(value) {
  if (!Number.isSafeInteger(value) || value < 1) throw Error('Enter a positive whole number for the epoch target.');
  return value;
}
function stop() { running = false; clearTimeout(timer); }
function snapshot() {
  const metrics = model.metrics(train), validation = model.metrics(test);
  const row = { epoch, loss: metrics?.loss ?? null, testLoss: validation?.loss ?? null };
  if (history.at(-1)?.epoch === epoch) history[history.length - 1] = row;
  else history.push(row);
  if (history.length > 300) history.splice(1, 1);
  postMessage({ type: 'snapshot', version, epoch, epochTarget, running, metrics, validation,
    train, test, history, curve: Array.from({ length: 201 }, (_, i) => model.predict(-1 + i / 100)),
    inspection: model.inspect(probe, train) });
}
function tick() {
  try {
    if (!running) return;
    const until = performance.now() + 18;
    do { model.trainEpoch(train); epoch++; } while (performance.now() < until && epoch < epochTarget);
    if (epoch >= epochTarget) stop();
    snapshot();
    if (running) timer = setTimeout(tick, 30);
  } catch (error) { stop(); postMessage({ type: 'error', version, message: error.message }); }
}
onmessage = ({ data: d }) => {
  try {
    if (d.type === 'init') {
      stop(); version = d.version;
      const points = validateRegressionPoints(d.points, 0);
      epochTarget = target(d.epochTarget);
      model = new PolynomialRegression(d.degree, d.rate);
      ({ train, test } = splitRegression(points, d.seed));
      epoch = 0; history = []; snapshot();
    } else if (d.version === version && model) {
      if (d.type === 'run' || d.type === 'step') {
        if (train.length < 2) throw Error('Add at least two points before training.');
        if (epoch >= epochTarget) throw Error('Increase the epoch target to continue training.');
        if (d.type === 'step') { stop(); model.trainEpoch(train); epoch++; snapshot(); }
        else if (!running) { running = true; tick(); }
      } else if (d.type === 'pause') { stop(); snapshot(); }
      else if (d.type === 'setEpochTarget') { epochTarget = target(d.epochTarget); if (epoch >= epochTarget) stop(); snapshot(); }
      else if (d.type === 'probe') {
        model.features(d.x); probe = d.x;
        postMessage({ type: 'probe', version, inspection: model.inspect(probe, train) });
      }
    }
  } catch (error) { stop(); postMessage({ type: 'error', version, message: error.message }); }
};
