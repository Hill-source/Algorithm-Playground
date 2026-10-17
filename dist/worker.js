import { NeuralNetwork } from './model.js';
import { validatePoints, split, validateClassCount } from './data.js';

let model;
let train = [];
let test = [];
let epoch = 0;
let epochTarget = 20000;
let running = false;
let timer;
let version = 0;
let probe = [0.5, 0.5];
let history = [];

function validateEpochTarget(value) {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw Error('Enter a positive whole number for the epoch target.');
  }
  return value;
}

function snapshot() {
  const metrics = model.metrics(train);
  const validation = model.metrics(test);
  const entry = { epoch, loss: metrics?.loss ?? null, testLoss: validation?.loss ?? null };
  if (history.at(-1)?.epoch === epoch) history[history.length - 1] = entry;
  else history.push(entry);
  if (history.length > 300) history.splice(1, 1);
  const grid = [];
  for (let y = 0; y < 55; y++) {
    for (let x = 0; x < 55; x++) grid.push(model.predict([x / 54, y / 54]));
  }
  postMessage({
    type: 'snapshot', version, epoch, epochTarget, running,
    metrics, validation, history, grid, train, test,
    network: model.inspect(probe)
  });
}

function stop() {
  running = false;
  clearTimeout(timer);
}

function tick() {
  if (!running) return;
  try {
    const until = performance.now() + 22;
    do {
      if (epoch >= epochTarget) break;
      model.trainEpoch(train);
      epoch++;
    } while (performance.now() < until && epoch < epochTarget);
    if (epoch >= epochTarget) stop();
    snapshot();
    if (running) timer = setTimeout(tick, 25);
  } catch (e) {
    stop();
    postMessage({ type: 'error', version, message: e.message });
  }
}

onmessage = ({ data: d }) => {
  try {
    if (d.type === 'init') {
      stop();
      version = d.version;
      const nextTarget = validateEpochTarget(d.epochTarget ?? epochTarget);
      const classCount = validateClassCount(d.classCount ?? 2);
      const points = d.points.length ? validatePoints(d.points, 1, classCount) : [];
      ({ train, test } = split(points, d.seed));
      model = new NeuralNetwork([2, ...d.hidden, classCount === 2 ? 1 : classCount], d.rate, d.seed);
      epochTarget = nextTarget;
      epoch = 0;
      history = [];
      snapshot();
    } else if (d.version === version && model) {
      if (d.type === 'setEpochTarget') {
        // Changing the target keeps all learned weights and the epoch count.
        epochTarget = validateEpochTarget(d.epochTarget);
        if (epoch >= epochTarget) stop();
        snapshot();
      } else if (d.type === 'run') {
        if (!train.length) throw Error('Add at least two points before training.');
        if (epoch >= epochTarget) {
          throw Error('Epoch target reached. Increase the target to continue training.');
        }
        if (!running) {
          running = true;
          tick();
        }
      } else if (d.type === 'pause') {
        stop();
        snapshot();
      } else if (d.type === 'step') {
        stop();
        if (epoch >= epochTarget) {
          throw Error('Epoch target reached. Increase the target to continue training.');
        }
        model.trainEpoch(train);
        epoch++;
        snapshot();
      } else if (d.type === 'probe') {
        probe = d.input;
        postMessage({ type: 'probe', version, network: model.inspect(probe) });
      }
    }
  } catch (e) {
    stop();
    postMessage({ type: 'error', version, message: e.message });
  }
};
