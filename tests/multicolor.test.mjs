import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { NeuralNetwork } from '../dist/model.js';
import { sample, split, parseCSV, validatePoints } from '../dist/data.js';

test('2–7 color samples and splits retain every class, with strict CSV labels', () => {
  for (let count = 2; count <= 7; count++) {
    for (const name of ['clusters', 'circle']) {
      const points = sample(name, 140, 42, count);
      assert.deepEqual(points, sample(name, 140, 42, count));
      assert.equal(validatePoints(points, 2, count).length, 140);
      const { train, test } = split(points);
      assert.equal(train.length + test.length, points.length);
      assert.equal(new Set(train.map(p => p.label)).size, count);
      assert.equal(new Set(test.map(p => p.label)).size, count);
      const held = new Set(test.map(p => JSON.stringify(p)));
      assert.ok(train.every(p => !held.has(JSON.stringify(p))));
    }
  }
  assert.deepEqual(parseCSV('x,y,label\n0.1,0.2,0\n0.8,0.9,6', 7).map(p => p.label), [0, 6]);
  for (const label of ['7', '-1', '1.5', 'NaN']) {
    assert.throws(() => parseCSV(`0.1,0.2,0\n0.8,0.9,${label}`, 7));
  }
  assert.throws(() => sample('xor', 100, 42, 7));
  assert.throws(() => sample('clusters', 100, 42, 8));
});

test('multi-output backprop matches numerical gradients through all output paths', () => {
  const model = new NeuralNetwork([2, 3, 7], 0.01, 42);
  const point = { x: 0.3, y: 0.7, label: 6 };
  const epsilon = 1e-5;
  const gradients = [];
  for (const layer of model.layers) {
    for (const values of [...layer.weights, layer.biases]) {
      for (let i = 0; i < values.length; i++) {
        const initial = values[i];
        values[i] = initial + epsilon;
        const above = model.metrics([point]).loss;
        values[i] = initial - epsilon;
        const below = model.metrics([point]).loss;
        values[i] = initial;
        gradients.push({ values, i, initial, gradient: (above - below) / (2 * epsilon) });
      }
    }
  }
  model.trainEpoch([point]);
  for (const { values, i, initial, gradient } of gradients) {
    assert.ok(Math.abs(values[i] - (initial - 0.01 * gradient)) < 1e-9);
  }
});

test('seven output neurons learn seven held-out color clusters', () => {
  const { train, test } = split(sample('clusters', 140, 42, 7));
  const model = new NeuralNetwork([2, 12, 8, 7], 0.3, 42);
  const initial = model.metrics(train).loss;
  for (let epoch = 0; epoch < 2500; epoch++) model.trainEpoch(train);
  assert.ok(model.metrics(train).loss < initial * 0.1);
  assert.ok(model.metrics(test).accuracy >= 0.95);
  assert.equal(new Set(test.map(p => model.inspect([p.x, p.y]).label)).size, 7);
  const inspected = model.inspect([0.2, 0.8]);
  assert.equal(inspected.scores.length, 7);
  assert.equal(inspected.activations.at(-1).length, 7);
  assert.equal(inspected.score, Math.max(...inspected.scores));
  assert.throws(() => model.trainEpoch([{ x: 0.1, y: 0.2, label: 7 }]));
});

test('worker supports seven colors, snapshots, stepping and target extension', async () => {
  const url = new URL('../dist/worker.js', import.meta.url).href;
  const worker = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.postMessage=d=>parentPort.postMessage(d);globalThis.onmessage=null;import(${JSON.stringify(url)}).then(()=>parentPort.on('message',data=>onmessage({data})));`, { eval: true });
  const request = data => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('Worker timeout')), 5000);
    worker.once('message', d => { clearTimeout(timer); resolve(d); });
    worker.postMessage({ version: 1, ...data });
  });
  try {
    let state = await request({ type: 'init', hidden: [12, 8], rate: 0.3, seed: 42,
      points: sample('clusters', 140, 42, 7), classCount: 7, epochTarget: 1 });
    assert.equal(state.network.scores.length, 7);
    assert.equal(state.grid.length, 3025);
    assert.ok(state.grid.every(scores => scores.length === 7 && scores.every(Number.isFinite)));
    assert.equal(new Set(state.train.map(p => p.label)).size, 7);
    state = await request({ type: 'step' });
    assert.equal(state.epoch, 1);
    const weights = state.network.weights;
    state = await request({ type: 'setEpochTarget', epochTarget: 2 });
    assert.deepEqual(state.network.weights, weights);
    state = await request({ type: 'run' });
    assert.equal(state.epoch, 2);
    assert.equal(state.running, false);
  } finally { await worker.terminate(); }
});
