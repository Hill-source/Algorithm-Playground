import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { PolynomialRegression } from '../dist/regression-model.js';
import { sampleRegression, parseRegressionCSV, splitRegression } from '../dist/regression-data.js';

test('backpropagation matches numerical derivatives for every polynomial coefficient', () => {
  for (let degree = 1; degree <= 6; degree++) {
    const model = new PolynomialRegression(degree, 0.1);
    model.weights = model.weights.map((_, i) => (i - 2) * 0.17);
    const points = sampleRegression('cubic', 30, 0.02);
    const gradients = model.gradients(points), before = [...model.weights], epsilon = 1e-6;
    for (let i = 0; i < model.weights.length; i++) {
      model.weights[i] = before[i] + epsilon;
      const above = model.metrics(points).loss;
      model.weights[i] = before[i] - epsilon;
      const below = model.metrics(points).loss;
      model.weights[i] = before[i];
      assert.ok(Math.abs(gradients[i] - (above - below) / (2 * epsilon)) < 1e-8);
    }
    model.trainEpoch(points);
    model.weights.forEach((weight, i) => assert.ok(Math.abs(weight - (before[i] - 0.1 * gradients[i])) < 1e-12));
  }
});

test('learns held-out line, quadratic and cubic data; degree one underfits a parabola', () => {
  for (const [name, degree, expected] of [['line', 1, [0.1, 0.65]], ['quadratic', 2, [-0.5, 0, 1.2]], ['cubic', 3, [0, -0.9, 0, 1.8]]]) {
    const { train, test } = splitRegression(sampleRegression(name, 100, 0));
    const model = new PolynomialRegression(degree, 0.2);
    for (let i = 0; i < 6000; i++) model.trainEpoch(train);
    assert.ok(model.metrics(test).rmse < 0.002, name);
    model.weights.forEach((weight, i) => assert.ok(Math.abs(weight - expected[i]) < 0.02, name));
  }
  const points = sampleRegression('quadratic', 100, 0);
  const linear = new PolynomialRegression(1), quadratic = new PolynomialRegression(2);
  for (let i = 0; i < 2000; i++) { linear.trainEpoch(points); quadratic.trainEpoch(points); }
  assert.ok(quadratic.metrics(points).loss < linear.metrics(points).loss * 0.001);
});

test('CSV, metrics, input limits and repeatable split handle boundary cases', () => {
  assert.deepEqual(parseRegressionCSV('x,y\r\n-1,1\r\n1,-1'), [{ x: -1, y: 1 }, { x: 1, y: -1 }]);
  for (const csv of ['x,y\n0,0', '0,0\n1,', '0,0\nInfinity,0', '0,0\n0,2', '0,0\n0,0,1']) assert.throws(() => parseRegressionCSV(csv));
  const points = sampleRegression('quadratic', 100, 0.1, 17);
  assert.deepEqual(points, sampleRegression('quadratic', 100, 0.1, 17));
  const split = splitRegression(points, 17);
  assert.deepEqual(split, splitRegression(points, 17));
  assert.equal(split.train.length, 80); assert.equal(split.test.length, 20);
  assert.ok(split.train.every(p => !split.test.some(q => q.x === p.x)));
  const tiny = splitRegression(points.slice(0, 2)); assert.equal(tiny.test.length, 0);
  const model = new PolynomialRegression();
  assert.equal(model.metrics([]), null);
  assert.equal(model.metrics([{ x: 0, y: 0 }, { x: 1, y: 0 }]).r2, null);
  assert.throws(() => model.trainEpoch([])); assert.throws(() => model.predict(2));
  assert.throws(() => new PolynomialRegression(7)); assert.throws(() => new PolynomialRegression(2, 1));
  model.weights = [0, 0, 3]; assert.equal(model.predict(1), 3); // linear output isn't clamped
});

test('regression worker runs, steps, preserves weights on target changes, probes and rejects empty training', async () => {
  const url = new URL('../dist/regression-worker.js', import.meta.url).href;
  const worker = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.postMessage=d=>parentPort.postMessage(d);globalThis.onmessage=null;import(${JSON.stringify(url)}).then(()=>parentPort.on('message',data=>onmessage({data})));`, { eval: true });
  const request = (data, predicate = () => true) => new Promise((resolve, reject) => {
    const listener = d => { if (predicate(d)) { clearTimeout(timer); worker.off('message', listener); resolve(d); } };
    const timer = setTimeout(() => { worker.off('message', listener); reject(Error('Worker timeout')); }, 5000);
    worker.on('message', listener); worker.postMessage({ version: 1, ...data });
  });
  try {
    let r = await request({ type: 'init', degree: 3, rate: 0.1, seed: 42, points: sampleRegression('cubic'), epochTarget: 3 });
    assert.equal(r.curve.length, 201); assert.equal(r.inspection.weights.length, 4);
    r = await request({ type: 'run' }, d => !d.running); assert.equal(r.epoch, 3);
    const weights = r.inspection.weights;
    r = await request({ type: 'setEpochTarget', epochTarget: 5 }); assert.deepEqual(r.inspection.weights, weights);
    r = await request({ type: 'step' }); assert.equal(r.epoch, 4);
    r = await request({ type: 'probe', x: -0.4 }); assert.equal(r.inspection.x, -0.4);
    r = await request({ type: 'setEpochTarget', epochTarget: 1 }); assert.equal(r.epoch, 4);
    r = await request({ type: 'setEpochTarget', epochTarget: 0 }); assert.equal(r.type, 'error');
    r = await request({ type: 'init', version: 2, degree: 1, rate: 0.1, seed: 42, points: [], epochTarget: 10 });
    assert.equal(r.metrics, null); assert.equal(r.validation, null);
    r = await request({ type: 'run', version: 2 }); assert.equal(r.type, 'error');
  } finally { await worker.terminate(); }
});
