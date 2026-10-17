import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { KMeans, clusterSample, parseClusterCSV, distanceSquared } from '../dist/kmeans-model.js';

test('assignment uses nearest centers and updates use every assigned point equally', () => {
  const points = [{ x: 0.1, y: 0.1 }, { x: 0.2, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.9, y: 0.9 }];
  const model = new KMeans(points, 2, 42);
  assert.ok(model.assignments.every(a => a === -1));
  const initialized = structuredClone(model.centers);
  model.step(); assert.equal(model.phase, 'update');
  assert.deepEqual(model.centers, initialized);
  model.assignments.forEach((label, i) => assert.equal(distanceSquared(points[i], model.centers[label]), Math.min(...model.centers.map(c => distanceSquared(points[i], c)))));
  const assignments = [...model.assignments];
  model.step(); assert.equal(model.iteration, 1); assert.equal(model.phase, 'assign');
  assert.deepEqual(model.assignments, assignments);
  for (let label = 0; label < 2; label++) {
    const set = points.filter((_, i) => assignments[i] === label);
    assert.equal(model.centers[label].x, set.reduce((s, p) => s + p.x, 0) / set.length);
    assert.equal(model.centers[label].y, set.reduce((s, p) => s + p.y, 0) / set.length);
  }
  for (let i = 0; i < 20 && !model.converged; i++) model.step();
  assert.ok(model.converged);
  const xs = model.centers.map(c => c.x).sort();
  assert.ok(Math.abs(xs[0] - 0.15) < 1e-12); assert.ok(Math.abs(xs[1] - 0.85) < 1e-12);
});

test('1–7 clusters converge reproducibly with nonincreasing inertia and conserved membership', () => {
  for (let k = 1; k <= 7; k++) {
    const points = clusterSample('blobs', 140, 42);
    const model = new KMeans(points, k, 19), same = new KMeans(points, k, 19);
    assert.deepEqual(model.centers, same.centers);
    for (let phase = 0; phase < 300 && !model.converged; phase++) { model.step(); same.step(); }
    assert.ok(model.converged);
    assert.deepEqual(model.snapshot(), same.snapshot());
    assert.equal(model.snapshot().counts.reduce((s, n) => s + n, 0), points.length);
    assert.ok(model.assignments.every(a => a >= 0 && a < k));
    model.history.slice(1).forEach((row, i) => assert.ok(row.inertia <= model.history[i].inertia + 1e-10));
    assert.ok(Math.abs(model.inertia - points.reduce((s, p, i) => s + distanceSquared(p, model.centers[model.assignments[i]]), 0)) < 1e-12);
  }
});

test('duplicates, empty clusters, insufficient points and CSV validation are explicit', () => {
  const points = Array.from({ length: 7 }, () => ({ x: 0.5, y: 0.5 }));
  const model = new KMeans(points, 7);
  for (let i = 0; i < 10 && !model.converged; i++) model.step();
  assert.ok(model.converged); assert.equal(model.inertia, 0);
  assert.equal(model.snapshot().counts.filter(n => n === 0).length, 6);
  assert.ok(model.centers.every(c => c.x === 0.5 && c.y === 0.5));
  assert.equal(new KMeans([], 3).centers.length, 0);
  assert.equal(new KMeans(points.slice(0, 2), 3).centers.length, 0);
  assert.throws(() => new KMeans([], 3).step());
  assert.throws(() => new KMeans(points, 8));
  assert.deepEqual(parseClusterCSV('x,y\r\n0,1\r\n1,0'), [{ x: 0, y: 1 }, { x: 1, y: 0 }]);
  for (const csv of ['', 'x,y', '0,', '0,2', 'Infinity,0', '0,0,1']) assert.throws(() => parseClusterCSV(csv));
});

test('worker steps through both phases, pauses, ignores stale commands and converges', async () => {
  const url = new URL('../dist/kmeans-worker.js', import.meta.url).href;
  const worker = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.postMessage=d=>parentPort.postMessage(d);globalThis.onmessage=null;import(${JSON.stringify(url)}).then(()=>parentPort.on('message',data=>onmessage({data})));`, { eval: true });
  const request = (data, predicate = () => true) => new Promise((resolve, reject) => {
    const listener = d => { if (predicate(d)) { clearTimeout(timer); worker.off('message', listener); resolve(d); } };
    const timer = setTimeout(() => { worker.off('message', listener); reject(Error('Worker timeout')); }, 5000);
    worker.on('message', listener); worker.postMessage({ version: 1, ...data });
  });
  try {
    let r = await request({ type: 'init', k: 4, seed: 42, points: clusterSample('blobs', 120) });
    assert.equal(r.centers.length, 4); assert.equal(r.inertia, null);
    r = await request({ type: 'step' }); assert.equal(r.phase, 'update');
    r = await request({ type: 'step' }); assert.equal(r.iteration, 1);
    worker.postMessage({ type: 'step', version: 0 });
    r = await request({ type: 'pause' }); assert.equal(r.iteration, 1); assert.equal(r.running, false);
    worker.postMessage({ type: 'speed', version: 1, delay: 50 });
    r = await request({ type: 'run' }, d => d.converged); assert.equal(r.running, false);
    r = await request({ type: 'init', version: 2, k: 7, seed: 42, points: [] });
    assert.equal(r.centers.length, 0);
    r = await request({ type: 'step', version: 2 }); assert.equal(r.type, 'error');
  } finally { await worker.terminate(); }
});
