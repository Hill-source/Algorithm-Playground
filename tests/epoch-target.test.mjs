import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

test('editable targets stop exactly, extend learned runs, validate input and exceed 20,000', async () => {
  const url = new URL('../dist/worker.js', import.meta.url).href;
  const worker = new Worker(`
    const { parentPort } = require('node:worker_threads');
    globalThis.postMessage = data => parentPort.postMessage(data);
    globalThis.onmessage = null;
    import(${JSON.stringify(url)}).then(() => {
      parentPort.on('message', data => onmessage({ data }));
    });
  `, { eval: true });

  const request = (data, predicate = () => true) => new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      worker.off('message', listener);
      reject(Error('Timed out waiting for worker'));
    }, 10000);
    const listener = result => {
      if (!predicate(result)) return;
      clearTimeout(timeout);
      worker.off('message', listener);
      resolve(result);
    };
    worker.on('message', listener);
    worker.postMessage({ version: 1, ...data });
  });

  try {
    let result = await request({
      type: 'init', hidden: [2], rate: 0.01, seed: 42, epochTarget: 3,
      points: [{ x: 0, y: 0, label: 0 }, { x: 1, y: 1, label: 1 }]
    });
    assert.equal(result.epochTarget, 3);
    result = await request({ type: 'run' }, r => r.type === 'snapshot' && !r.running);
    assert.equal(result.epoch, 3);
    const weights = result.network.weights;
    const metrics = result.metrics;
    result = await request({ type: 'setEpochTarget', epochTarget: 5 });
    assert.equal(result.epoch, 3);
    assert.equal(result.epochTarget, 5);
    assert.deepEqual(result.network.weights, weights);
    assert.deepEqual(result.metrics, metrics);
    result = await request({ type: 'step' });
    assert.equal(result.epoch, 4);
    result = await request({ type: 'run' }, r => r.type === 'snapshot' && !r.running);
    assert.equal(result.epoch, 5);
    result = await request({ type: 'setEpochTarget', epochTarget: 2 });
    assert.equal(result.epoch, 5);
    assert.equal(result.running, false);
    assert.equal((await request({ type: 'step' })).type, 'error');
    for (const target of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      assert.equal((await request({ type: 'setEpochTarget', epochTarget: target })).type, 'error');
    }
    result = await request({ type: 'pause' });
    assert.equal(result.epochTarget, 2);
    result = await request({ type: 'setEpochTarget', epochTarget: 20001 });
    assert.equal(result.epoch, 5);
    result = await request({ type: 'run' }, r => r.type === 'snapshot' && !r.running);
    assert.equal(result.epoch, 20001);
    assert.equal(result.epochTarget, 20001);
    assert.ok(result.history.length <= 300);

    // A lower target can stop an active run without rewinding completed epochs.
    await request({ type: 'setEpochTarget', epochTarget: 1000000000 });
    result = await request({ type: 'run' }, r => r.type === 'snapshot' && r.running);
    const reached = result.epoch;
    result = await request({ type: 'setEpochTarget', epochTarget: 1 },
      r => r.type === 'snapshot' && r.epochTarget === 1);
    assert.equal(result.running, false);
    assert.ok(result.epoch >= reached);
    const stoppedEpoch = result.epoch;
    result = await request({ type: 'pause' });
    assert.equal(result.epoch, stoppedEpoch);
  } finally {
    await worker.terminate();
  }
});
