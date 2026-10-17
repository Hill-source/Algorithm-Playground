import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { sample } from '../dist/data.js';


test('worker handles initialize, step, run, pause, inspect, stale requests and empty data', async () => {
    const url = new URL('../dist/worker.js', import.meta.url).href;
    const w = new Worker(`const {parentPort}=require('node:worker_threads');globalThis.postMessage=d=>parentPort.postMessage(d);globalThis.onmessage=null;import(${JSON.stringify(url)}).then(()=>parentPort.on('message',data=>onmessage({data})));`, { eval: true });
    const request = (data, predicate = () => true) => new Promise((resolve, reject) => { const timer = setTimeout(() => { w.off('message', listener); reject(Error('Worker timeout')); }, 5000); const listener = d => { if (predicate(d)) { clearTimeout(timer); w.off('message', listener); resolve(d); } }; w.on('message', listener); w.postMessage(data); });
    try { let r = await request({ type: 'init', version: 1, hidden: [4, 3], rate: .01, seed: 42, points: sample('clusters') }); assert.equal(r.epoch, 0); assert.equal(r.grid.length, 3025); r = await request({ type: 'step', version: 1 }); assert.equal(r.epoch, 1); r = await request({ type: 'run', version: 1 }); assert.equal(r.running, true); r = await request({ type: 'pause', version: 1 }, d => d.running === false); const epoch = r.epoch; assert.ok(epoch > 1); r = await request({ type: 'probe', version: 1, input: [.2, .8] }, d => d.type === 'probe'); assert.equal(r.network.activations[0][0], .2); w.postMessage({ type: 'run', version: 0 }); r = await request({ type: 'pause', version: 1 }); assert.equal(r.epoch, epoch); r = await request({ type: 'init', version: 2, hidden: [4, 3], rate: .01, seed: 42, points: [] }); assert.equal(r.metrics, null); assert.equal(r.epoch, 0); r = await request({ type: 'run', version: 2 }); assert.equal(r.type, 'error'); } finally { await w.terminate(); }
});
