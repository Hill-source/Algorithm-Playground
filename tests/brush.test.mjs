import test from 'node:test';
import assert from 'node:assert/strict';
import { attachPointBrush } from '../dist/brush.js';

test('brush spaces drag points, handles release/cancel, and keeps inspect separate', () => {
  const handlers = {};
  const previousWindow = globalThis.window;
  let blur;
  globalThis.window = { addEventListener: (_, fn) => { blur = fn; } };
  const points = [];
  let captured = false, drawing = true, starts = 0, ends = 0, inspections = 0, limit = 1000;
  const canvas = {
    addEventListener: (name, fn) => { handlers[name] = fn; },
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 200 }),
    setPointerCapture: () => { captured = true; },
    hasPointerCapture: () => captured,
    releasePointerCapture: () => { captured = false; }
  };
  const fire = (name, x, y, extra = {}) => handlers[name]({
    clientX: x + 10, clientY: y + 20, pointerId: 1,
    button: 0, buttons: 1, isPrimary: true, preventDefault() {}, ...extra
  });
  try {
    attachPointBrush(canvas, {
      isDrawing: () => drawing,
      onStart: () => { starts++; return points.length < limit; },
      onPoint: (x, y) => { points.push({ x, y }); return points.length < limit; },
      onUpdate() {}, onEnd: () => ends++, onInspect: () => inspections++
    });
    fire('pointerdown', 0, 100);
    fire('pointermove', 80, 100);
    assert.equal(points.length, 11);
    fire('pointermove', 80, 100); // stationary pointer cannot flood the dataset
    assert.equal(points.length, 11);
    fire('pointerup', 80, 100);
    assert.equal(ends, 1);
    assert.equal(captured, false);
    fire('pointermove', 100, 100);
    assert.equal(points.length, 11);
    assert.deepEqual(points[10], { x: 0.4, y: 0.5 });

    fire('pointerdown', 20, 20); // one click is one point
    fire('pointerup', 20, 20);
    assert.equal(points.length, 12);
    fire('pointerdown', 40, 40);
    fire('pointermove', 240, 40); // outside the canvas
    fire('pointerup', 240, 40);
    assert.equal(points.length, 13);
    fire('pointerdown', 40, 40);
    fire('pointercancel', 40, 40);
    assert.equal(captured, false);
    fire('pointerdown', 40, 40);
    blur();
    assert.equal(captured, false);

    const before = points.length;
    drawing = false;
    fire('pointerdown', 50, 50);
    fire('pointermove', 100, 50);
    fire('pointerup', 100, 50);
    assert.equal(inspections, 1);
    assert.equal(points.length, before);
    drawing = true;
    limit = points.length + 3;
    fire('pointerdown', 0, 0);
    fire('pointermove', 200, 0);
    assert.equal(points.length, limit);
    assert.equal(captured, false);
    assert.equal(starts, ends);
    fire('pointerdown', 0, 0, { button: 2 });
    assert.equal(points.length, limit);
  } finally { globalThis.window = previousWindow; }
});
