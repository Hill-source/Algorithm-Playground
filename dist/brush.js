// Pointer capture lets a stroke finish reliably even when released off the map.
// Spacing is measured in CSS pixels so drawing speed does not change density.
export function attachPointBrush(canvas, { isDrawing, onStart, onPoint, onUpdate, onEnd, onInspect }) {
  const spacing = 8;
  let pointer = null;
  let last = null;

  function position(event) {
    const box = canvas.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top, box };
  }

  function finish() {
    if (pointer === null) return;
    const id = pointer;
    pointer = null;
    last = null;
    if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
    onEnd();
  }

  function paint(event) {
    const { x, y, box } = position(event);
    if (x < 0 || y < 0 || x > box.width || y > box.height) {
      last = null;
      return;
    }
    const add = (px, py) => onPoint(px / box.width, py / box.height);
    if (!last) {
      last = { x, y };
      if (add(x, y) === false) finish();
    } else {
      let distance = Math.hypot(x - last.x, y - last.y);
      while (distance >= spacing && pointer !== null) {
        last = {
          x: last.x + (x - last.x) * spacing / distance,
          y: last.y + (y - last.y) * spacing / distance
        };
        if (add(last.x, last.y) === false) { finish(); break; }
        distance = Math.hypot(x - last.x, y - last.y);
      }
    }
    if (pointer !== null) onUpdate();
  }

  canvas.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0 || pointer !== null) return;
    event.preventDefault();
    if (!isDrawing()) {
      const { x, y, box } = position(event);
      onInspect(x / box.width, y / box.height);
      return;
    }
    if (onStart() === false) return;
    pointer = event.pointerId;
    canvas.setPointerCapture(pointer);
    paint(event);
  });
  canvas.addEventListener('pointermove', event => {
    if (event.pointerId !== pointer) return;
    if (!(event.buttons & 1)) { finish(); return; }
    paint(event);
  });
  canvas.addEventListener('pointerup', event => {
    if (event.pointerId !== pointer) return;
    paint(event);
    finish();
  });
  canvas.addEventListener('pointercancel', event => {
    if (event.pointerId === pointer) finish();
  });
  canvas.addEventListener('lostpointercapture', event => {
    if (event.pointerId === pointer) finish();
  });
  window.addEventListener('blur', finish);
}
