import { random } from './model.js';

export function sampleRegression(name = 'quadratic', count = 100, noise = 0.08, seed = 42) {
  if (!['line', 'quadratic', 'cubic'].includes(name)) throw Error('Unknown dataset.');
  if (!Number.isInteger(count) || count < 2 || count > 1000) throw Error('Use 2–1,000 points.');
  if (!Number.isFinite(noise) || noise < 0 || noise > 0.3) throw Error('Noise must be between 0 and 0.3.');
  const rng = random(seed);
  return Array.from({ length: count }, (_, i) => {
    const x = -0.95 + 1.9 * i / (count - 1);
    const clean = name === 'line' ? 0.65 * x + 0.1 :
      name === 'quadratic' ? 1.2 * x * x - 0.5 : 1.8 * x ** 3 - 0.9 * x;
    return { x, y: Math.max(-1, Math.min(1, clean + (rng() * 2 - 1) * noise)) };
  });
}

export function validateRegressionPoints(points, min = 2) {
  if (!Array.isArray(points) || points.length < min || points.length > 1000) throw Error(`Provide ${min}–1,000 points.`);
  return points.map((p, i) => {
    if (!p || ![p.x, p.y].every(value => Number.isFinite(value) && value >= -1 && value <= 1)) {
      throw Error(`Row ${i + 1}: x and y must be numbers between −1 and 1.`);
    }
    return { x: p.x, y: p.y };
  });
}

export function parseRegressionCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(line => line.trim());
  if (/^x\s*,\s*y$/i.test(lines[0]?.trim())) lines.shift();
  return validateRegressionPoints(lines.map((line, i) => {
    const cells = line.split(',').map(value => value.trim());
    if (cells.length !== 2 || cells.some(value => !value)) throw Error(`Row ${i + 1}: expected x,y.`);
    const [x, y] = cells.map(Number);
    return { x, y };
  }));
}

export function splitRegression(points, seed = 42) {
  const shuffled = points.map(p => ({ ...p }));
  const rng = random(seed + 19);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const n = points.length >= 5 ? Math.floor(points.length * 0.2) : 0;
  return { train: shuffled.slice(n), test: shuffled.slice(0, n) };
}
