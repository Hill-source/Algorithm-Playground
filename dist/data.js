import { random } from './model.js';

export function validateClassCount(classCount) {
  if (!Number.isInteger(classCount) || classCount < 2 || classCount > 7) {
    throw Error('Choose 2–7 colors.');
  }
  return classCount;
}

export function sample(name, count = 100, seed = 42, classCount = 2) {
  validateClassCount(classCount);
  if (name === 'xor' && classCount !== 2) throw Error('XOR uses two colors.');
  const rng = random(seed);
  const pts = [];

  for (let i = 0; i < count; i++) {
    const label = i % classCount;
    let x, y;

    if (classCount > 2) {
      if (name === 'clusters') {
        const angle = label * Math.PI * 2 / classCount - Math.PI / 2;
        x = 0.5 + Math.cos(angle) * 0.34 + (rng() - 0.5) * 0.12;
        y = 0.5 + Math.sin(angle) * 0.34 + (rng() - 0.5) * 0.12;
      } else {
        const angle = rng() * Math.PI * 2;
        const radius = (label + 0.2 + rng() * 0.55) * 0.48 / classCount;
        x = 0.5 + Math.cos(angle) * radius;
        y = 0.5 + Math.sin(angle) * radius;
      }
    } else if (name === 'clusters') {
      x = 0.25 + 0.5 * label + (rng() - 0.5) * 0.38;
      y = 0.28 + 0.44 * label + (rng() - 0.5) * 0.38;
    } else if (name === 'xor') {
      const side = rng() > 0.5 ? 1 : 0;

      x = 0.2 + 0.6 * side + (rng() - 0.5) * 0.3;
      y = 0.2 + 0.6 * (label ? 1 - side : side) + (rng() - 0.5) * 0.3;
    } else {
      const a = rng() * Math.PI * 2;
      const r = label ? 0.28 + rng() * 0.17 : rng() * 0.18;

      x = 0.5 + Math.cos(a) * r;
      y = 0.5 + Math.sin(a) * r;
    }

    pts.push({ x, y, label });
  }

  return pts;
}

export function validatePoints(points, min = 2, classCount = 2) {
  validateClassCount(classCount);
  if (
    !Array.isArray(points) ||
    points.length < min ||
    points.length > 1000
  ) {
    throw Error('Provide 2–1,000 points.');
  }

  return points.map((p, i) => {
    if (
      !p ||
      ![p.x, p.y].every(
        v =>
          typeof v === 'number' &&
          Number.isFinite(v) &&
          v >= 0 &&
          v <= 1
      ) ||
      !Number.isInteger(p.label) || p.label < 0 || p.label >= classCount
    ) {
      throw Error(
        `Row ${i + 1}: x and y must be 0–1; label must be an integer from 0 to ${classCount - 1}.`
      );
    }

    return {
      x: p.x,
      y: p.y,
      label: p.label
    };
  });
}

export function parseCSV(text, classCount = 2) {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .filter(l => l.trim());

  if (/^x\s*,\s*y\s*,\s*label$/i.test(lines[0]?.trim())) {
    lines.shift();
  }

  return validatePoints(
    lines.map((line, i) => {
      const cells = line.split(',').map(s => s.trim());

      if (cells.length !== 3 || cells.some(s => s === '')) {
        throw Error(`Row ${i + 1}: expected x,y,label.`);
      }

      const [x, y, label] = cells.map(Number);

      return { x, y, label };
    }),
    2,
    classCount
  );
}

export function split(points, seed = 42) {
  const rng = random(seed + 19);
  const train = [];
  const test = [];

  for (const label of [...new Set(points.map(p => p.label))].sort((a, b) => a - b)) {
    const group = points
      .filter(p => p.label === label)
      .map(p => ({ ...p }));

    for (let i = group.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));

      [group[i], group[j]] = [group[j], group[i]];
    }

    const n =
      group.length >= 5
        ? Math.max(1, Math.floor(group.length * 0.2))
        : 0;

    test.push(...group.slice(0, n));
    train.push(...group.slice(n));
  }

  return { train, test };
}