import { random } from './model.js';

export const distanceSquared = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
export function validateClusterPoints(points) {
  if (!Array.isArray(points) || points.length > 1000) throw Error('Use up to 1,000 points.');
  return points.map((p, i) => {
    if (!p || ![p.x, p.y].every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw Error(`Row ${i + 1}: x and y must be between 0 and 1.`);
    return { x: p.x, y: p.y };
  });
}

export class KMeans {
  constructor(points, k = 3, seed = 42) {
    if (!Number.isInteger(k) || k < 1 || k > 7) throw Error('Choose 1–7 clusters.');
    this.points = validateClusterPoints(points);
    this.k = k;
    this.iteration = 0;
    this.phase = 'assign';
    this.converged = false;
    this.assignments = this.points.map(() => -1);
    this.centers = [];
    this.previousCenters = [];
    this.shift = 0;
    this.inertia = null;
    this.history = [];
    if (!points.length) return;
    if (points.length < k) return; // Allow drawing an incomplete dataset.
    // K-means++: choose real points with probability proportional to squared
    // distance from the closest existing center. Selection is reproducible.
    const rng = random(seed), chosen = new Set();
    let index = Math.floor(rng() * points.length);
    while (this.centers.length < k) {
      chosen.add(index); this.centers.push({ ...this.points[index] });
      if (this.centers.length === k) break;
      const weights = this.points.map((p, i) => chosen.has(i) ? 0 : Math.min(...this.centers.map(c => distanceSquared(p, c))));
      const total = weights.reduce((s, v) => s + v, 0);
      index = this.points.findIndex((_, i) => !chosen.has(i));
      if (total > 0) {
        let ticket = rng() * total;
        for (let i = 0; i < weights.length; i++) {
          ticket -= weights[i];
          if (weights[i] > 0 && ticket < 0) { index = i; break; }
        }
      }
    }
  }

  step() {
    if (!this.centers.length) throw Error('Add at least k points before running.');
    if (this.converged) return;
    if (this.phase === 'assign') {
      let changed = false;
      this.assignments = this.points.map((point, i) => {
        const distances = this.centers.map(center => distanceSquared(point, center));
        const minimum = Math.min(...distances);
        // Keep an existing assignment on an exact tie; otherwise choose first.
        const old = this.assignments[i];
        const label = old >= 0 && distances[old] === minimum ? old : distances.indexOf(minimum);
        if (label !== old) changed = true;
        return label;
      });
      this.converged = this.iteration > 0 && !changed;
      this.phase = this.converged ? 'done' : 'update';
    } else {
      this.previousCenters = this.centers.map(c => ({ ...c }));
      this.centers = this.centers.map((center, label) => {
        const members = this.points.filter((_, i) => this.assignments[i] === label);
        // Empty clusters retain their center; the UI explicitly calls them out.
        if (!members.length) return { ...center };
        return { x: members.reduce((s, p) => s + p.x, 0) / members.length,
          y: members.reduce((s, p) => s + p.y, 0) / members.length };
      });
      this.shift = Math.max(...this.centers.map((c, i) => Math.sqrt(distanceSquared(c, this.previousCenters[i]))));
      this.iteration++;
      this.phase = 'assign';
    }
    this.inertia = this.points.reduce((sum, p, i) => sum + distanceSquared(p, this.centers[this.assignments[i]]), 0);
    this.history.push({ step: (this.history.at(-1)?.step ?? -1) + 1, inertia: this.inertia });
    if (this.history.length > 600) this.history.splice(1, 1);
  }

  snapshot() {
    return { points: this.points, centers: this.centers, previousCenters: this.previousCenters,
      assignments: this.assignments, counts: this.centers.map((_, label) => this.assignments.filter(a => a === label).length),
      iteration: this.iteration, phase: this.phase, converged: this.converged,
      shift: this.shift, inertia: this.inertia, history: this.history };
  }
}

export function clusterSample(name, count = 120, seed = 42) {
  const rng = random(seed);
  return Array.from({ length: count }, (_, i) => {
    if (name === 'uniform') return { x: 0.05 + rng() * 0.9, y: 0.05 + rng() * 0.9 };
    if (name === 'rings') {
      const a = rng() * Math.PI * 2, r = (i % 2 ? 0.32 : 0.13) + rng() * 0.05;
      return { x: 0.5 + Math.cos(a) * r, y: 0.5 + Math.sin(a) * r };
    }
    const a = i % 4 * Math.PI / 2 + Math.PI / 4;
    return { x: 0.5 + Math.cos(a) * 0.3 + (rng() - 0.5) * 0.22,
      y: 0.5 + Math.sin(a) * 0.3 + (rng() - 0.5) * 0.22 };
  });
}

export function parseClusterCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(line => line.trim());
  if (/^x\s*,\s*y$/i.test(lines[0]?.trim())) lines.shift();
  if (!lines.length) throw Error('Provide at least one point.');
  return validateClusterPoints(lines.map((line, i) => {
    const cells = line.split(',').map(c => c.trim());
    if (cells.length !== 2 || cells.some(c => !c)) throw Error(`Row ${i + 1}: expected x,y.`);
    const [x, y] = cells.map(Number); return { x, y };
  }));
}
