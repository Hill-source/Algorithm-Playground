// Polynomial regression is linear in its learned coefficients, even for x², x³…
// Backpropagation: loss -> prediction -> weighted polynomial features.
export class PolynomialRegression {
  constructor(degree = 2, rate = 0.1) {
    if (!Number.isInteger(degree) || degree < 1 || degree > 6) {
      throw Error('Choose a degree from 1 to 6.');
    }
    if (!Number.isFinite(rate) || rate < 0.001 || rate > 0.2) {
      throw Error('Learning rate must be between 0.001 and 0.2.');
    }
    this.degree = degree;
    this.rate = rate;
    this.weights = Array(degree + 1).fill(0);
  }

  features(x) {
    if (!Number.isFinite(x) || x < -1 || x > 1) throw Error('x must be between −1 and 1.');
    return Array.from({ length: this.degree + 1 }, (_, power) => x ** power);
  }

  predict(x) {
    return this.features(x).reduce((sum, feature, i) => sum + feature * this.weights[i], 0);
  }

  gradients(points) {
    if (!points.length) throw Error('Add training points first.');
    const gradients = Array(this.weights.length).fill(0);
    for (const { x, y } of points) {
      const features = this.features(x);
      const prediction = features.reduce((sum, value, i) => sum + value * this.weights[i], 0);
      const error = prediction - y; // d(½(ŷ − y)²)/dŷ
      for (let i = 0; i < gradients.length; i++) {
        // dŷ/dw_i = x^i; chain rule accumulates every example's contribution.
        gradients[i] += error * features[i] / points.length;
      }
    }
    return gradients;
  }

  trainEpoch(points) {
    const gradients = this.gradients(points);
    const next = this.weights.map((weight, i) => weight - this.rate * gradients[i]);
    if (next.some(value => !Number.isFinite(value))) throw Error('Training diverged. Reset with a smaller learning rate.');
    this.weights = next;
  }

  metrics(points) {
    if (!points.length) return null;
    const mean = points.reduce((sum, p) => sum + p.y, 0) / points.length;
    let squaredError = 0, variance = 0;
    for (const p of points) {
      squaredError += (this.predict(p.x) - p.y) ** 2;
      variance += (p.y - mean) ** 2;
    }
    return { loss: squaredError / (2 * points.length), rmse: Math.sqrt(squaredError / points.length),
      r2: variance < 1e-12 ? null : 1 - squaredError / variance };
  }

  inspect(x, points) {
    const features = this.features(x);
    return { x, output: this.predict(x), features, weights: [...this.weights],
      contributions: features.map((value, i) => value * this.weights[i]),
      gradients: points.length ? this.gradients(points) : null };
  }
}
