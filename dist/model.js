// Adapted from the supplied neural.js. Weight layout, sigmoid, backward pass,
// uniform initialization and sampling with replacement preserve the original math.

export const sigmoid = x => 1 / (1 + Math.exp(-x));

export function random(seed = 42) {
  let a = seed >>> 0;

  return () => {
    a += 0x6D2B79F5;

    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Layer {
  constructor(inputCount, outputCount, rng, rate) {
    this.rate = rate;
    this.inputs = Array(inputCount).fill(0);
    this.outputs = Array(outputCount).fill(0);
    this.biases = [];
    this.weights = [];

    for (let i = 0; i < outputCount; i++) {
      this.biases.push(rng() * 2 - 1);

      this.weights.push(
        Array.from(
          { length: inputCount },
          () => rng() * 2 - 1
        )
      );
    }
  }

  feedForward(inputs) {
    this.inputs = inputs;
      
    // Matrix multiplication: [weights] x [inputs] + [biases] = [results]
    for (let i = 0; i < this.weights.length; i++) {
      let value = 0;

      for (let j = 0; j < inputs.length; j++) {
        value += this.weights[i][j] * inputs[j];
      }

      this.outputs[i] = sigmoid(value + this.biases[i]);
    }

    return this.outputs;
  }

  feedBackward(inputs, targets = null, isLastLayer = false) {
    const errors = inputs.map(
      (v, i) =>
        (isLastLayer ? targets[i] - v : v) *
        this.outputs[i] *
        (1 - this.outputs[i])
    );

    const outputs = this.inputs.map((_, i) =>
      this.weights.reduce(
        (s, row, j) => s + row[i] * errors[j],
        0
      )
    );

    for (let j = 0; j < this.weights.length; j++) {
      for (let i = 0; i < this.inputs.length; i++) {
        this.weights[j][i] +=
          this.inputs[i] * errors[j] * this.rate;
      }

      this.biases[j] += errors[j] * this.rate;
    }

    return outputs;
  }
}

export class NeuralNetwork {
  constructor(sizes = [2, 4, 3, 1], rate = 0.01, seed = 42) {
    if (
      sizes[0] !== 2 ||
      ![1, 2, 3, 4, 5, 6, 7].includes(sizes.at(-1)) ||
      sizes.length < 3 ||
      sizes.length > 5 ||
      sizes.some(n => !Number.isInteger(n) || n < 1 || n > 12)
    ) {
      throw Error('Use 1–3 hidden layers of 1–12 neurons and 1–7 output neurons.');
    }

    if (
      !Number.isFinite(rate) ||
      rate < 0.001 ||
      rate > 2
    ) {
      throw Error('Learning rate must be between 0.001 and 2.');
    }

    this.rng = random(seed);
    this.sizes = [...sizes];
    this.outputCount = sizes.at(-1);
    this.classCount = this.outputCount === 1 ? 2 : this.outputCount;

    this.layers = sizes
      .slice(1)
      .map((n, i) => new Layer(sizes[i], n, this.rng, rate));
  }

  predict(input) {
    if (
      input.length !== 2 ||
      input.some(x => !Number.isFinite(x) || x < 0 || x > 1)
    ) {
      throw Error('Inputs must be two numbers between 0 and 1.');
    }

    let output = input;

    for (const layer of this.layers) {
      output = layer.feedForward(output);
    }

    return this.outputCount === 1 ? output[0] : [...output];
  }

  // Binary mode preserves the original scalar output. Multiclass mode uses
  // independent sigmoid scores and selects the largest (ties choose class 0 first).
  classification(output) {
    const scores = this.outputCount === 1 ? [1 - output, output] : [...output];
    const label = this.outputCount === 1
      ? (output >= 0.5 ? 1 : 0)
      : scores.indexOf(Math.max(...scores));
    return { scores, label, score: scores[label] };
  }

  targets(label) {
    if (!Number.isInteger(label) || label < 0 || label >= this.classCount) {
      throw Error(`Label must be an integer from 0 to ${this.classCount - 1}.`);
    }
    return this.outputCount === 1
      ? [label]
      : Array.from({ length: this.outputCount }, (_, i) => Number(i === label));
  }

  trainEpoch(points) {
    if (!points.length) {
      throw Error('Add training points first.');
    }

    for (let i = 0; i < points.length; i++) {
      const p = points[Math.floor(this.rng() * points.length)];
      const targets = this.targets(p.label);
      const prediction = this.predict([p.x, p.y]);
      let backward = this.outputCount === 1 ? [prediction] : prediction;

      for (let j = this.layers.length - 1; j >= 0; j--) {
        backward = this.layers[j].feedBackward(
          backward,
          targets,
          j === this.layers.length - 1
        );
      }
    }
  }

  metrics(points) {
    if (!points.length) {
      return null;
    }

    let loss = 0;
    let correct = 0;

    for (const p of points) {
      const y = this.predict([p.x, p.y]);

      const targets = this.targets(p.label);
      const outputs = this.outputCount === 1 ? [y] : y;
      // Half the sum across outputs, averaged across examples below.
      loss += outputs.reduce((sum, value, i) => sum + 0.5 * (targets[i] - value) ** 2, 0);
      correct += Number(this.classification(y).label === p.label);
    }

    return {
      loss: loss / points.length,
      accuracy: correct / points.length
    };
  }

  inspect(input) {
    const output = this.predict(input);

    return {
      output,
      ...this.classification(output),
      activations: [
        input,
        ...this.layers.map(l => [...l.outputs])
      ],
      weights: this.layers.map(l =>
        l.weights.map(w => [...w])
      ),
      biases: this.layers.map(l => [...l.biases])
    };
  }
}