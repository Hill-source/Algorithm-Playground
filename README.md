# Algorithm Playground

An interactive browser app built around the supplied Neural-Network.zip project. No package installation, backend service, API key, or GPU is required.

## Run locally

Install Node.js 20 or newer, open this folder in a terminal, then run:

```sh
npm start
```

Open http://127.0.0.1:4173. The server binds only to your computer. Stop with Ctrl+C. Alternatively, serve `dist/` using any static HTTP server. Opening index.html through `file://` will not work because the app uses ES modules and a module Web Worker.

Run verification with `npm test`. There are no npm dependencies or build step. Google Fonts is optional; system fonts are used when offline. The model, data, charts, and controls work without external services once the local server is running.

## What was in the original

The uploaded project consists of `index.html`, `style.css`, `main.js`, `utils.js`, `neural.js`, and `visualiser.js`. It has no library dependencies or pretrained weights. The original executable logic is preserved unchanged in `original/` for reference; Git metadata from the ZIP is intentionally excluded.

`main.js` creates a 2–4–3–1 network, collects clicked coordinates and binary labels, and alternates drawing and fitting in an animation loop. The XOR arrays in that file are demonstration data but are not used by the active training loop. `visualiser.js` evaluates a 100×100 input grid and renders its scalar outputs in grayscale. `utils.js` provides sigmoid and its derivative.

Inputs are two normalized numbers `[x,y]` in `[0,1]`. Targets are one binary number `[0]` or `[1]`. The output is a sigmoid score, not a calibrated probability; the new UI uses 0.5 as its classification threshold. Canvas y increases downwards, as in the original.

`neural.js` implements dense layers with row-per-output weight matrices, independent uniform `[-1,1]` weights and biases, sigmoid activation in every layer, and backpropagation with stochastic gradient descent at learning rate 0.01. Each epoch makes N updates, each choosing a random training example with replacement (not a shuffled pass). The original `MSE` method returns half the summed squared error for one example. With a single output, the UI reports `0.5 × mean((target - score)^2)` across the complete evaluated set.

## Architecture

| Component | Responsibility |
| --- | --- |
| `dist/index.html`, `style.css` | Accessible controls, responsive layout, explanatory dialogs |
| `dist/app.js` | UI state, validated configuration, canvas rendering, worker messages |
| `dist/model.js` | DOM-free neural network and seeded random generator |
| `dist/palette.js` | Seven stable class colors and display names |
| `dist/data.js` | Sample datasets, strict CSV validation, stratified split |
| `dist/worker.js` | Model ownership, training scheduling, inference and metrics |
| `serve.mjs` | Dependency-free local static server |
| `original/` | Unmodified supplied source, outside published static assets |
| `tests/` | Numerical parity, convergence, input validation and worker lifecycle checks |

The browser sends `init`, `run`, `pause`, `step`, `setEpochTarget`, and `probe` messages. The worker returns snapshots containing actual metrics, a decision grid, weights, and activations. Version numbers prevent old experiment responses from overwriting a newer experiment. Worker batches target about 22 ms of training and yield between batches; snapshots use a 55×55 grid to reduce rendering cost. Training stops at a user-editable epoch target (20,000 by default). Set any positive safe integer; there is no fixed 20,000-epoch ceiling. Raising the target lets a completed run resume without resetting learned weights or history. Lowering it to the current count or below pauses training without undoing completed epochs. There is no fabricated data in the metrics or model visuals.

## Preserved and adapted

The model retains the original initialization order, matrix layout, forward equations, backward equations, error propagation before weight updates, and sampling-with-replacement training. A parity test initializes the original source and the adapted model with the same RNG and verifies predictions after training to a tolerance of 1e-14.

Changes: random generation is injectable and seeded; learning rate is configurable; the awkward original `new Array(targets.length)` hidden-layer allocation is replaced by correctly sized arrays; verbose per-example console logging is removed; model execution moves to a Worker; epochs now count actual passes instead of animation frames. Drawing is separate from model logic. Determinism holds for identical datasets, seed, configuration and number of epochs; wall-clock batch lengths vary by device.

## Experiments

Start with Clusters, then try XOR and Circle. The default reproduces the original architecture and learning rate. For a nonlinear problem, try a rate of 0.3 and hidden layers `8, 6`; convergence is not guaranteed. Training and test loss, test accuracy, current output, and signed connection strengths show what the network is doing. Larger hidden layers omit tiny numeric node labels for readability; output remains available as text.

Inspect mode accepts map clicks or numeric coordinates. Add mode adds points in any selected color (class IDs 0–6). Changing data or applying configuration resets weights and history. Reset weights repeats the same initialization. Seed changes apply to the model/split immediately on Apply; Resample uses the active seed to recreate the dataset. Sample count is applied by Resample or a preset button. Imports stay in memory on your device and replace the dataset; refreshing loses the experiment.

CSV accepts an optional `x,y,label` header, 2–1,000 data rows, coordinates in `[0,1]`, and integer labels 0–6. Imports automatically expand the color count if necessary. File imports are capped at 100 KB. Example:

```csv
x,y,label
0.1,0.2,0
0.8,0.9,1
0.2,0.3,0
0.9,0.8,1
```

The split is deterministic and stratified: each class with at least 5 points holds out floor(20%) examples; smaller classes use all examples for training. Test examples never update weights. Empty test sets show an em dash rather than misleading accuracy. Both losses are evaluated after each batch on the full relevant set. Charts keep the initial point and most recent 299 snapshots to bound memory. Adding examples creates a new split and restarts the experiment; test scores should not be compared as if the test set stayed fixed.

An optional feature-detected WebMCP read tool exposes the same current experiment state to compatible browser agents. It adds no network service and makes no changes to the experiment. Other browsers work normally.

## Hosting

Publish the contents of `dist/` on a static host that supports module Workers with JavaScript MIME types. No secrets or user data are stored in source. Source and setup instructions are included here; only `dist/` assets are served by the website.

## Experiments with up to seven colors

Select **Number of colors** (2–7). Clusters provides one group per color; Circle
provides concentric bands. XOR remains a two-color experiment. Select a colored
button above the map to add points of that class. The legend maps class IDs to
Mint, Violet, Orange, Blue, Pink, Yellow, and Red. CSV labels use these IDs (0–6).
Changing the count regenerates preset data and resets the model. Custom points
are preserved; reducing the count is rejected if it would remove one of their
labels. Clear the points or load a compatible dataset first.

For seven clusters, try hidden layers `12, 8`, learning rate `0.3`, and an epoch
target of `2500`. Press **Apply & reset network**, then **Start training**.
Inspect any coordinate to see all class scores and the predicted color.
Concentric bands are harder and may need more training or a different seed.

Two-color mode retains the original single sigmoid output and 0.5 threshold,
including numerical parity with the original code. For 3–7 colors, the worker
creates that many output neurons. A point labeled 6 in seven-color mode uses
target `[0,0,0,0,0,0,1]`. The original sigmoid and backward equations apply to
every output; propagated signals sum all downstream paths. Prediction selects
the highest score, with ties choosing the lowest class ID. These independent
sigmoid scores are not calibrated probabilities and do not have to sum to 1.
Binary inspection displays `[1-output, output]` for its two color scores.

`predict()` returns a scalar for one output, or a copied array for multiple
outputs. `inspect()` additionally returns `scores`, `label` and `score` (the
winning score). Loss is half the sum of squared errors across outputs, averaged
across points; its scale therefore changes with the number of outputs.
The stratified split operates on every class. All decision-map cells and
inspection values are computed by the worker from the actual model.

The test suite checks the original binary behavior, seven-class learning,
finite-difference gradients for every weight and bias, per-class splits, CSV
validation, multi-output worker execution, and editable epoch targets.


## Polynomial regression page

Use the **Regression** link in the header or open `/regression.html`. This page
shares the neural-network layout and supports polynomial degrees 1–6, noisy
line/parabola/cubic samples, CSV imports (`x,y`), brush drawing, numeric point
entry, residual lines, prediction inspection, editable epoch targets, and
training/test loss, test RMSE and R². Inputs and targets range from −1 to 1;
the plot uses the mathematical convention of positive y pointing upwards.
Predictions are not restricted to that range and may be clipped visually.

The model is `y_hat = sum(w_i * x^i)` for i=0 through degree. All coefficients
start at zero. One epoch uses the whole training set: forward prediction,
backpropagation of `error = y_hat - y`, averaging `error * x^i` for each
coefficient, then `w_i -= learningRate * gradient_i`. Loss is half mean squared
error. This is batch gradient descent, not the neural page's stochastic update.
There is no closed-form solver or sigmoid. A polynomial is nonlinear in x but
linear in its learned coefficients. High powers are correlated and can require
many epochs to learn; increasing the degree does not guarantee a better fit.

Start with Parabola, degree 2, learning rate 0.1, and 2,000 epochs. Compare degree
1 on the same data. Inspect the coefficient table to see each current gradient
and proposed next update. Switch to Cubic and degree 3 for an S-shaped curve.
Samples and the 80/20 random split are repeatable by seed. Fewer than five points
use the full dataset for training. Test data never updates coefficients. R² is
undefined for constant test targets and may be negative for a poor fit.

Drawing resets the model once per completed stroke, and both pages support the
same point brush. Degree/data/configuration changes reset training; raising the
epoch target preserves coefficients. Experiments are in-memory and reset on page
navigation or reload. No dependencies, build step, or separate backend is added.

Files: `regression.html` and `regression.css` provide the page, `regression-app.js`
manages UI and charts, `regression-model.js` implements prediction and gradients,
`regression-data.js` handles samples/import/splitting, and `regression-worker.js`
owns execution and snapshots. `brush.js` and `style.css` are shared by both pages.
Regression tests cover numerical gradients, polynomial convergence, validation,
deterministic splitting, and worker lifecycle/epoch targets.


## K-means clustering page

Open `/kmeans.html` or choose **K-means** in the header. Select 1–7 clusters,
sample Blobs/Scatter/Rings, import unlabeled `x,y` CSV data (0–1), or brush your
own points. Changing k keeps the same dataset so you can compare solutions.

The worker initializes centers with seeded K-means++. **Assign points** compares
every point to every center by squared Euclidean distance. **Move centers** takes
the arithmetic mean of all members of each cluster. **Start clustering** alternates
these phases at the chosen speed, stopping when assignments no longer change.
Each iteration counts a completed mean update. Reset centers repeats the same
initialization; use a different seed and Apply to try another solution.

Diamonds mark numbered centers; rings and dashed paths show their previous
positions and movement. Point-to-center lines show actual current assignments.
Choose a cluster to highlight every member and see the mean calculation. In
Inspect mode, click near a point (or enter coordinates) to see its assigned
cluster and distances to every current center. Numeric inspection selects the
nearest existing point. Background regions show the nearest center; immediately
after movement they can differ from the point colors until assignment runs again.

Inertia is the sum of squared distances for the displayed assignments/centers.
All points participate equally, with no target labels or train/test split.
Colors are arbitrary cluster identifiers. Empty clusters retain their old centers
and are labeled empty. Repeated coordinates can produce fewer occupied clusters
than k. Ties retain a previous tied assignment, otherwise use the first center.
At least k points are needed to initialize; an empty or incomplete drawing remains
editable. There is a 1,000-point limit. K-means may find a local optimum, prefers
compact groups, and will not recover concentric rings as separate clusters.

`kmeans-model.js` contains the DOM-free algorithm, seeded initialization, samples,
and CSV validation. `kmeans-worker.js` owns stepping/scheduling. `kmeans-app.js`
handles controls, cluster/point inspection and charts. `kmeans.html` and
`kmeans.css` reuse the shared layout, palette, and brush. No installation or build
step is added. Tests verify exact assignments/means, monotone inertia, reproducible
convergence for k=1–7, duplicate/empty clusters, and worker lifecycle handling.
