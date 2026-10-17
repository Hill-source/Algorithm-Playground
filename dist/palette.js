// Class IDs stay stable across sample data, CSV imports, plots and predictions.
export const palette = [
  { name: 'Mint', rgb: [131, 232, 206] },
  { name: 'Violet', rgb: [172, 155, 250] },
  { name: 'Orange', rgb: [255, 169, 83] },
  { name: 'Blue', rgb: [91, 177, 255] },
  { name: 'Pink', rgb: [245, 136, 205] },
  { name: 'Yellow', rgb: [239, 220, 103] },
  { name: 'Red', rgb: [249, 108, 111] }
];

export const classColor = (label, alpha = 1) =>
  `rgba(${palette[label].rgb.join(',')},${alpha})`;
