import * as tf from "@tensorflow/tfjs";
import { Chart, Svg, Axis, Bars } from 'layerchart';

export function createData(functionType: string, maxX: number, minX: number, pointCount: number): [tf.Tensor, tf.Tensor] {
  if (functionType === "sine") {
    const xValues = [];
    const step = (maxX - minX) / (pointCount - 1);
    for (let i = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues = xValues.map(x => Math.sin(x));
    const xTensor = tf.tensor(xValues);
    const yTensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  }
  alert("ERROR");
  return [tf.tensor([0]), tf.tensor([0])];
}

export function makePrediction(inputValue: number, model: tf.Sequential) {
  const inputTensor = tf.tensor([inputValue]);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  prediction.data().then(predictedValue => {
    console.log(`Prediction for input ${inputValue}:`, predictedValue);
  });
}

export function plotGraph(xValues: number[], yValues: number[], predictedValues: number[]) {
  const trace1 = {
    x: xValues,
    y: yValues,
    type: 'scatter',
    mode: 'lines',
    name: 'Training Data',
    line: { color: 'blue' },
  };

  const trace2 = {
    x: xValues,
    y: predictedValues,
    type: 'scatter',
    mode: 'lines',
    name: 'Model Predictions',
    line: { color: 'red' },
  };

  const layout = {
    title: 'Model Training and Predictions',
    xaxis: { title: 'X' },
    yaxis: { title: 'Y' },
  };

}

export function startTraining(functionType: string, model: tf.Sequential, maxX: number, minX: number, pointCount: number, epochs: number, batchSize: number) {
  const [xData, yData] = createData(functionType, maxX, minX, pointCount);
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  const printMSECallback = {
    onEpochEnd: async (epoch: number, logs: tf.Logs) => {
      console.log(`Epoch ${epoch + 1}: MSE = ${logs.loss}`);

    },
  };

  model.fit(xData, yData, {
    epochs: epochs,
    batchSize: batchSize,
    shuffle: true,
    callbacks: [printMSECallback],
  }).then(info => {
    console.log('Final accuracy', info.history.mse);
  });
}

