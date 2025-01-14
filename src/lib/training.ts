import * as tf from "@tensorflow/tfjs";
import Chart from 'chart.js/auto';

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
  return prediction;
}

export function visMakePrediction(inputValue: number, model: tf.Sequential){
  const inputTensor = tf.tensor([inputValue]);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  prediction.data().then(predictedValue => {
    return predictedValue;
  });
}

export function startTraining(functionType: string, model: tf.Sequential, maxX: number, minX: number, pointCount: number, epochs: number, batchSize: number, framerate: number) {
  console.log(framerate)
  const [xData, yData] = createData(functionType, maxX, minX, pointCount);
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  const predictions: number[] = [];
  const xValuesForPlotting: number[] = Array.from(xData.dataSync());

  const printMSECallback = {
    onEpochEnd: async (epoch: number, logs: tf.Logs) => {
      console.log(`Epoch ${epoch + 1}: MSE = ${logs.loss}`);

      // Save predictions every 20 epochs
      if ((epoch + 1) % framerate === 0) {
        console.log(`Prediction`);
        const predictionsAtEpoch: number[] = [];
        xData.dataSync().forEach((inputValue, index) => {
          const prediction = makePrediction(inputValue, model);
          prediction.data().then(predictedValue => {
            predictionsAtEpoch.push(predictedValue[0]);
          });
        });
        predictions.push(predictionsAtEpoch);
      }
    },
  };

  model.fit(xData, yData, {
    epochs: epochs,
    batchSize: batchSize,
    shuffle: true,
    callbacks: [printMSECallback],
  }).then(info => {
    console.log('Final accuracy', info.history.mse);

    // After training, plot the results using Chart.js
    const ctx = document.getElementById('myChart') as HTMLCanvasElement;
    console.log(ctx);
    const chartData = {
      labels: xValuesForPlotting,
      datasets: [{
        label: 'Training Data (Sine Function)',
        data: Array.from(yData.dataSync()),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: false,
        tension: 0.1
      }]
    };
      console.log(predictions);

    predictions.forEach((predictionSet, epochIndex) => {
      chartData.datasets.push({
        label: `Epoch ${framerate * (epochIndex + 1)} Predictions`,
        data: predictionSet,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: false,
        tension: 0.1
      });
    });

    const myChart = new Chart(ctx, {
      type: 'line',
      data: chartData,
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Model Training and Predictions'
          },
        },
        scales: {
          x: {
            title: {
              display: false,
              text: 'X Values'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Y Values'
            }
          }
        }
      }
    });
  });
}
