import * as tf from "@tensorflow/tfjs";
import Chart, { LinearScale } from 'chart.js/auto';
function getRandomRedColor(): { borderColor: string; backgroundColor: string } {
  const red = Math.floor(Math.random() * 256); // Random red value (0-255)
  const green = Math.floor(Math.random() * 128); // Random green value (0-127 for a red-dominated shade)
  const blue = Math.floor(Math.random() * 128); // Random blue value (0-127 for a red-dominated shade)

  return {
    borderColor: `rgb(${red}, ${green}, ${blue})`,
    backgroundColor: `rgba(${red}, ${green}, ${blue}, 1)`,
  };
}
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
  } else if (functionType === "exponential") {
    const xValues = [];
    const step = (maxX - minX) / (pointCount - 1);
    for (let i = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues = xValues.map(x => Math.pow(2,x));
    const xTensor = tf.tensor(xValues);
    const yTensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  } else if (functionType === "parabola") {
    const xValues = [];
    const step = (maxX - minX) / (pointCount - 1);
    for (let i = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues = xValues.map(x => x ** 2);
    const xTensor = tf.tensor(xValues);
    const yTensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  }
  alert("Error creating data");
  return [tf.tensor([0]), tf.tensor([0])];
}

export function makePrediction(inputValue: number, model: tf.Sequential) {
  const inputTensor = tf.tensor([inputValue]);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  return prediction;
}

export function visMakePrediction(inputValue: number, model: tf.Sequential): any {
  const inputTensor = tf.tensor([inputValue]);
  const prediction = model.predict(inputTensor) as tf.Tensor;
  return prediction.dataSync();
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
  const temp: number[] = Array.from(xData.dataSync());
  let xValuesForPlotting: number[] = []; 
  let step = Math.round(temp.length / 500);
  for (let i = 0; i < temp.length; i += step) {
    const val: number = temp[temp.length - i - 1]; 
    const secondary: string = val.toFixed(2); 
    const final: number = +secondary;
    xValuesForPlotting.push(final);
  }

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
          label: 'Training Data (Base Function)',
          data: Array.from(yData.dataSync()),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          fill: false,
          tension: 0.1
        }]
      };
      predictions.forEach((predictionSet, epochIndex) => {
        const colors = getRandomRedColor();
        chartData.datasets.push({
          label: `Epoch ${framerate * (epochIndex + 1)} Predictions`,
          data: predictionSet,
          borderColor: colors.borderColor,
          backgroundColor: colors.backgroundColor,
          fill: false,
          tension: 0.1
        });
      });

const myChart = new Chart(ctx, {
  type: 'line',
  data: chartData,
  options: {
    responsive: true,
    aspectRatio: 1,
    plugins: {
      title: {
        display: true,
        text: 'Model Training and Predictions'
      },
      decimation: {
        enabled: true,
        algorithm: 'lttb',
      },
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true, // Display x-axis title
          text: 'X Values'
        },
        grid: {
          drawOnChartArea: true, // Enable grid lines across the chart area
          color: 'rgba(200, 200, 200, 0.2)', // Set grid line color
          lineWidth: 1, // Set grid line width
        },
        ticks: {
          stepSize: 1, // Control tick spacing (optional)
        },
      },
      y: {
        type: 'linear',
        title: {
          display: true, // Display y-axis title
          text: 'Y Values'
        },
        grid: {
          drawOnChartArea: true, // Enable grid lines across the chart area
          color: 'rgba(200, 200, 200, 0.2)', // Set grid line color
          lineWidth: 1, // Set grid line width
        },
        ticks: {
          stepSize: 1, // Customize tick intervals for the y-axis (optional)
        },
      },
    },
  },
});
    });
}
