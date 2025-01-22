import * as tf from "@tensorflow/tfjs";
import Chart from 'chart.js/auto';
import type { w } from "vitest/dist/chunks/reporters.D7Jzd9GS.js";
let myChart: Chart | null = null;
export function createData(functionType: string, maxX: number, minX: number, pointCount: number): [tf.Tensor, tf.Tensor] {
  if (functionType === "sine") {
    const xValues: number[] = [];
    const step: number = (maxX - minX) / (pointCount - 1);
    for (let i: number = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues: number[] = xValues.map(x => -Math.sin(x));
    const xTensor: tf.Tensor = tf.tensor(xValues);
    const yTensor: tf.Tensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  } else if (functionType === "exponential") {
    const xValues: number[] = [];
    const step: number = (maxX - minX) / (pointCount - 1);
    for (let i: number = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues: number[] = xValues.map(x => Math.pow(2,-x));
    const xTensor: tf.Tensor = tf.tensor(xValues);
    const yTensor: tf.Tensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  } else if (functionType === "parabola") {
    const xValues: number[] = [];
    const step: number = (maxX - minX) / (pointCount - 1);
    for (let i: number = 0; i < pointCount; i++) {
      xValues.push(minX + i * step);
    }
    const yValues: number[] = xValues.map(x => x ** 2);
    const xTensor: tf.Tensor = tf.tensor(xValues);
    const yTensor: tf.Tensor = tf.tensor(yValues);
    return [xTensor, yTensor];
  }
  alert("Error creating data");
  return [tf.tensor([0]), tf.tensor([0])];
}

export function makePrediction(inputValue: number, model: tf.Sequential): tf.Tensor {
  const inputTensor: tf.Tensor = tf.tensor([inputValue]);
  const prediction: tf.Tensor = model.predict(inputTensor) as tf.Tensor ;
  return prediction;
}

export function visMakePrediction(inputValue: number, model: tf.Sequential): Float32Array | Int32Array | Uint8Array {
  const inputTensor: tf.Tensor = tf.tensor([inputValue]);
  const prediction: tf.Tensor = model.predict(inputTensor) as tf.Tensor;
  return prediction.dataSync();
}

export function startTraining(functionType: string, model: tf.Sequential, maxX: number, minX: number, pointCount: number, epochs: number, batchSize: number, framerate: number): void {
  if (myChart) {
    myChart.destroy();
  }
  const [xData, yData] = createData(functionType, maxX, minX, pointCount);
  const trainingPreview: HTMLParagraphElement = document.getElementById('progress') as HTMLParagraphElement;
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  const predictions: number[] = [];
  const temp: number[] = Array.from(xData.dataSync());
  const xValuesForPlotting: number[] = []; 
  for (let i: number = 0; i < temp.length; i++) {
    const val: number = temp[temp.length - i - 1]; 
    const secondary: string = val.toFixed(2); 
    const final: number = +secondary;
    xValuesForPlotting.push(final);
  }

  const printMSECallback: { onEpochEnd: (epoch: number, logs: tf.Logs) => Promise<void>; } = {
    onEpochEnd: async (epoch: number, logs: tf.Logs) => {
      trainingPreview.innerText = `Epoch: ${epoch}`;
      const colors: string[] = ['red', 'orange', 'yellow', 'green', 'purple', 'pink', 'white'];
      // Save predictions every 20 epochs
      if ((epoch + 1) % framerate === 0) {
        if (myChart) {
          myChart.destroy();
        }
        const predictionsAtEpoch: number[] = [];
        xData.dataSync().forEach((inputValue, index) => {
          const prediction: tf.Tensor = makePrediction(inputValue, model);
          prediction.data().then(predictedValue => {
            predictionsAtEpoch.push(predictedValue[0]);
          });
        });
        predictions.push(predictionsAtEpoch);
      // After training, plot the results using Chart.js
      const ctx = document.getElementById('myChart') as HTMLCanvasElement;
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
        chartData.datasets.push({
          label: `Epoch ${framerate * (epochIndex + 1)} Predictions`,
          data: predictionSet,
          borderColor: colors[epochIndex % colors.length],
          backgroundColor: colors[epochIndex % colors.length],
          fill: false,
          tension: 0.1
        });
      });

      myChart = new Chart(ctx, {
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
              algorithm: 'lttb'
            }
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
                lineWidth: 1 // Set grid line width
              },
              ticks: {
                stepSize: 1 // Control tick spacing (optional)
              }
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
                lineWidth: 1 // Set grid line width
              },
              ticks: {
                stepSize: 1 // Customize tick intervals for the y-axis (optional)
              }
            }
          }
        }
      });
      }
    }
  };

  model.fit(xData, yData, {
    epochs: epochs,
    batchSize: batchSize,
    shuffle: true,
    callbacks: [printMSECallback]
  }).then(info => {
      if (myChart) {
        myChart.destroy();
      }
      trainingPreview.innerText = `Finished Training`;
      // After training, plot the results using Chart.js
      const ctx: HTMLCanvasElement = document.getElementById('myChart') as HTMLCanvasElement;
      const chartData: { labels: number[]; datasets: { label: string; data: number[]; borderColor: string; backgroundColor: string; fill: boolean; tension: number; }[]; } = {
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
        const colors: string[] = ['red', 'orange', 'yellow', 'green', 'purple', 'pink', 'white'];
        chartData.datasets.push({
          label: `Epoch ${framerate * (epochIndex + 1)} Predictions`,
          data: predictionSet,
          borderColor: colors[epochIndex % colors.length],
          backgroundColor: colors[epochIndex % colors.length],
          fill: false,
          tension: 0.1
        });
      });

      myChart = new Chart(ctx, {
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
              algorithm: 'lttb'
            }
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
                lineWidth: 1 // Set grid line width
              },
              ticks: {
                stepSize: 1 // Control tick spacing (optional)
              }
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
                lineWidth: 1 // Set grid line width
              },
              ticks: {
                stepSize: 1 // Customize tick intervals for the y-axis (optional)
              }
            }
          }
        }
      });
});
}
