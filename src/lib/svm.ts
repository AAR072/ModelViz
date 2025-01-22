import * as tf from "@tensorflow/tfjs";
import * as tfModels from "@tensorflow-models/knn-classifier";
import Chart, { LinearScale, type ChartData, type Point } from 'chart.js/auto';
let myChart: Chart | null = null;
/**
 * Generates data points and adds them to the provided KNN classifier.
 *
 * @param functionType Type of function to generate data points ('shotgun', 'td', 'lr').
 * @param pointCount Number of data points to generate.
 * @param classifier The KNN classifier to add data points to.
 */
export function createData(functionType: string, pointCount: number, classifier: tfModels.KNNClassifier) {

  const data = []; // Array to hold points and labels

  for (let i = 0; i < pointCount; i++) {
    // Generate random x and y values
    const x = +(Math.random()).toFixed(2);  // Random x value between 0 and 1
    const y = +(Math.random()).toFixed(2);  // Random y value between 0 and 1

    let label: number;

    if (functionType === "shotgun") {
      // Classify the point based on its position relative to y = x
      label = y > x ? 1 : 0;  // Points above the line y = x are labeled 1, below are 0
    } else if (functionType === "td") {
      // Classify the point based on its position relative to y = 0.5
      label = y > 0.5 ? 1 : 0;  // Points above y = 0.5 are labeled 1, below are 0
    } else if (functionType === "lr") {
      // Classify the point based on its position relative to x = 0.5
      label = x > 0.5 ? 1 : 0;  // Points where x > 0.5 are labeled 1, others are 0
    } else {
      return [];
    }

    // Create a feature tensor
    const feature = tf.tensor([x, y]); // Feature vector [x, y]

    // Add the feature and label to the classifier
    classifier.addExample(feature, label);

    // Optionally keep track of the data (useful for testing/debugging)
    data.push({ x: [x, y], y: label });
  }
  return data;
}
export function viewKNN(functionType: string, pointCount: number) {
  const model = tfModels.create();
  const predictionSet = createData(functionType, pointCount, model);
  const chartData = { 
    labels: ["Red Points","Blue Points"] as string[], 
    datasets: [] as { [key: string]: any }[]  // array of objects with string keys and any values
  };
  // 1. Create the dataset that we need to input
  const redArray: number[][] = [];
  const blueArray: number[][] = [];
  const whiteArray: number[][] = [];

  for (let i = 0; i < predictionSet.length; i++) {
    if (predictionSet[i].y === 0) {
      redArray.push(predictionSet[i].x);
    } else {
      blueArray.push(predictionSet[i].x);
    }
  }

  // Generate 100 white points based on the functionType logic
  for (let i = 1; i < 100; i++) {
    let x = i / 100;
    let y = 0;

    if (functionType === "shotgun") {
      y = x;
    }
    if (functionType === "td") {
      y = 0.5;
    }
    if (functionType === "lr") {
      y = x;
      x = 0.5;
    }

    // Ensure points remain within bounds [0, 1]
    x = Math.min(Math.max(x, 0), 1);
    y = Math.min(Math.max(y, 0), 1);

    whiteArray.push([x, y, 2]);
  }

  async function processPredictions() {
    // Store promises for redArray predictions
    const redPromises = redArray.map((item, i) => {
      const feature = tf.tensor(item); // Feature vector [x, y]
      return model.predictClass(feature).then((prediction) => {
        redArray[i].push(+prediction.label);
      });
    });

    // Store promises for blueArray predictions
    const bluePromises = blueArray.map((item, i) => {
      const feature = tf.tensor(item); // Feature vector [x, y]
      return model.predictClass(feature).then((prediction) => {
        blueArray[i].push(+prediction.label);
      });
    });

    // Wait for all predictions to complete
    await Promise.all([...redPromises, ...bluePromises]);

    // Now you can continue with the rest of the code
    const dataset = model.getClassifierDataset();

    // 2. Make predictions and store.
    // 3. Add to data and label
    // Colors for each class
    const colors = ['red', 'blue', 'white'];

    // Create a dataset for this class
    for (let i = 0; i < redArray.length; i++) {
      chartData.datasets.push({
        label: `${redArray[i][2]}` as string,
        data: [{x: redArray[i][0], y: redArray[i][1]}],
        backgroundColor: colors[+redArray[i][2]] // Cycle through colors
      });
    }
    for (let i = 0; i < blueArray.length; i++) {
      chartData.datasets.push({
        label: `${blueArray[i][2]}` as string,
        data: [{x: blueArray[i][0], y: blueArray[i][1]}],
        backgroundColor: colors[+blueArray[i][2]] // Cycle through colors
      });
    }
    for (let i = 0; i < whiteArray.length; i++) {
      chartData.datasets.push({
        label: `${whiteArray[i][2]}` as string,
        data: [{x: whiteArray[i][0], y: whiteArray[i][1]}],
        backgroundColor: colors[+whiteArray[i][2]] // Cycle through colors
      });
    }
    console.log(redArray,blueArray,whiteArray);
    const ctx = document.getElementById('chart') as HTMLCanvasElement;
    if (!ctx) {
      console.error("Canvas element with id 'chart' not found");
      return;
    }
    if (myChart) {
      myChart.destroy();
    }

    myChart = new Chart(ctx, {
      type: 'scatter',
      data: chartData,
      options: {
        plugins: {
          legend: {
            display: false
          }
        },
        aspectRatio: 1,
        events: [],
        interaction: false,
        responsive: true,
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'X Coordinate' },
            grid: {
              drawOnChartArea: true, // Enable grid lines across the chart area
              color: 'rgba(200, 200, 200, 0.2)', // Set grid line color
              lineWidth: 1 // Set grid line width
            }

          },
          y: {
            grid: {
              drawOnChartArea: true, // Enable grid lines across the chart area
              color: 'rgba(200, 200, 200, 0.2)', // Set grid line color
              lineWidth: 1 // Set grid line width
            },
            title: { display: true, text: 'Y Coordinate' }
          }
        }
      }
    });
    // You can proceed to the rest of the code here, knowing the predictions have been processed.
  }
  processPredictions();
  // Call the function to process predictions
}
