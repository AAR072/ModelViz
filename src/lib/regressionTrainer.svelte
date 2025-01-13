<script lang="ts">
import * as tf from "@tensorflow/tfjs"; // TensorFlow.js

let { functionType, model } = $props() as {functionType: string, model: tf.Sequential}
let epochs: number = $state(500);
let batchSize: number = $state(64);
let framerate: number = $state(15);
let minX: number = $state(-6.28);
let maxX: number = $state(6.28);
let pointCount: number = $state(2000);
let length: number = $state(5);
let prediction: number = $state(0);
let snapshotRate: number = $derived(Math.floor(epochs / (length * framerate)));
let plotData: any; // To store plot data

function createData(functionType: string): [tf.Tensor, tf.Tensor] {
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

function makePrediction(inputValue) {
  const inputTensor = tf.tensor([inputValue]);
  const prediction = model.predict(inputTensor);
  prediction.data().then(predictedValue => {
    console.log(`Prediction for input ${inputValue}:`, predictedValue);
  });
}

function plotGraph(xValues: number[], yValues: number[], predictedValues: number[]) {
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

  if (!plotData) {
    // Initialize the plot if it doesn't exist
    Plotly.newPlot('predictionChart', [trace1, trace2], layout);
  } else {
    // Update the plot if it already exists
    Plotly.react('predictionChart', [trace1, trace2], layout);
  }
}

function startTraining() {
  const [xData, yData] = createData(functionType);
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  const printMSECallback = {
    onEpochEnd: async (epoch: number, logs: tf.Logs) => {
      console.log(`Epoch ${epoch + 1}: MSE = ${logs.loss}`);

      if (epoch % snapshotRate === 0) {
        // Generate predictions and plot snapshot
        const predictedValues = xData.arraySync().map((x: number) => model.predict(tf.tensor([x])).dataSync()[0]);
        plotGraph(xData.arraySync(), yData.arraySync(), predictedValues);
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
  });
}
</script>

<div class="goodDiv">
  <p id="pageTitle">Training Parameters</p>
  <div class="trainingParams">
    <label>
      Epochs:
      <input
        class="unitInput"
        type="number"
        defaultvalue={500}
        bind:value={epochs}
      />
    </label>
    <br>
    <label>
      Batch Size:
      <input
        class="unitInput"
        type="number"
        defaultvalue={64}
        bind:value={batchSize}
      />
    </label>
    <br>
    <label>
      Minimum X Value:
      <input
        class="unitInput"
        type="number"
        defaultvalue={-6.28}
        bind:value={minX}
      />
    </label>
    <br>
    <label>
      Maximum X Value:
      <input
        class="unitInput"
        type="number"
        defaultvalue={6.28}
        bind:value={maxX}
      />
    </label>
    <br>
    <label>
      Point Count:
      <input
        class="unitInput"
        type="number"
        defaultvalue={2000}
        bind:value={pointCount}
      />
    </label>

    <br>
    <label>
      Frame Rate:
      <input
        class="unitInput"
        type="number"
        defaultvalue={15}
        bind:value={framerate}
      />
    </label>
    <br>
    <label>
      Timelapse Length (s):
      <input
        class="unitInput"
        type="number"
        defaultvalue={5}
        bind:value={length}
      />
    </label>
    <br>
    <label>
      Prediction:
      <input
        class="unitInput"
        type="number"
        defaultvalue={0.5}
        bind:value={prediction}
      />
    </label>

    <p>Snapshot every {snapshotRate} epochs</p>
  </div>
  <button class="boton-elegante" id="greenElon" onclick={startTraining}>Start Training</button>
  <button class="boton-elegante" id="yellowElon" onclick={()=>makePrediction(prediction)}>Make Prediction</button>
  <div id="predictionChart" style="width:100%; height:400px;"></div>
</div>
<style>
@import '$lib/styles/main.css';
.boton-elegante {
margin-top: 1vh;
margin-bottom: 2vh;
padding: 1vw 3vw;
border: 2px solid #2c2c2c;
background-color: #1a1a1a;
font-size: 1vw;
color: #ffffff;
cursor: pointer;
border-radius: 30px;
transition: all 0.4s ease;
outline: none;
position: relative;
overflow: hidden;
font-weight: bold;
}

.boton-elegante::after {
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background: radial-gradient(
circle,
rgba(255, 255, 255, 0.25) 0%,
rgba(255, 255, 255, 0) 70%
);
transform: scale(0);
transition: transform 0.5s ease;
}

.boton-elegante:hover::after {
transform: scale(4);
}

.boton-elegante:hover {
border-color: #666666;
background: #34a1eb;
}
#greenElon:hover {
background: #25a834;
}
.unitInput {
background-color: black;
color: white;
}
.trainingParams {
font-size: 2vw;
}
@media (max-width: 768px) { 
.trainingParams {
font-size: 5vw;
}
#pageTitle {
font-size: 7vw;
}

.boton-elegante {
padding: 3vw 6vw;
font-size: 3vw;
}

}
#pageTitle {
margin-bottom: 10vh;
}
</style>
