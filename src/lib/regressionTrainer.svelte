<script lang="ts">
import {
  startTraining,
  visMakePrediction,
} from '$lib/training';

let epochs = $state(100);
let batchSize = $state(64);
let framerate = $state(3);
let minX = $state(-6.28)
let maxX = $state(6.28);
let pointCount = $state(500);
let prediction = $state(0);
let answer: any = $state(0);
let snapshotRate = $derived(Math.floor(epochs / framerate));

import * as tf from "@tensorflow/tfjs"; // Ensure this matches the TensorFlow version used in training.ts

let { functionType, model } = $props() as {functionType: string, model: tf.Sequential}

</script>

<div class="essentialDiv">
  <p id="pageTitle">Training Parameters</p>
  <div class="trainingParams">
    <label>
      Epochs:
      <input class="unitInput" type="number" bind:value={epochs} />
    </label>
    <br />
    <label>
      Batch Size:
      <input class="unitInput" type="number" bind:value={batchSize} />
    </label>
    <br />
    <label>
      Minimum X Value:
      <input class="unitInput" type="number" bind:value={minX} />
    </label>
    <br />
    <label>
      Maximum X Value:
      <input class="unitInput" type="number" bind:value={maxX} />
    </label>
    <br />
    <label>
      Point Count:
      <input class="unitInput" type="number" bind:value={pointCount} />
    </label>
    <br />
    <label>
      Snapshot Count:
      <input class="unitInput" type="number" bind:value={framerate} />
    </label>
    <br />
    <label>
      Make a prediction:
      <input class="unitInput" type="number" bind:value={prediction} />
    </label>
    <button class="boton-elegante" id="yellowElon" on:click={() => {answer = visMakePrediction(prediction, model)}}>
      Make Prediction
    </button>
    <p id="progress"></p>
    <p>Snapshot every {snapshotRate} epochs</p>
    <p>Model Prediction: {answer}</p>
  </div>
  <button class="boton-elegante" id="greenElon" on:click={() => startTraining(functionType, model, minX, maxX, pointCount, epochs + 1, batchSize, snapshotRate)}>
    Start Training
  </button>
  <canvas id="myChart" width="400" height="200"></canvas>
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
#pageTitle {
font-size: 3vw;
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
