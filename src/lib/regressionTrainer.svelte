<script lang="ts">
import {
  startTraining,
  makePrediction,
} from '$lib/training';

let epochs = $state(500);
let batchSize = $state(64);
let framerate = $state(15);
let minX = $state(-6.28);
let maxX = $state(6.28);
let pointCount = $state(2000);
let length = $state(5);
let prediction = $state(0);
let snapshotRate = $derived(Math.floor(epochs / (length * framerate)));

import * as tf from "@tensorflow/tfjs"; // Ensure this matches the TensorFlow version used in training.ts

let { functionType, model } = $props() as {functionType: string, model: tf.Sequential}

</script>

<div class="goodDiv">
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
      Frame Rate:
      <input class="unitInput" type="number" bind:value={framerate} />
    </label>
    <br />
    <label>
      Timelapse Length (s):
      <input class="unitInput" type="number" bind:value={length} />
    </label>
    <br />
    <label>
      Prediction:
      <input class="unitInput" type="number" bind:value={prediction} />
    </label>

    <p>Snapshot every {snapshotRate} epochs</p>
  </div>
  <button class="boton-elegante" id="greenElon" on:click={() => startTraining(functionType, model, minX, maxX, pointCount, epochs, batchSize)}>
    Start Training
  </button>
  <button class="boton-elegante" id="yellowElon" on:click={() => makePrediction(prediction, model)}>
    Make Prediction
  </button>
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
