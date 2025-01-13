<script lang="ts">
import { writable } from "svelte/store";
import RegressionTrainer from "$lib/regressionTrainer.svelte"
import * as tf from "@tensorflow/tfjs"; // TensorFlow.js

let { functionType } = $props();
let readyToTrain: boolean = $state(false);
let secondReady: boolean = $state(false);

// Store for layers
let layers = writable([]);



// TensorFlow.js model instance
let model: tf.Sequential | null = null;
layers.update((current) => [
  ...current,
  { type: "dense", units: 32, activation: "relu", inputShape: "[1]" },
]);

// Add a new layer to the list
function addLayer() {
  layers.update((current) => [
    ...current,
    {  units: 32, activation: "relu", inputShape: "" },
  ]);
}

function useDefaults() {
  layers.update((current) => [
    { units: 64, activation: "tanh", inputShape: "[1]" },
    { units: 64, activation: "tanh", inputShape: "" },
  ]);
}

// Remove a layer
function removeLayer(index) {
  layers.update((current) => current.filter((_, i) => i !== index));
}

// Update layer properties
function updateLayer(index, field, value) {
  layers.update((current) => {
    current[index][field] = value;
    return current;
  });
}

// Build the TensorFlow.js model based on user input
function buildModel() {
  const userLayers = $layers; // Access the current layers
  model = tf.sequential(); // Create a new sequential model

  userLayers.forEach((layer, index) => {
    const layerConfig: any = { units: parseInt(layer.units), activation: layer.activation };

    // Only add inputShape for the first layer
    if (index === 0 && layer.inputShape) {
      layerConfig.inputShape = JSON.parse(layer.inputShape); // Parse inputShape from string
    }

    // Add the layer to the model
      model.add(tf.layers.dense(layerConfig));
  });

  // Ensure the last layer outputs a single value with no activation (for regression)
  model.add(tf.layers.dense({ units: 1, activation: null }));

}

// Export the TensorFlow.js model
function exportModel() {
  buildModel();
  readyToTrain = true;
  setTimeout(() => {
    secondReady = true; 
  }, 1500);


}
</script>

<!-- Main UI -->
{#if !secondReady}
  <!-- content here -->
  <div class="goodDiv" class:fade-in={!readyToTrain} class:hidden={readyToTrain}>
    <h1 id="pageTitle" style="padding-bottom: 4vh;">Model Builder</h1>

    <!-- Add Layer Button -->
    <button on:click={addLayer} class="boton-elegante">Add Layer</button>
    <button on:click={buildModel} class="boton-elegante" id="yellowElon" on:click={useDefaults}>Use Default Setup</button>

    <!-- Layer Configuration List -->
    {#each $layers as layer, index}
      <div>
        <h3>Layer {index + 1}</h3>

        <!-- Layer Type -->

        <!-- Units (for Dense Layers) -->
          <label>
            Units:
            <input
              class="unitInput"
              type="number"
              bind:value={layer.units}
              on:input={(e) => updateLayer(index, "units", e.target.value)}
            />
          </label>

        <!-- Activation Function -->
        <label >
          Activation:
          <select class="dropdownMenu"
            bind:value={layer.activation}
            on:change={(e) => updateLayer(index, "activation", e.target.value)}
          >
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="softmax">Softmax</option>
            <option value="tanh">Tanh</option>
          </select>
        </label>


        <!-- Remove Layer Button -->
        <button on:click={() => removeLayer(index)} class="boton-elegante" id="redElon" style="margin-left: 1vw;">Remove Layer</button>
        <hr />
      </div>
    {/each}

    <!-- Export Model Button -->
    <button on:click={exportModel} class="boton-elegante" id="greenElon">Start Training</button>

    <!-- Debugging -->
    <pre>{JSON.stringify($layers, null, 2)}</pre>
  </div>
{/if}
{#if secondReady}
  <div class:fade-in={secondReady} class:hidden={!secondReady}>
    <RegressionTrainer functionType={functionType} model={model}/>
  </div>
{/if}



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
#redElon:hover {
background: #a32123;
}
#yellowElon:hover {
background: #d9d979;
}


@media (max-width: 768px) {

.boton-elegante {
padding: 3vw 6vw;
font-size: 3vw;
}
}

.fade-in {
opacity: 1;
transition: opacity 1s ease-in-out,transform 0.3s ease, box-shadow 0.3s ease;

}

.hidden {
opacity: 0;
transition: opacity 1s ease-in-out;
}
.dropdownMenu {
background-color: black;
color: white;
border-radius: 99px;
}
.unitInput {
background-color: black;
color: white;
}
</style>
