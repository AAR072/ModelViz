<script lang="ts">
import { writable } from "svelte/store";
let {functionType}= $props();

// Store for layers
let layers = writable([]);

// Add a new layer to the list
function addLayer() {
  layers.update((current) => [
    ...current,
    { type: "dense", units: 128, activation: "relu", inputShape: "" },
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

// Export the model configuration
function exportConfig() {
  $layers; // Reactively access the store
  alert(JSON.stringify($layers, null, 2));
}
</script>

<!-- Main UI -->
<h1>Model Builder</h1>

<!-- Add Layer Button -->
<button on:click={addLayer}>Add Layer</button>

<!-- Layer Configuration List -->
{#each $layers as layer, index}
  <div>
    <h3>Layer {index + 1}</h3>

    <!-- Layer Type -->
    <label>
      Type:
      <select bind:value={layer.type} on:change={(e) => updateLayer(index, "type", e.target.value)}>
        <option value="dense">Dense</option>
        <option value="conv2d">Conv2D</option>
        <option value="flatten">Flatten</option>
        <option value="dropout">Dropout</option>
      </select>
    </label>

    <!-- Units (for Dense Layers) -->
    {#if layer.type === "dense"}
      <label>
        Units:
        <input
          type="number"
          bind:value={layer.units}
          on:input={(e) => updateLayer(index, "units", e.target.value)}
        />
      </label>
    {/if}

    <!-- Activation Function -->
    <label>
      Activation:
      <select
        bind:value={layer.activation}
        on:change={(e) => updateLayer(index, "activation", e.target.value)}
      >
        <option value="relu">ReLU</option>
        <option value="sigmoid">Sigmoid</option>
        <option value="softmax">Softmax</option>
        <option value="tanh">Tanh</option>
      </select>
    </label>

    <!-- Input Shape (only for the first layer) -->
    {#if index === 0}
      <label>
        Input Shape:
        <input
          type="text"
          bind:value={layer.inputShape}
          placeholder="[784]"
          on:input={(e) => updateLayer(index, "inputShape", e.target.value)}
        />
      </label>
    {/if}

    <!-- Remove Layer Button -->
    <button on:click={() => removeLayer(index)}>Remove Layer</button>
    <hr />
  </div>
{/each}

<!-- Export Button -->
<button on:click={exportConfig}>Export Model Config</button>

<!-- Debugging -->
<pre>{JSON.stringify($layers, null, 2)}</pre>

