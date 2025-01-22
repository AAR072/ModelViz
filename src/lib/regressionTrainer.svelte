<script lang="ts">
  import { DataGenerator, ModelTrainer, Visualization } from "$lib/training";
  import * as tf from "@tensorflow/tfjs"; // Ensure this matches the TensorFlow version used in training.ts
  import { onMount } from 'svelte';

  let epochs: number = $state(100);
  let batchSize: number = $state(64);
  let framerate: number = $state(3);
  let minX: number = $state(-6.28);
  let maxX: number = $state(6.28);
  let pointCount: number = $state(500);
  const answer: unknown = $state(0);
  const snapshotRate: number = $derived(Math.floor(epochs / framerate));

  const { functionType, model } = $props() as { functionType: string; model: tf.Sequential };

  let trainer: ModelTrainer = $derived(new ModelTrainer(model, snapshotRate));
  let visualizer: Visualization;

  onMount(() => {
    visualizer = new Visualization();
  });

  async function startTraining(
    functionType: string,
    model: tf.Sequential,
    minX: number,
    maxX: number,
    pointCount: number,
    epochs: number,
    batchSize: number,
    snapshotRate: number
  ) {
    const progressElement = document.getElementById("progress") as HTMLElement;

    // Compile the model
    trainer.compileModel();

    try {
      // Train the model
      await trainer.train(
        functionType,
        maxX,
        minX,
        pointCount,
        epochs,
        batchSize,
        (epoch, logs) => {
          progressElement.textContent = `Epoch ${epoch + 1}/${epochs}: Loss = ${logs?.loss?.toFixed(4)}`;
        }
      );

      progressElement.textContent = "Training completed!";
      
      // Generate data for visualization
      const [xData, yData] = DataGenerator.createData(functionType, maxX, minX, pointCount);

      // Visualize the results
      visualizer.plot(
        Array.from(xData.dataSync()),
        Array.from(yData.dataSync()),
        trainer.predictions,
        snapshotRate
      );
    } catch (error) {
      progressElement.textContent = "An error occurred during training!";
      console.error(error);
    }
  }
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
    <p id="progress"></p>
    <p>Snapshot every {snapshotRate} epochs</p>
    <p>Model Prediction: {answer}</p>
  </div>
  <button
    class="boton-elegante"
    id="greenElon"
    on:click={() =>
      startTraining(functionType, model, minX, maxX, pointCount, epochs + 1, batchSize, snapshotRate)
    }
  >
    Start Training
  </button>
  <canvas id="myChart" width="400" height="200"></canvas>
</div>

<style>
  @import "$lib/styles/main.css";
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
    background: radial-gradient(circle, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0) 70%);
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
