<script lang="ts">
  import { onMount } from 'svelte';
  import { KNNApp } from '$lib/svm';  

  let pointCount = $state(100);  
  let chartReady = $state(false); 

  let knnApp: KNNApp;

  // Extract functionType from the props
  const { dataset } = $props() as { dataset: string };

  // Initialize the KNN app and render on mount
  onMount(() => {
    knnApp = new KNNApp(document.getElementById('chart'));
    knnApp.run(dataset, pointCount);
    chartReady = true;
  });

  // Function to handle the form submission and trigger KNN
  function updateChart() {
    knnApp.run(dataset, pointCount);  // Use the dataset passed from the parent component
  }
</script>

<div class="essentialDiv">
  <h1>KNN Visualizer</h1>

  <!-- Controls to change the point count -->
  <div>
    <label for="pointCount">Select Number of Points:</label>
    <input
      id="pointCount"
      type="number"
      bind:value={pointCount}
      min="1"
      max="1000"
      on:change={updateChart}
    />
  </div>

  <!-- The Canvas element for the chart -->
  <canvas id="chart" width="400" height="400"></canvas>
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
