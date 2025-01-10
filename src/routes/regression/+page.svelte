<script lang="ts">
import { onMount } from 'svelte';
import { goto } from '$app/navigation';
let innerWidth = $state(0);
let onPC = $derived(innerWidth > 768);
let titleVisible = $state(false);
let subtitleVisible = $state(false);
let cards: boolean [] = $state([]);
cards.push(false);
cards.push(false);
cards.push(false);
cards.push(false);
cards.push(false);
cards.push(false);
onMount(() => {
  setTimeout(() => {
    titleVisible = true;
  }, 500);
  setTimeout(() => {
    subtitleVisible = true;
  }, 1500 );
  for (let i = 0; i < cards.length; i++) {
    setTimeout(() => {
    cards[i] = true;
    }, 1500 + (i + 1) * 500);
  }
});
</script>
<svelte:window bind:innerWidth />
<div class="goodDiv">
  <p id="pageTitle" class:fade-in={titleVisible} class:hidden={!titleVisible}>ModelViz</p>
  <p id="pageSubtitle" class:fade-in={subtitleVisible} class:hidden={!subtitleVisible}>Choose a problem to visualize</p>
</div>
<div class="allCardContainer">
  <div class="card-container">
    <div class="card" class:fade-in={cards[0]} class:hidden={!cards[0]}>
      <svg viewBox="0 0 200 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        <g stroke="lightgrey" stroke-width="0.5">
          <!-- Vertical gridlines -->
          <line x1="25" y1="0" x2="25" y2="250" />
          <line x1="50" y1="0" x2="50" y2="250" />
          <line x1="75" y1="0" x2="75" y2="250" />
          <line x1="100" y1="0" x2="100" y2="250" />
          <line x1="125" y1="0" x2="125" y2="250" />
          <line x1="150" y1="0" x2="150" y2="250" />
          <line x1="175" y1="0" x2="175" y2="250" />
          <line x1="200" y1="0" x2="200" y2="250" />
          <line x1="225" y1="0" x2="225" y2="250" />

          <!-- Horizontal gridlines -->
          <line x1="0" y1="1" x2="250" y2="1" />
          <line x1="0" y1="25" x2="250" y2="25" />
          <line x1="0" y1="50" x2="250" y2="50" />
          <line x1="0" y1="75" x2="250" y2="75" />
          <line x1="0" y1="100" x2="250" y2="100" />
          <line x1="0" y1="125" x2="250" y2="125" />
          <line x1="0" y1="150" x2="250" y2="150" />
          <line x1="0" y1="175" x2="250" y2="175" />
          <line x1="0" y1="200" x2="250" y2="200" />
          <line x1="0" y1="225" x2="250" y2="225" />
          <line x1="0" y1="250" x2="250" y2="250" />
        </g>
        <!-- Axes -->
        <line x1="0" y1="125" x2="250" y2="125" stroke="white" stroke-width="2" /> <!-- X-axis -->
        <line x1="0" y1="0" x2="0" y2="250" stroke="white" stroke-width="2" /> <!-- Y-axis -->

        <!-- Sine Graph Points -->
        <circle cx="0" cy="125" r="4" fill="#3477eb" />
        <circle cx="50" cy="10" r="4" fill="#3477eb" />
        <circle cx="100" cy="125" r="4" fill="#3477eb" />
        <circle cx="150" cy="240" r="4" fill="#3477eb" />
        <circle cx="190" cy="125" r="4" fill="#3477eb" />

        <!-- Line Fit -->
        <path d="M0,125 Q25,50 50,10 T100,125 T150,240 T190,125" 
          stroke="#eb4034" fill="none" stroke-width="2" />
      </svg>
      <p class="card-text">Sine Function</p>
    </div>
  </div>
  <div class="card-container">
    <div class="card" class:fade-in={cards[1]} class:hidden={!cards[1]}> 
      <svg viewBox="0 0 200 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        <g stroke="lightgrey" stroke-width="0.5">
          <!-- Vertical gridlines -->
          <line x1="25" y1="0" x2="25" y2="250" />
          <line x1="50" y1="0" x2="50" y2="250" />
          <line x1="75" y1="0" x2="75" y2="250" />
          <line x1="100" y1="0" x2="100" y2="250" />
          <line x1="125" y1="0" x2="125" y2="250" />
          <line x1="150" y1="0" x2="150" y2="250" />
          <line x1="175" y1="0" x2="175" y2="250" />
          <line x1="200" y1="0" x2="200" y2="250" />
          <line x1="225" y1="0" x2="225" y2="250" />

          <!-- Horizontal gridlines -->
          <line x1="0" y1="1" x2="250" y2="1" />
          <line x1="0" y1="25" x2="250" y2="25" />
          <line x1="0" y1="50" x2="250" y2="50" />
          <line x1="0" y1="75" x2="250" y2="75" />
          <line x1="0" y1="100" x2="250" y2="100" />
          <line x1="0" y1="125" x2="250" y2="125" />
          <line x1="0" y1="150" x2="250" y2="150" />
          <line x1="0" y1="175" x2="250" y2="175" />
          <line x1="0" y1="200" x2="250" y2="200" />
          <line x1="0" y1="225" x2="250" y2="225" />
          <line x1="0" y1="250" x2="250" y2="250" />
        </g>
        <!-- Axes -->
        <line x1="0" y1="250" x2="250" y2="250" stroke="white" stroke-width="2" /> <!-- X-axis -->
        <line x1="0" y1="0" x2="0" y2="250" stroke="white" stroke-width="2" /> <!-- Y-axis -->

        <!-- Sine Graph Points -->
        <circle cx="0" cy="10" r="4" fill="#3477eb" />
        <circle cx="37.5" cy="150" r="4" fill="#3477eb" />
        <circle cx="100" cy="250" r="4" fill="#3477eb" />
        <circle cx="162.5" cy="150" r="4" fill="#3477eb" />
        <circle cx="200" cy="10" r="4" fill="#3477eb" />

  <!-- Parabola Fit -->
  <path 
    d="M0,0 Q100,500, 200,0" 
    stroke="#eb4034" 
    fill="none" 
    stroke-width="2" />
      </svg>
      <p class="card-text">Parabola</p>
    </div>
  </div>

  <div class="card-container">
    <div class="card" class:fade-in={cards[2]} class:hidden={!cards[2]}>
      <svg viewBox="0 0 200 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        <g stroke="lightgrey" stroke-width="0.5">
          <!-- Vertical gridlines -->
          <line x1="25" y1="0" x2="25" y2="250" />
          <line x1="50" y1="0" x2="50" y2="250" />
          <line x1="75" y1="0" x2="75" y2="250" />
          <line x1="100" y1="0" x2="100" y2="250" />
          <line x1="125" y1="0" x2="125" y2="250" />
          <line x1="150" y1="0" x2="150" y2="250" />
          <line x1="175" y1="0" x2="175" y2="250" />
          <line x1="200" y1="0" x2="200" y2="250" />
          <line x1="225" y1="0" x2="225" y2="250" />

          <!-- Horizontal gridlines -->
          <line x1="0" y1="1" x2="250" y2="1" />
          <line x1="0" y1="25" x2="250" y2="25" />
          <line x1="0" y1="50" x2="250" y2="50" />
          <line x1="0" y1="75" x2="250" y2="75" />
          <line x1="0" y1="100" x2="250" y2="100" />
          <line x1="0" y1="125" x2="250" y2="125" />
          <line x1="0" y1="150" x2="250" y2="150" />
          <line x1="0" y1="175" x2="250" y2="175" />
          <line x1="0" y1="200" x2="250" y2="200" />
          <line x1="0" y1="225" x2="250" y2="225" />
          <line x1="0" y1="250" x2="250" y2="250" />
        </g>
        <!-- Axes -->
        <line x1="0" y1="250" x2="250" y2="250" stroke="white" stroke-width="2" /> <!-- X-axis -->
        <line x1="100" y1="0" x2="100" y2="250" stroke="white" stroke-width="2" /> <!-- Y-axis -->

<!-- Exponential Graph Points -->
<circle cx="0" cy="250" r="4" fill="#3477eb" />
<circle cx="50" cy="205" r="4" fill="#3477eb" />
<circle cx="100" cy="150" r="4" fill="#3477eb" />
<circle cx="150" cy="82" r="4" fill="#3477eb" />
<circle cx="200" cy="0" r="4" fill="#3477eb" />
<path d="M200 0 Q 150 90, 110 140 Q 70 190, 0 250"
          stroke="#eb4034" stroke-width="2" fill="none" />
      </svg>
      <p class="card-text">Exponential Function</p>
    </div>
  </div>
  <div class="card-container">
    <div class="card" class:fade-in={cards[3]} class:hidden={!cards[3]}> 
      <svg viewBox="0 0 200 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        <g stroke="lightgrey" stroke-width="0.5">
          <!-- Vertical gridlines -->
          <line x1="25" y1="0" x2="25" y2="250" />
          <line x1="50" y1="0" x2="50" y2="250" />
          <line x1="75" y1="0" x2="75" y2="250" />
          <line x1="100" y1="0" x2="100" y2="250" />
          <line x1="125" y1="0" x2="125" y2="250" />
          <line x1="150" y1="0" x2="150" y2="250" />
          <line x1="175" y1="0" x2="175" y2="250" />
          <line x1="200" y1="0" x2="200" y2="250" />
          <line x1="225" y1="0" x2="225" y2="250" />

          <!-- Horizontal gridlines -->
          <line x1="0" y1="1" x2="250" y2="1" />
          <line x1="0" y1="25" x2="250" y2="25" />
          <line x1="0" y1="50" x2="250" y2="50" />
          <line x1="0" y1="75" x2="250" y2="75" />
          <line x1="0" y1="100" x2="250" y2="100" />
          <line x1="0" y1="125" x2="250" y2="125" />
          <line x1="0" y1="150" x2="250" y2="150" />
          <line x1="0" y1="175" x2="250" y2="175" />
          <line x1="0" y1="200" x2="250" y2="200" />
          <line x1="0" y1="225" x2="250" y2="225" />
          <line x1="0" y1="250" x2="250" y2="250" />
        </g>
        <!-- Axes -->
        <line x1="0" y1="250" x2="250" y2="250" stroke="white" stroke-width="2" /> <!-- X-axis -->
        <line x1="0" y1="0" x2="0" y2="250" stroke="white" stroke-width="2" /> <!-- Y-axis -->

        <!-- Blue Points -->
        <circle cx="10" cy="30" r="4" fill="#3477eb" />
        <circle cx="30" cy="50" r="4" fill="#3477eb" />
        <circle cx="50" cy="20" r="4" fill="#3477eb" />
        <circle cx="70" cy="40" r="4" fill="#3477eb" />
        <circle cx="90" cy="60" r="4" fill="#3477eb" />
        <circle cx="110" cy="30" r="4" fill="#3477eb" />
        <circle cx="130" cy="50" r="4" fill="#3477eb" />
        <circle cx="150" cy="20" r="4" fill="#3477eb" />


        <circle cx="50" cy="140" r="4" fill="#eb4034" />
        <circle cx="70" cy="120" r="4" fill="#eb4034" />
        <circle cx="90" cy="160" r="4" fill="#eb4034" />
        <circle cx="110" cy="130" r="4" fill="#eb4034" />
        <circle cx="130" cy="170" r="4" fill="#eb4034" />
      </svg>
      <p class="card-text">3</p>
    </div>
  </div>
  <div class="card-container">
    <div class="card" class:fade-in={cards[4]} class:hidden={!cards[4]}> 
      <svg viewBox="0 0 200 250" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
        <g stroke="lightgrey" stroke-width="0.5">
          <!-- Vertical gridlines -->
          <line x1="25" y1="0" x2="25" y2="250" />
          <line x1="50" y1="0" x2="50" y2="250" />
          <line x1="75" y1="0" x2="75" y2="250" />
          <line x1="100" y1="0" x2="100" y2="250" />
          <line x1="125" y1="0" x2="125" y2="250" />
          <line x1="150" y1="0" x2="150" y2="250" />
          <line x1="175" y1="0" x2="175" y2="250" />
          <line x1="200" y1="0" x2="200" y2="250" />
          <line x1="225" y1="0" x2="225" y2="250" />

          <!-- Horizontal gridlines -->
          <line x1="0" y1="1" x2="250" y2="1" />
          <line x1="0" y1="25" x2="250" y2="25" />
          <line x1="0" y1="50" x2="250" y2="50" />
          <line x1="0" y1="75" x2="250" y2="75" />
          <line x1="0" y1="100" x2="250" y2="100" />
          <line x1="0" y1="125" x2="250" y2="125" />
          <line x1="0" y1="150" x2="250" y2="150" />
          <line x1="0" y1="175" x2="250" y2="175" />
          <line x1="0" y1="200" x2="250" y2="200" />
          <line x1="0" y1="225" x2="250" y2="225" />
          <line x1="0" y1="250" x2="250" y2="250" />
        </g>
        <!-- Axes -->
        <line x1="0" y1="250" x2="250" y2="250" stroke="white" stroke-width="2" /> <!-- X-axis -->
        <line x1="0" y1="0" x2="0" y2="250" stroke="white" stroke-width="2" /> <!-- Y-axis -->

        <!-- Blue Points -->
        <circle cx="10" cy="30" r="4" fill="#3477eb" />
        <circle cx="30" cy="50" r="4" fill="#3477eb" />
        <circle cx="50" cy="20" r="4" fill="#3477eb" />
        <circle cx="70" cy="40" r="4" fill="#3477eb" />
        <circle cx="90" cy="60" r="4" fill="#3477eb" />
        <circle cx="110" cy="30" r="4" fill="#3477eb" />
        <circle cx="130" cy="50" r="4" fill="#3477eb" />
        <circle cx="150" cy="20" r="4" fill="#3477eb" />


        <circle cx="50" cy="140" r="4" fill="#eb4034" />
        <circle cx="70" cy="120" r="4" fill="#eb4034" />
        <circle cx="90" cy="160" r="4" fill="#eb4034" />
        <circle cx="110" cy="130" r="4" fill="#eb4034" />
        <circle cx="130" cy="170" r="4" fill="#eb4034" />
      </svg>
      <p class="card-text">4</p>
    </div>
  </div>
</div>

<style>
@import "$lib/styles/main.css";

.card-container {
padding-bottom: 5vh;
display: flex;
flex-direction: column;
align-items: center;
text-align: center; /* Center the text under the card */
}
.card:hover {
transform: scale(1.2);
box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.8); /* White glow effect */
}

.card {
box-shadow: 0px 0px 0px rgba(255, 255, 255, 0); /* Initial shadow is none */
padding-top: 2vh;
padding-left: 2vw;
padding-right: 2vw;
background-color: #171717;
border-radius: 12px;
width: 20vw; /* Adjust width as needed */
height: auto;
display: inline-block;
vertical-align: top;
}

.card-text {
font-size: 2vw;
font-weight: 700;
}

.allCardContainer {
display: flex;
flex-wrap: wrap;
justify-content: space-evenly; /* Evenly space the cards */
padding: 20px;
}

@media (max-width: 768px) {
.card-text {
font-size: 5vw;
}
.allCardContainer {
display: block;
}
.card {
width: 60vw;
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
</style>
