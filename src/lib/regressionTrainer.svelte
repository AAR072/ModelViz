<script lang="ts">
let { functionType, model } = $props();
console.log(model)
let epochs: number = $state(1000);
let framerate: number = $state(15);
let length: number = $state(5);
let snapshotRate: number = $derived(Math.floor(epochs / (length * framerate)));
</script>
<div class="goodDiv">
  <p id="pageTitle">Training Parameters</p>
  <div class="trainingParams">
    <label>
      Epochs:
      <input
        class="unitInput"
        type="number"
        defaultvalue={1000}
        bind:value={epochs}
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

    <p>Snapshot every {snapshotRate} epochs</p>
  </div>
  <button class="boton-elegante" id="greenElon">Start Training</button>

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
