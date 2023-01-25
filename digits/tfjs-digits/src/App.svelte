<script lang="ts">
 import * as tf from '@tensorflow/tfjs'

 let i = 0;
 let ready = false;
 let drawing = false;
 let canvas = null;
 let ctx = null;

 const CANVAS_SIZE = 28*7;
 const factor = CANVAS_SIZE / 28

 const getEmpty = (size) => Array.from(Array(size), () => .0)
 let predictions = getEmpty(10)
 let pixels = getEmpty(28*28)

 const model_urls = [
   './data/model/model.json',
   './data/model2/model.json',
 ]
 let model = null
 const loadNet = async (i) => {
   ready = false
   model = await tf.loadLayersModel(model_urls[i])
   ready = true
   predict();
 }
 $: loadNet(i)
 $: ctx = canvas ? canvas.getContext('2d') : null

 const draw = ({offsetX, offsetY}) => {
   const p = {
     x: Math.min(27, Math.floor(offsetX / factor)),
     y: Math.min(27, Math.floor(offsetY / factor))
   }

   if (pixels[p.x + 28*p.y]) {
     return
   }
   pixels[p.x + 28*p.y] = 1.
                        ctx.beginPath();
   ctx.fillStyle='red';
   ctx.fillRect(p.x*factor, p.y*factor, factor - 1, factor -1)
   predict();
 }

 const predict = () => {
   if (!model) return
   let tensor = tf.tensor(pixels, [28, 28, 1], 'float32');
   tensor = tf.expandDims(tensor, 0)
   predictions = model.predict(tensor).dataSync()
 }

 const stopDrawing = () => {
   drawing = false
   predict()
 }

 const clear = () => {
   ctx && ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
   pixels = getEmpty(28*28)
   predictions = getEmpty(10)
 }
</script>

<main>
  <canvas width={CANVAS_SIZE} height={CANVAS_SIZE} bind:this={canvas}
          on:mousedown={() => drawing = true}
          on:mouseup={stopDrawing}
          on:mouseleave={stopDrawing}
          on:mousemove={e => drawing && draw(e)}
  >
  </canvas>
  <svg viewBox="0 0 100 110" width=200 height=220>
    {#each predictions as pred, i}
      <rect id={i} x={i*10} y={20} width=8 height={Math.floor(pred*100)}/>
      <text x={i*10} y={15}>{i}</text>
    {/each}
  </svg>
  <div>
    <h2>selected model: {i} {model_urls[i]}</h2>
    <button on:click={clear}>clear</button>
    <button on:click={() => {i = (i + 1) % model_urls.length}}>switch</button>
    {#if ready}
      <span style:color="green">ready</span>
    {:else}
      <span style:color="red">loading...</span>
    {/if}
  </div>
</main>

<style>
 canvas {
   border: solid 2px black;
 }
 rect {
   fill: red;
 }
 text {
   fill: black;
 }
</style>
