import {attachPointBrush} from './brush.js';
let brushStart = null;
import {sample,parseCSV} from './data.js';
import {palette,classColor} from './palette.js';
let classCount=2,isCustom=false;
const $=id=>document.getElementById(id),mint=[131,232,206],violet=[172,155,250];
let preset='clusters',points=sample(preset),version=0,state=null,mode='draw',label=0,probe=[.5,.5],activeConfig={hidden:[4,3],rate:.01,seed:42};
let epochTarget = 20000;
const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
const send=(type,extra={})=>worker.postMessage({type,version,...extra});
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function config(){const hidden=$('hidden').value.split(',').map(v=>v.trim()).map(v=>v===''?NaN:Number(v)),rate=Number($('rate').value),seed=Number($('seed').value);if(hidden.length<1||hidden.length>3||hidden.some(n=>!Number.isInteger(n)||n<1||n>12))throw Error('Enter 1–3 hidden layers, each containing 1–12 neurons.');if(!Number.isInteger(seed)||seed<0||seed>999999||$('seed').value==='')throw Error('Choose a seed from 0 to 999999.');return{hidden,rate,seed};}
function reset(next=activeConfig){activeConfig=next;version++;state=null;send('init',{...next,points,epochTarget,classCount});$('architecture').textContent=[2,...next.hidden,classCount===2?1:classCount].join(' → ');$('status').textContent='Resetting';$('train').disabled=true;$('step').disabled=true;}
function safe(fn){return(...args)=>{try{return fn(...args);}catch(e){message(e.message,true);}};}
worker.onmessage=({data:d})=>{if(d.version!==version)return;if(d.type==='error'){message(d.message,true);$('status').textContent='Paused';$('train').textContent='▶ Start training';if(state)state.running=false;return;}if(d.type==='probe'){if(state){state.network=d.network;drawNetwork();}return;}state=d;epochTarget=d.epochTarget;$('epoch').textContent=d.epoch.toLocaleString();$('epoch-caption').textContent='of '+epochTarget.toLocaleString()+' target';$('loss').textContent=d.metrics?d.metrics.loss.toFixed(4):'—';$('accuracy').textContent=d.validation?(d.validation.accuracy*100).toFixed(0)+'%':'—';$('accuracy-caption').textContent=d.validation?'on held-out points':'no held-out points';$('train-accuracy').textContent='Training accuracy: '+(d.metrics?(d.metrics.accuracy*100).toFixed(0)+'%':'—');$('split-count').textContent=`${d.train.length} train / ${d.test.length} test`;$('point-count').textContent=points.length+' points';$('status').textContent=d.running?'Training':d.epoch>=epochTarget?'Target reached':d.epoch?'Paused':'Ready';$('train').textContent=d.running?'Ⅱ Pause training':'▶ Start training';$('train').disabled=points.length<2||d.epoch>=epochTarget;$('step').disabled=points.length<2||d.epoch>=epochTarget;drawMap();drawNetwork();drawCurve();};
worker.onerror=()=>{message('Model execution stopped. Reload the page to restart.',true);$('status').textContent='Error';$('train').disabled=true;$('step').disabled=true;};
function color(p,alpha=1){return `rgba(${mint.map((v,i)=>Math.round(v*(1-p)+violet[i]*p)).join(',')},${alpha})`;}
function canvas(id){const el=$(id),box=el.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);el.width=Math.round(box.width*ratio);el.height=Math.round(box.height*ratio);const ctx=el.getContext('2d');ctx.scale(ratio,ratio);return{ctx,w:box.width,h:box.height};}
function drawMap(){if(!state)return;const{ctx,w,h}=canvas('map');for(let y=0;y<55;y++)for(let x=0;x<55;x++){const output=state.grid[y*55+x];const scores=Array.isArray(output)?output:[1-output,output];const winner=scores.indexOf(Math.max(...scores));const rgb=Array.isArray(output)?palette[winner].rgb:mint.map((v,i)=>v*(1-output)+violet[i]*output);const strength=Array.isArray(output)?.18+.42*scores[winner]:.46;ctx.fillStyle='rgb('+rgb.map((v,i)=>Math.round(v*strength+[25,30,41][i]*(1-strength))).join(',')+')';ctx.fillRect(x*w/55,y*h/55,w/55+1,h/55+1);}ctx.strokeStyle='#ffffff14';ctx.lineWidth=1;for(let i=1;i<10;i++){ctx.beginPath();ctx.moveTo(i*w/10,0);ctx.lineTo(i*w/10,h);ctx.moveTo(0,i*h/10);ctx.lineTo(w,i*h/10);ctx.stroke();}for(const [set,held]of[[state.train,false],[state.test,true], [brushStart === null ? [] : points.slice(brushStart),false]])for(const p of set){ctx.beginPath();ctx.arc(p.x*w,p.y*h,4.5,0,Math.PI*2);ctx.fillStyle=held?'#171e2c':classColor(p.label);ctx.fill();ctx.strokeStyle=held?classColor(p.label):'#111722';ctx.lineWidth=held?2:1.5;ctx.stroke();}if(mode==='inspect'){const x=probe[0]*w,y=probe[1]*h;ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.moveTo(x-15,y);ctx.lineTo(x+15,y);ctx.moveTo(x,y-15);ctx.lineTo(x,y+15);ctx.stroke();}}
function drawNetwork(){if(!state)return;const {ctx,w,h}=canvas('network'),n=state.network,a=n.activations,pad=35;const pos=(l,j)=>({x:pad+l*(w-pad*2)/(a.length-1),y:20+(j+1)*(h-40)/(a[l].length+1)});for(let l=0;l<n.weights.length;l++)for(let j=0;j<n.weights[l].length;j++)for(let i=0;i<n.weights[l][j].length;i++){const from=pos(l,i),to=pos(l+1,j),v=n.weights[l][j][i];ctx.beginPath();ctx.moveTo(from.x,from.y);ctx.lineTo(to.x,to.y);ctx.strokeStyle=color(v<0?1:0,Math.min(.65,.1+Math.abs(v)/8));ctx.lineWidth=Math.min(3,.5+Math.abs(v)/4);ctx.stroke();}for(let l=0;l<a.length;l++)for(let j=0;j<a[l].length;j++){const p=pos(l,j),v=a[l][j],r=Math.min(18,(h-45)/(a[l].length+1)/2.4);ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle='#1a2630';ctx.fill();const outputNode=l===a.length-1&&classCount>2;ctx.fillStyle=outputNode?classColor(j,.22+.5*v):color(v,.22+.5*v);ctx.fill();ctx.strokeStyle=outputNode?classColor(j):color(v);ctx.lineWidth=1.4;ctx.stroke();if(r>=12){ctx.fillStyle='#f1f8fa';ctx.font='10px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(v.toFixed(1),p.x,p.y);}}$('probability').textContent=n.score.toFixed(3);$('probability').style.color=classColor(n.label);$('prediction').textContent=`${palette[n.label].name} · class ${n.label} · top score`;for(let i=0;i<classCount;i++){$('score-value-'+i).textContent=n.scores[i].toFixed(3);$('score-bar-'+i).style.width=(n.scores[i]*100)+'%';}$('probe-coords').textContent=`(${probe[0].toFixed(2)}, ${probe[1].toFixed(2)})`;}
function drawCurve(){if(!state)return;const{ctx,w,h}=canvas('curve'),left=42,right=w-14,top=12,bottom=h-27;const rows=state.history,maximum=Math.max(.01,...rows.flatMap(r=>[r.loss??0,r.testLoss??0]))*1.12;ctx.font='11px monospace';ctx.textBaseline='middle';for(let i=0;i<4;i++){const y=top+i*(bottom-top)/3;ctx.strokeStyle='#ffffff12';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#9ba9bf';ctx.textAlign='right';ctx.fillText((maximum*(1-i/3)).toFixed(2),left-7,y);}ctx.textAlign='left';ctx.fillText('0',left,bottom+17);ctx.textAlign='right';ctx.fillText(state.epoch+' epochs',right,bottom+17);for(const [key,c]of[['loss',0],['testLoss',1]]){ctx.strokeStyle=color(c);ctx.lineWidth=2;ctx.beginPath();let start=true;for(const r of rows){if(r[key]===null)continue;const x=left+(right-left)*r.epoch/Math.max(1,state.epoch),y=bottom-(bottom-top)*r[key]/maximum;if(start){ctx.moveTo(x,y);start=false;}else ctx.lineTo(x,y);}ctx.stroke();}if(state.epoch===0){ctx.fillStyle='#aebbcf';ctx.textAlign='center';ctx.fillText('Start training to trace the learning curve',(left+right)/2,(top+bottom)/2);}}
function datasetChanged(text){reset();message(text+' Network and history reset.');}
function describeDataset() {
  $('dataset-description').textContent = isCustom ? 'Custom points. Choose a color and add it to the map.' :
    preset === 'xor' ? 'Opposite corners share a class. A two-color challenge.' :
    preset === 'circle' ? (classCount === 2 ? 'One class surrounds the other.' : 'Concentric bands, one for each color.') :
    `${classCount} groups, one for each color.`;
  document.querySelectorAll('[data-preset]').forEach(b => {
    const selected = !isCustom && b.dataset.preset === preset;
    b.classList.toggle('selected', selected);
    b.setAttribute('aria-pressed', String(selected));
    b.disabled = b.dataset.preset === 'xor' && classCount > 2;
    b.title = b.disabled ? 'XOR is a two-color dataset. Choose 2 colors to use it.' : '';
  });
}
function renderColors() {
  $('class-count').value = String(classCount);
  if (label >= classCount) label = 0;
  $('class-buttons').replaceChildren();
  $('map-legend').replaceChildren();
  $('class-scores').replaceChildren();
  for (let i = 0; i < classCount; i++) {
    const name = `${i} · ${palette[i].name}`;
    const button = document.createElement('button');
    button.id = 'class' + i;
    button.innerHTML = `<i class="dot" style="background:${classColor(i)}"></i>${i}`;
    button.setAttribute('aria-label', `Add ${palette[i].name} points (class ${i})`);
    button.title = name;
    button.classList.toggle('selected', label === i);
    button.setAttribute('aria-pressed', String(label === i));
    button.onclick = () => { label = i; renderColors(); if (state) drawNetwork(); };
    $('class-buttons').append(button);
    const legend = document.createElement('span');
    legend.innerHTML = `<i class="dot" style="background:${classColor(i)}"></i>${name}`;
    $('map-legend').append(legend);
    const row = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `<span>${name}</span><div class="score-track"><div id="score-bar-${i}" style="background:${classColor(i)};width:0%"></div></div><output id="score-value-${i}">—</output>`;
    $('class-scores').append(row);
  }
  describeDataset();
}
$('class-count').onchange = safe(() => {
  const next = Number($('class-count').value);
  if (isCustom && points.some(p => p.label >= next)) {
    $('class-count').value = String(classCount);
    throw Error('Your data contains colors beyond this selection. Clear the points or load a compatible dataset first.');
  }
  classCount = next;
  if (classCount > 2 && preset === 'xor') preset = 'clusters';
  if (!isCustom) points = sample(preset, Number($('count').value), activeConfig.seed, classCount);
  renderColors();
  datasetChanged(`Selected ${classCount} colors.`);
});
document.querySelectorAll('[data-preset]').forEach(button => button.onclick = safe(() => {
  preset = button.dataset.preset;
  isCustom = false;
  points = sample(preset, Number($('count').value), activeConfig.seed, classCount);
  describeDataset();
  datasetChanged('Loaded ' + button.textContent.trim() + '.');
}));

$('epoch-form').onsubmit = event => {
  event.preventDefault();
  const input = $('epoch-target');
  const target = Number(input.value);
  if (!input.value.trim() || !Number.isSafeInteger(target) || target < 1) {
    input.setCustomValidity('Enter a positive whole number for the epoch target.');
    input.reportValidity();
    return;
  }
  input.setCustomValidity('');
  send('setEpochTarget', { epochTarget: target });
  message('Epoch target set to ' + target.toLocaleString() + '. ' +
    (state && state.epoch >= target
      ? 'This run has already reached the target.'
      : state?.running ? 'Training continues toward this target.'
      : 'Press Start training to continue.'));
};
$('epoch-target').oninput = () => $('epoch-target').setCustomValidity('');

$('count').oninput=()=>{$('count-value').value=$('count').value;};
$('resample').onclick=safe(()=>{isCustom=false;points=sample(preset,Number($('count').value),activeConfig.seed,classCount);describeDataset();datasetChanged('Sampled '+points.length+' points with seed '+activeConfig.seed+'.');});
$('apply').onclick=safe(()=>{reset(config());message('Configuration applied. Weights and history reset; dataset preserved.');});
$('reset').onclick=safe(()=>{reset();message('Weights reset with the same seed.');});
$('train').onclick=()=>{send(state?.running?'pause':'run');message(state?.running?'Training paused.':'Training on your points. Inspect the map while the model learns.');};
$('step').onclick=()=>{send('step');message('One epoch: one randomly sampled update per training point.');};
function setMode(next){mode=next;for(const name of ['draw','inspect']){$(name).classList.toggle('selected',name===mode);$(name).setAttribute('aria-pressed',String(name===mode));}$('point-action').textContent=mode==='draw'?'Add point':'Inspect point';drawMap();}
$('draw').onclick=()=>setMode('draw');$('inspect').onclick=()=>setMode('inspect');
function pointAction(x,y){if(![x,y].every(v=>Number.isFinite(v)&&v>=0&&v<=1))throw Error('Coordinates must be numbers between 0 and 1.');if(mode==='inspect'){probe=[x,y];send('probe',{input:probe});drawMap();}else{if(points.length>=1000)throw Error('The limit is 1,000 points.');points.push({x,y,label});isCustom=true;describeDataset();datasetChanged('Added class '+label+' at ('+x.toFixed(2)+', '+y.toFixed(2)+').');}}
function updateCoordinates(x, y) {
  $('px').value = x.toFixed(3);
  $('py').value = y.toFixed(3);
}
attachPointBrush($('map'), {
  isDrawing: () => mode === 'draw',
  onStart() {
    if (!state) return false;
    if (points.length >= 1000) {
      message('The limit is 1,000 points. Clear some space by loading a new dataset.', true);
      return false;
    }
    // Pause the old model and ignore its in-flight snapshots during this stroke.
    send('pause');
    version++;
    state.running = false;
    brushStart = points.length;
    $('status').textContent = 'Drawing';
    $('train').disabled = true;
    $('step').disabled = true;
    return true;
  },
  onPoint(x, y) {
    if (points.length >= 1000) return false;
    points.push({ x, y, label });
    updateCoordinates(x, y);
    return points.length < 1000;
  },
  onUpdate() {
    $('point-count').textContent = points.length + ' points';
    drawMap();
  },
  onEnd() {
    const added = points.length - brushStart;
    brushStart = null;
    isCustom = true;
    describeDataset();
    datasetChanged(`Painted ${added} ${palette[label].name.toLowerCase()} points.` +
      (points.length >= 1000 ? ' The 1,000-point limit is reached.' : ''));
  },
  onInspect: safe((x, y) => { updateCoordinates(x, y); pointAction(x, y); })
});
$('point-action').onclick=safe(()=>{if(!$('px').value||!$('py').value)throw Error('Enter both x and y.');pointAction(Number($('px').value),Number($('py').value));});
$('clear').onclick=()=>{points=[];isCustom=true;describeDataset();datasetChanged('Canvas cleared. Add at least two points to train.');};
$('about').onclick=()=>$('about-dialog').showModal();$('import-open').onclick=()=>$('import-dialog').showModal();document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('csv-file').onchange=async()=>{try{const f=$('csv-file').files[0];if(!f)return;if(f.size>100000)throw Error('Choose a CSV smaller than 100 KB.');$('csv').value=await f.text();$('import-error').textContent='';}catch(e){$('import-error').textContent=e.message;}};
$('import-confirm').onclick=()=>{try{const next=parseCSV($('csv').value,7);points=next;isCustom=true;classCount=Math.max(classCount,...points.map(p=>p.label+1));if(classCount>2&&preset==='xor')preset='clusters';renderColors();document.querySelectorAll('[data-preset]').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed','false');});$('dataset-description').textContent='Custom CSV dataset loaded.';$('import-error').textContent='';$('import-dialog').close();datasetChanged('Imported '+points.length+' points.');}catch(e){$('import-error').textContent=e.message;}};
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{drawMap();drawNetwork();drawCurve();},80);});
// Optional structured read access uses the same live state as the visible UI.
if(document.modelContext?.registerTool){try{Promise.resolve(document.modelContext.registerTool({name:'read_neural_experiment',title:'Read neural experiment',description:'Read current neural network configuration, training progress, metrics and inspected output.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute(input){if(!input||typeof input!=='object'||Object.keys(input).length)throw Error('This tool accepts an empty object.');if(!state)throw Error('The model is initializing.');return{configuration:{...activeConfig,classCount},epoch:state.epoch,epochTarget,running:state.running,training:state.metrics,test:state.validation,probe,output:state.network.output,pointCount:points.length};}})).catch(()=>{});}catch{}}
renderColors();
reset();


