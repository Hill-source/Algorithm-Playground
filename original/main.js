const NeuralCanvas = document.getElementById('NeuralCanvas');
NeuralCanvas.width = 1000;
NeuralCanvas.height = 800;

const ChartCanvas = document.getElementById('ChartCanvas');
ChartCanvas.width = 1000;
ChartCanvas.height = 1000;




const NeuralCtx = NeuralCanvas.getContext('2d');
const NN = new Neural_Network([2,4,3,1]);

const ChartCtx = ChartCanvas.getContext('2d');

const visualiser = new Visualiser(NN,ChartCanvas);

const datapoints = [];
const labels = []

const mouse = {x:0, y:0, label:0};
function SetLabel(label){
    mouse.label = label;
}

ChartCanvas.onmousedown = event=>{
    mouse.x = event.clientX - ChartCanvas.offsetLeft;
    mouse.y = event.clientY - ChartCanvas.offsetTop;
    datapoints.push([mouse.x / (visualiser.row * visualiser.cellSize), mouse.y / (visualiser.col * visualiser.cellSize)]);
    labels.push([mouse.label]);
}




let inputs = [[0,1],[1,0],[1,1],[0,0]];
let targets = [[1],[1],[0],[0]];

let iteration = 0;
let fit = false;

// NN.fit(inputs,targets,5000);
visualiser.display(ChartCtx,datapoints,labels);
function start(){
    fit = !fit;
}

run();

function run(){
    ChartCtx.clearRect(0,0,ChartCanvas.width,ChartCanvas.height);
    if(fit){
        NN.fit(datapoints,labels,10,iteration);
        iteration++;
    }
    visualiser.display(ChartCtx,datapoints,labels);



    requestAnimationFrame(run);
}






// console.log(`output: ${NN.feedForward([0,1])[0]}`);
// console.log(`output: ${NN.feedForward([1,0])[0]}`);
// console.log(`output: ${NN.feedForward([1,1])[0]}`);
// console.log(`output: ${NN.feedForward([0,0])[0]}`);





// console.log(loss2)
Neural_Network.Visualise(NeuralCtx,NN);
