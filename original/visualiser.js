class Visualiser{
    constructor(model,canvas){
        this.model = model;
        this.canvas = canvas
        this.row = 100;
        this.col = 100;
        this.cellSize = canvas.width / this.row;
    }



    display(ctx,datapoints,labels){
        for(let i = 0; i < this.row; i++){
            for(let j = 0; j < this.col; j++){
                let output = this.model.predict([i/this.row,j/this.col]);
                ctx.beginPath();
                ctx.rect(i * this.cellSize, j * this.cellSize, this.cellSize, this.cellSize);
                ctx.fillStyle = `rgb(${output * 255},${output * 255}, ${output * 255})`;
                ctx.fill();
            }
        }

        for(let i = 0; i < datapoints.length; i++){
            let datapoint = datapoints[i];
            let x = Math.floor(datapoint[0] * this.row) * this.cellSize + this.cellSize/2;
            let y = Math.floor(datapoint[1] * this.col) * this.cellSize + this.cellSize/2;
            ctx.beginPath();
            ctx.arc(x,y,this.cellSize/2,0,Math.PI * 2);
            ctx.fillStyle = `rgb(${labels[i] * 255},${labels[i] * 255}, ${labels[i] * 255})`;
            ctx.fill();
        }
    }
}