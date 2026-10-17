class Neural_Network{
    constructor(layerScript){
        // [Layer1, Layer2, Layer3,...]
        this.layerScript = [];

        for(let i = 1; i < layerScript.length; i++){
            this.layerScript.push(new Layer(layerScript[i-1],layerScript[i]));
        }
    }

    feedForward(inputs){
        // outputs -> inputs -> outputs
        for(let i = 0; i < this.layerScript.length;i++){
            inputs = this.layerScript[i].feedForward(inputs);
        }
        return inputs;
    }

    predict(inputs){
        return this.feedForward(inputs);
    }

    static Visualise(ctx, network){
        let horizontalSpace = ctx.canvas.width / (network.layerScript.length + 2);
        for(let i = 0; i < network.layerScript.length + 1; i++){
            if(i != network.layerScript.length){
                Layer.Visualise(ctx,network.layerScript[i],horizontalSpace * (i+1), horizontalSpace);
            }
            else {
                Layer.Visualise(ctx,network.layerScript[i-1],horizontalSpace * (i+1), horizontalSpace,true);
            }
        }
    }

    static MSE(outputs, targets){
        let loss = 0;
        for(let i = 0; i < outputs.length; i++){
            loss = loss + 1/2 * Math.pow(targets[i]-outputs[i],2);
        }
        return loss;
    }

    fit(inputs,targets,epochs,iteration = false){
        // E * sigmoid(x) * (1 - sigmoid(x)) * Lr * transpose(A)
        for(let epoch = 0; epoch < epochs; epoch++){
            let totalError = 0;

            for(let i = 0; i < inputs.length; i++){
                let index = Math.floor(Math.random() * inputs.length);
    
                let predictedY = this.feedForward(inputs[index]);


                totalError += Neural_Network.MSE(predictedY,targets[index])
                let BackwardOutputs = this.layerScript[this.layerScript.length-1].feedBackward(predictedY,targets[index],true);
    
                // Backward
                for(let j = this.layerScript.length - 2; j >= 0; j--){
                    BackwardOutputs = this.layerScript[j].feedBackward(BackwardOutputs);
    
                
                }
                if(iteration){
                    console.log(`Iteration ${iteration}, Loss: ${totalError / inputs.length}`);
                    console.log(`Train Input: ${inputs[index]}, Target: ${targets[index]}`)
                    console.log(`Predicted: ${this.feedForward(inputs[index])}`)
                    console.log('')
                } else {
                    if (epoch % 100 === 0) {console.log(`Epoch ${epoch}, Loss: ${totalError / inputs.length}`);
                    console.log(`Train Input: ${inputs[index]}, Target: ${targets[index]}`)
    
                    console.log(`Predicted: ${this.feedForward(inputs[index])}`)}
                    console.log('')

                }

         


            }


        }



        
    }
}

  

class Layer{
    constructor(inputCount, outputCount){
        this.inputs = new Array(inputCount);
        this.outputs = new Array(outputCount);
        this.biases = new Array(outputCount);
        this.weights = new Array(outputCount);


        for(let i = 0; i < outputCount; i++){
            this.biases[i] = this.randomiseBias();
            let weight = [];
            for(let j = 0; j < inputCount; j++){
                weight.push(this.randomiseWeight());
            }
            this.weights[i] = weight;
        }

    }

    feedForward(inputs){
        
        this.inputs = inputs;
        

        // Matrix multiplication: [weights] x [inputs] + [biases] = [results]
        for(let i = 0; i < this.weights.length; i++){
            let output = 0;
            for(let j = 0; j < this.weights[i].length; j++){
                output += this.weights[i][j] * this.inputs[j];
            }
            this.outputs[i] = sigmoid(output + this.biases[i]);
        }
        return this.outputs;
    }

    feedBackward(inputs, targets = false, isLastLayer = false){


        // -E * sigmoid'(x) * Lr * transpose(A)

        let errors = new Array(targets.length);

        if(isLastLayer){
            for (let i = 0; i < targets.length; i++) {
                errors[i] = (targets[i] - inputs[i]) * dsigmoid(inputs[i]);
            }
        } else {
            for (let i = 0; i < inputs.length; i++) {
                errors[i] = inputs[i] * dsigmoid(this.outputs[i]);
            }
        }
 


        let outputs = new Array(this.inputs.length);

        for(let i = 0; i < this.weights[0].length;i++){
            let output = 0;
            for(let j = 0; j < this.weights.length; j++){
                output = output + this.weights[j][i] * errors[j]; 
            }
            outputs[i] = output;
        }
   


        // Matrix multiplication: -transpose[activations] * E * sigmoid(x) * (1 - sigmoid(x))
        for(let i = 0; i < this.weights[0].length;i++){
            for(let j = 0; j < this.weights.length; j++){
      
                this.weights[j][i] += this.inputs[i] * errors[j] * 0.01; // shift

            }

        }

        for(let i = 0; i < this.weights.length; i++){
            this.biases[i] += errors[i] * 0.01;
        }
        return outputs;
    }

    static Visualise(ctx,layer,x, horizontalSpace,isLastLayer = false){

        if(!isLastLayer){
            let verticalSpace = ctx.canvas.height / (layer.inputs.length + 1);
            let verticalSpaceNextLayer = ctx.canvas.height / (layer.outputs.length + 1)
            for(let i = 0; i < layer.inputs.length; i++){
                let y = verticalSpace * (i+1);
                for(let j = 0; j < layer.outputs.length; j++){
                    ctx.beginPath();
                    ctx.strokeStyle = 'grey';
                    ctx.moveTo(x,y);
                    ctx.lineTo(x+horizontalSpace,(j+1) * verticalSpaceNextLayer);
                    ctx.stroke();
                }
           

                ctx.beginPath();
                ctx.fillStyle = 'black';
                ctx.arc(x, y,26,0,Math.PI * 2);
                ctx.fill();
            }
        } else {
            let verticalSpace = ctx.canvas.height / (layer.outputs.length + 1);
            for(let i = 0; i < layer.outputs.length; i++){
                ctx.beginPath();
                ctx.fillStyle = 'black';
                ctx.arc(x, verticalSpace * (i+1),26,0,Math.PI * 2);
                ctx.fill();
            }
        }
  

    }

    randomiseBias() {
        // Return bias in range between -1,1
        return Math.random() * 2 - 1;
    }

    randomiseWeight(){
        // Return weight in range between -1,1
        return Math.random() * 2 - 1;
    }
}





