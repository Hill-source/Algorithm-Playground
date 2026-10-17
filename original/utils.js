function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidPrime(x){
    return sigmoid(x) * (1 - sigmoid(x));
}

function dsigmoid(x){
    return x * (1 - x);
}

function sigmoidInverse(x){
    return Math.log(x/(1-x));
}

function sumArray(array){
    let result = 0;
    for(let i of array){
        result = result + i;
    }
    return result;
}



