class Connection {
    constructor(weight) {
        let myWeight = weight;
        let myOutputWeight;
    }
}

class Neuron {
    constructor(numOutputs, myIndex) {
        this.index = myIndex;
        this.myOutputVal;
        this.myOutputWeights; // { weight: weight, deltaWeight: deltaWeight } 
        this.eta = 0.15; // training rate
        this.alpha = 0.5; // momentum


        for(let i = 0; i < numOutputs; i++) {
            this.myOutputWeights.push(new Connection(randomWeight()));
        }

    }

    feedForward(prevLayer) {
        let sum = 0;
        
        // loop through previous layer neurons outputs
        for(let n = 0; n < prevLayer.length; n++) {
            sum += prevLayer[n].getOutputVal() * prevLayer[n].myOutputWeights[this.index].weight;
        }

        this.myOutputVal = transferFunction(sum);
    }

    setOutputVal(val) {
        this.myOutputVal = val;
    }

    getOutputVal() {
        return this.myOutputVal;
    }

    randomWeight() {
        return Math.random();
    }

    transferFunction(x) {
        // tanh - output range [-1...1]
        return Math.tanh(x);
    }

    transferFunctionDerivative(x) {
        // tanh - derivative
        return 1 - (x * x);
    }

    calcOutputGradients(targetVal) {
        let delta = targetVal - this.myOutputVal;
        this.myGradient = delta * this.transferFunctionDerivative(this.myOutputVal);
    }

    calcHiddenGradients(nextLayer) {
        let dow = sumDOW(nextLayer);
        this.myGradient = dow * this.transferFunctionDerivative(this.myOutputVal);
    }

    sumDOW(nextLayer) {
        let sum = 0.0;

        for(let n = 0; n < nextLayer.length; n++) {
            sum += this.myOutputWeights[n].weight * nextLayer[n].myGradient;
        }

        return sum;
    }

    updateInputWeights(prevLayer) {
        // the weights to be updated are in the connection object in the neurons in the preceding layer
        for(let n = 0; n < prevLayer.length; n++) {
            let nueron = prevLayer[n];
            let oldDeltaWeight = nueron.myOutputWeights[this.myIndex].deltaWeight;

            let newDeltaWeight = this.eta * neuron.getOutputVal() * this.myGradient + this.alpha * oldDeltaWeight;

            neuron.myOutputWeights[this.myIndex].deltaWeight = newDeltaWeight;
            neuron.myOutputWeights[this.myIndex].weight += newDeltaWeight;
        }
    }
}

class Layer {
    constructor(neurons) {
        this.myNeurons = [];
        for(let i = 0; i < neurons; i++) {
            this.myNeurons.push(new Neuron());
        }
    }
}


class Net {
    constructor(topology) {
        this.numLayers = topology.length;
        this.myLayers = [];
        this.myRecentAverageError;
        this.myRecentAverageSmoothingFactor;

        for(let i = 0; i < this.numLayers; i++) {
            console.log("this layer should have " + topology[i] + " neurons");
            this.myLayers.push(new Layer(topology[i]));
        }
    }

    feedForward(inputVals) {
        // assign input values to output values of the input neurons
        for(let i = 0; i < inputVals.length; i++) {
            this.myLayers[0][i].setOutputVal(inputVals[i]);
        }

        // forward propogate
        for(let layerNum = 1; layerNum < this.myLayers.length; layerNum++) {
            let prevLayer = this.myLayers[layerNum - 1];
            for(let n = 0; n < this.myLayers[layerNum].length; n++) {
                this.myLayers[layerNum][n].feedForward(prevLayer);
            }
        }
    }

    backProp(targetVals) {
        // calculate overall net error (RMS)
        let outputLayer = this.myLayers[this.numLayers];
        this.myError = 0.0;

        for(let n = 0; n < outputLayer.length; n++) {
            let delta = targetVals[n] - outputLayer[n].getOutputVal();
            this.myError += delta * delta;
        }
        this.myError /= outputLayer.length;
        this.myError = Math.sqrt(this.myError);

        // Implement a recent average measurement
        this.myRecentAverageError = (this.myRecentAverageError * this.myRecentAverageSmoothingFactor + this.myError) / (myRecentAverageSmoothingFactor + 1.0);

        // calculate output layer gradients
        for(let n = 0; n < outputLayer.length; n++) {
            outputLayer[n].calcOutputGradients(targetVals[n]);
        }

        // calculate hidden layer gradients
        for(let layerNum = this.myLayers.length - 2; layerNum > 0; layerNum--) {
            let hiddenLayer = this.myLayers[layerNum];
            let nextLayer = this.myLayers[layerNum + 1];

            for(let n = 0; n < hiddenLayer.length; n++) {
                hiddenLayer[n].calcHiddenGradients(nextLayer);
            }
        }

        // for all layers from outputs to first hidden layer update connection weights
        for(let layerNum = this.myLayers.length - 1; layerNum > 0; layerNum--) {
            let layer = this.myLayers[layerNum];
            let prevLayer = this.myLayers[layerNum - 1];

            for(let n = 0; n < layer.length; n++) {
                layer[n].updateInputWeights(prevLayer);
            }

        }

    }

    getResults(resultVals) {
        //resultsVals.clear();
        for(let n = 0; n < this.myLayers[this.numLayers]; n++) {
            resultVals.push(this.myLayers[this.numLayers][n].getOutputVal());
        }
    }
}

showVectorVals(label, vals) {
    let string = label + " ";
    for(let i = 0; i < vals.length; i++) {
        string += vals[i] + " ";
    }
    console.log(string);
}

/* main */

let trainData = new TrainingData("trainingData.txt");

let topology = [];
trainData.getTopology(topology);

let myNet = new Net(topology);

let inputVals = [];
let targetVals = [];
let resultVals = [];

let trainingPass = 0;

while(!trainData.isEof()) {
    trainingPass++;
    console.log("Pass " + trainingPass);

    if(trainData.getNextInputs(inputVals) != topology[0]) {
        break;
    }

    showVectorVals("Inputs:", inputVals);
    myNet.feedForward(inputVals);

    myNet.getResults(resultVals);
    showVectorVals("Outputs:", resultVals);

    trainData.getTargetOutputs(targetVals);
    showVectorVals("Targets:", targetVals);

    myNet.backProp(targetVals);

    console.log("Net recent average error: " + myNet.getRecentAverageError());
}