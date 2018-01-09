class Neuron {
    constructor() {
        console.log('Made a neuron!');
    }
}

class Layer {
    constructor(neurons) {
        let myNeurons = [];
        for(let i = 0; i < neurons; i++) {
            myNeurons.push(new Neuron());
        }
    }
}


class Net {
    constructor(topology) {
        let numLayers = topology.length;
        let myLayers = [];
        for(let i = 0; i < numLayers; i++) {
            console.log("this layer should have " + topology[i] + " neurons");
            myLayers.push(new Layer(topology[i]));
        }
    }

    feedForward(inputVals) {

    }

    backProp(targetVals) {

    }

    getResults(resultVals) {

    }
}

/* main */

let topology = [3,2,1];
let inputVals = [];
let targetVals = [];
let resultVals = [];
let myNet = new Net(topology);
myNet.feedForward(inputVals);
myNet.backProp(targetVals);
myNet.getResults(resultVals);
