let currentEpoch = 0

// you can toogle the learning_rate to see the impacts
// you can learn more about it reading the references at README
const learning_rate = .003

// initializing the weights
let w1 = initializeWeight()
let w2 = initializeWeight()
let w3 = initializeWeight()
let w4 = initializeWeight()
let w5 = initializeWeight()
let w6 = initializeWeight()

// initializing the bias
b1 = 0
b2 = 0
b3 = 0
b4 = 0

// linear functions
const f1 = x => w1 * x + b1
const f2 = x => w2 * x + b2
const f3 = x => w3 * x + b3

// activation function. You chan choose in the web
let a = getActivationFuncion('softplus')

// returns a cubic function with x range from -1 to 1
const { x, y } = getFakeData()

// run first foward propagation and return the first random prediction
let y_pred = getPredicts(x)

// this is the most important function!
// It will recive how many epochs will wish to run and for each epoch it will:
// - Do a backpropagation (Gradient Decentent)
// - update the weights and bieases
// - update the predictions points
// - call the function to update the chart
// - call the function to update the Network Graph
// you can learn more about it reading the references at README
function runEpochs(epochs) {
    let epoch = 0

    while (epoch < epochs) {
        const dSSR_w1 = getGradient1(w4, f1)
        const dSSR_w2 = getGradient1(w5, f2)
        const dSSR_w3 = getGradient1(w6, f3)

        const dSSR_w4 = getGradient2(f1)
        const dSSR_w5 = getGradient2(f2)
        const dSSR_w6 = getGradient2(f3)

        const dSSR_b1 = getGradientBias1(w4, f1)
        const dSSR_b2 = getGradientBias1(w5, f2)
        const dSSR_b3 = getGradientBias1(w6, f3)

        const dSSR_b4 = getGradientBias2()

        const step_size_dSSR_w1 = dSSR_w1 * learning_rate
        const step_size_dSSR_w2 = dSSR_w2 * learning_rate
        const step_size_dSSR_w3 = dSSR_w3 * learning_rate
        const step_size_dSSR_w4 = dSSR_w4 * learning_rate
        const step_size_dSSR_w5 = dSSR_w5 * learning_rate
        const step_size_dSSR_w6 = dSSR_w6 * learning_rate

        const step_size_dSSR_b1 = dSSR_b1 * learning_rate
        const step_size_dSSR_b2 = dSSR_b2 * learning_rate
        const step_size_dSSR_b3 = dSSR_b3 * learning_rate
        const step_size_dSSR_b4 = dSSR_b4 * learning_rate


        w1 = w1 - step_size_dSSR_w1
        w2 = w2 - step_size_dSSR_w2
        w3 = w3 - step_size_dSSR_w3
        w4 = w4 - step_size_dSSR_w4
        w5 = w5 - step_size_dSSR_w5
        w6 = w6 - step_size_dSSR_w6

        b1 = b1 - step_size_dSSR_b1
        b2 = b2 - step_size_dSSR_b2
        b3 = b3 - step_size_dSSR_b3
        b4 = b4 - step_size_dSSR_b4

        y_pred = getPredicts(x)
        epoch++
    }

    currentEpoch = currentEpoch + epoch
    document.getElementById('epochs').textContent = currentEpoch
    document.getElementById('loss').textContent = getLoss()

    myChart.data.datasets[1].data = y_pred
    myChart.update()
    drawNetworkGrph()
}

// just to initialize the view
document.addEventListener("DOMContentLoaded", () => {
    // just to plot the real data and the prediction
    plotGraph()

    // this will plot the network graph (input, output and neurons)
    drawNetworkGrph()
    document.getElementById('loss').textContent = getLoss()
});

// this function recieves the original x values and return the predicted y values
function getPredicts(x) {
    return x.map(x => {
        return a(f1(x)) * w4 + a(f2(x)) * w5 + a(f3(x)) * w6 + b4
    })
}

// this functions will verify how good our network is using residual sum of squares
// it's not the best aproach because it can leave to local minima, but it's easier to explain
function getLoss() {
    return sumArray(y.map((y, i) => Math.pow((y - y_pred[i]), 2)))
}

// this will generate a random value rangin from -1 to 1
function initializeWeight() {
    return Math.random() * isPositive()
}

// function to change the activation function using the select from ui
// this do not change the gradients, but it's fun to test. 
// ReLu shows great results in many cases
function setActivationFunction() {
    var val = document.getElementById("select").value
    a = getActivationFuncion(val)
}

function getGradient1(w, f) {
    return sumArray(x.map((x, i) => {
        return -2 * (y[i] - y_pred[i]) * w * x * Math.pow(Math.E, f(x)) / (1 + Math.pow(Math.E, f(x)))
    }))
}

function getGradient2(f) {
    return sumArray(x.map((x, i) =>
        -2 * (y[i] - y_pred[i]) * a(f(x))
    ))
}

function getGradientBias1(w, f) {
    return sumArray(x.map((x, i) =>
        -2 * (y[i] - y_pred[i]) * w * Math.pow(Math.E, f(x)) / (1 + Math.pow(Math.E, f(x)))
    ))
}

function getGradientBias2() {
    return sumArray(x.map((_, i) =>
        -2 * (y[i] - y_pred[i])
    ))
}

function sumArray(arr) {
    return arr.reduce((a, b) => a + b, 0)
}

// more about activation functions can be found in README.md
function getActivationFuncion(name) {
    const functions = {
        'sigmoid': x => 1 / (1 + Math.pow(Math.E, -x)),
        'relu': x => x < 0 ? 0 : x,
        'softplus': x => Math.log(1 + Math.pow(Math.E, x)),
        'tanh': x => Math.tanh(x)
    }

    return functions[name]
}

function isPositive() {
    return Math.random() > 0.5 ? -1 : 1
}

function getFakeData() {
    let a = Math.random() * isPositive() * 1.1
    let b = (Math.random() * isPositive()) / 3
    let c = Math.random() * isPositive()

    const range = function (start, stop, step) {
        step = step || 1
        let arr = []

        for (let i = start; i < stop; i += step) {
            arr.push(i)
        }

        return arr
    }

    const x = range(-1, 1, .01)

    return {
        x,
        y: x.map(x => a * Math.pow(x, 3) + b * Math.pow(x, 2) + c)
    }
}

function plotGraph() {
    const data = {
        labels: x,
        datasets: [
            {
                label: 'Real Data',
                data: y,
                backgroundColor: 'rgb(255, 99, 132)',
            },
            {
                label: 'Predicted',
                data: y_pred,
                backgroundColor: "rgba(0,255,0,.1)",
            },
        ]
    }

    let ctx = document.getElementById('myChart').getContext('2d')
    myChart = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'linear'
                },
            }
        },
    })
}

// the next functions are used just to prety print the Network Graph
function normalize(val, max, min) {
    return (val - min) / (max - min)
}

function normalize_array(arr) {
    const hold_normed_values = []
    const max = Math.max.apply(null, arr)
    const min = Math.min.apply(null, arr)

    arr.forEach((this_num) => {
        const val = normalize(this_num, max, min) + .1
        hold_normed_values.push(val > 1 ? 1 : val)
    })

    return hold_normed_values
}

function getSum(f) {
    return sumArray(x.map(x => {
        return a(f(x))
    }))
}

function drawNetworkGrph() {
    const [a1, a2, a3] = normalize_array([getSum(f1), getSum(f2), getSum(f3)])
    const ws1 = normalize_array([w1 + b1, w2 + b2, w3 + b3])
    const ws2 = normalize_array([w4, w5, w6])
    const len = y.length
    const [input, output] = [sumArray(normalize_array((y))) / len, sumArray(normalize_array((y_pred))) / len]
    const elements = [{
        "data": {
            "id": "e121",
            opacity: ws1[0],
            "weight": 19,
            "source": "input",
            "target": "a(z1)",
            label: "w1 * x + b1"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "e272",
            opacity: ws1[2],
            "weight": 77,
            "source": "input",
            "target": "a(z3)",
            label: "w3 * x + b3"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "e295",
            opacity: ws1[1],
            "weight": 98,
            "source": "input",
            "target": "a(z2)",
            label: "w2 * x + b2"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "aaa",
            opacity: ws2[1],
            "weight": 98,
            "source": "a(z2)",
            label: "w5",
            "target": "output"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "bbb",
            opacity: ws2[0],
            "weight": 98,
            "source": "a(z1)",
            label: "w4",
            "target": "output"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "c",
            opacity: ws2[2],
            "weight": 98,
            "source": "a(z3)",
            label: "w6",
            "target": "output"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "input",
            opacity: input,
            "weight": 77
        },
        "position": {
            "x": 150,
            "y": 0
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "a(z3)",
            opacity: a3,
            "weight": 3
        },
        "position": {
            "x": 250,
            "y": 100
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "a(z2)",
            opacity: a2,
            "weight": 33
        },
        "position": {
            "x": 150,
            "y": 100
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "a(z1)",
            opacity: a1,
            "weight": 23
        },
        "position": {
            "x": 50,
            "y": 100
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "output",
            opacity: output,
            "weight": 65
        },
        "position": {
            "x": 150,
            "y": 200
        },
        "group": "nodes"
    }
    ]

    if (window.cy && window.cy.json) {
        window.cy.json({ elements })
        return
    }

    window.cy = cytoscape({
        userZoomingEnabled: false,
        userPanningEnabled: false,
        boxSelectionEnabled: false,
        container: document.getElementById('cy'),
        layout: {
            name: 'preset',
            fit: true
        },
        style: [
            {
                selector: 'node',
                style: {
                    'height': 12,
                    'width': 12,
                    "label": "data(id)",
                    'background-color': '#ff0000',
                    'opacity': "data(opacity)",
                    'text-valign':'center',
                    'text-halign':'left', 
                }
            },
            {
                selector: 'edge',
                style: {
                    'curve-style': 'haystack',
                    'haystack-radius': 0,
                    'width': 3,
                    'opacity': "data(opacity)",
                    "font-size": "10px",
                    'line-color': '#000000'
                }
            },
            {
                selector: "edge[label]",
                css: {
                    "label": "data(label)",
                    "text-rotation": "autorotate",
                    color: 'black',
                    "text-margin-x": "11px",
                    "text-margin-y": "0px"
                }
            },
        ],
        elements
    })
}