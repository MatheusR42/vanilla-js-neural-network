let currentEpoch = 0
const learning_rate = .003

let w1 = getWeight()
let w2 = getWeight()
let w3 = getWeight()
let w4 = getWeight()
let w5 = getWeight()
let w6 = getWeight()

b1 = 0
b2 = 0
b3 = 0
b4 = 0

const f1 = x => w1 * x + b1
const f2 = x => w2 * x + b2
const f3 = x => w3 * x + b3

const a = getActivationFuncion('softplus')
const { x, y } = getFakeData()
let y_pred = getPredicts(x)


function runEpochs(epochs) {
    let epoch = 0

    while (epoch < epochs) {
        dSSR_w1 = getGradient1(w4, f1)
        dSSR_w2 = getGradient1(w5, f2)
        dSSR_w3 = getGradient1(w6, f3)
        dSSR_w4 = getGradient2(f1)
        dSSR_w5 = getGradient2(f2)
        dSSR_w6 = getGradient2(f3)

        dSSR_b1 = getGradientBias1(w4, f1)
        dSSR_b2 = getGradientBias1(w5, f2)
        dSSR_b3 = getGradientBias1(w6, f3)
        dSSR_b4 = getGradientBias2()

        step_size_dSSR_w1 = dSSR_w1 * learning_rate
        step_size_dSSR_w2 = dSSR_w2 * learning_rate
        step_size_dSSR_w3 = dSSR_w3 * learning_rate
        step_size_dSSR_w4 = dSSR_w4 * learning_rate
        step_size_dSSR_w5 = dSSR_w5 * learning_rate
        step_size_dSSR_w6 = dSSR_w6 * learning_rate

        step_size_dSSR_b1 = dSSR_b1 * learning_rate
        step_size_dSSR_b2 = dSSR_b2 * learning_rate
        step_size_dSSR_b3 = dSSR_b3 * learning_rate
        step_size_dSSR_b4 = dSSR_b4 * learning_rate


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

        y_pred = x.map(x => {
            return a(f1(x)) * w4 + a(f2(x)) * w5 + a(f3(x)) * w6 + b4
        })
        epoch++
    }

    currentEpoch = currentEpoch + epoch
    document.getElementById('epochs').textContent = currentEpoch
    document.getElementById('loss').textContent = getLoss()

    myChart.data.datasets[1].data = y_pred;
    myChart.update()
    drawNetworkGrph()
}

plotGraph()

function getPredicts(x) {
    return x.map(x => {
        return a(f1(x)) * w4 + a(f2(x)) * w5 + a(f3(x)) * w6 + b4
    })
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

function getLoss() {
    return sumArray(y.map((y, i) => Math.pow((y - y_pred[i]), 2)))
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

function getWeight() {
    return Math.random() * isPositive()
}

function sumArray(arr) {
    return arr.reduce((a, b) => a + b, 0)
}

function getActivationFuncion(name) {
    const functions = {
        'relu': x => x < 0 ? 0 : x,
        'softplus': x => Math.log(1 + Math.pow(Math.E, x))
    }

    return functions[name]
}

function isPositive() {
    return Math.random() > 0.5 ? -1 : 1
}

function getFakeData() {
    let a = Math.random() * isPositive()
    let b = (Math.random() * isPositive()) / 3
    let c = Math.random() * isPositive()

    const range = function (start, stop, step) {
        step = step || 1;
        let arr = [];

        for (let i = start; i < stop; i += step) {
            arr.push(i);
        }

        return arr;
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
    };

    let ctx = document.getElementById('myChart').getContext('2d');
    myChart = new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,

        },
    })
}

drawNetworkGrph()

function normalize_array(arr) {

    normalize = function (val, max, min) {
        return (val - min) / (max - min);
    }

    max = Math.max.apply(null, arr)
    min = Math.min.apply(null, arr)

    hold_normed_values = []
    arr.forEach(function (this_num) {
        let val = normalize(this_num, max, min) + .1
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
    const ws = normalize_array([w1, w2, w3, w4, w5, w6])
    const len = y.length
    const [input, output] = [sumArray(normalize_array((y)))/len, sumArray(normalize_array((y_pred)))/len]
    const elements = [{
        "data": {
            "id": "e121",
            opacity: ws[0],
            "weight": 19,
            "source": "input",
            "target": "n1"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "e272",
            opacity: ws[2],
            "weight": 77,
            "source": "input",
            "target": "n3"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "e295",
            opacity: ws[1],
            "weight": 98,
            "source": "input",
            "target": "n2"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "aaa",
            opacity: ws[4],
            "weight": 98,
            "source": "n2",
            "target": "output"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "bbb",
            opacity: ws[3],
            "weight": 98,
            "source": "n1",
            "target": "output"
        },
        "position": {},
        "group": "edges"
    },
    {
        "data": {
            "id": "c",
            opacity: ws[4],
            "weight": 98,
            "source": "n3",
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
            "x": 200,
            "y": 0
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "n3",
            opacity: a3,
            "weight": 3
        },
        "position": {
            "x": 300,
            "y": 100
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "n2",
            opacity: a2,
            "weight": 33
        },
        "position": {
            "x": 200,
            "y": 100
        },
        "group": "nodes"
    },
    {
        "data": {
            "id": "n1",
            opacity: a1,
            "weight": 23
        },
        "position": {
            "x": 100,
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
            "x": 200,
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
        container: document.getElementById('cy'),

        layout: {
            name: 'preset'
        },

        style: [
            {
                selector: 'node',
                style: {
                    'height': 20,
                    'width': 20,
                    "label": "data(id)",
                    'background-color': '#ff0000',
                    'opacity': "data(opacity)",
                }
            },

            {
                selector: 'edge',
                style: {
                    'curve-style': 'haystack',
                    'haystack-radius': 0,
                    'width': 5,
                    'opacity': "data(opacity)",
                    'line-color': '#000000'
                }
            }
        ],
        elements
    });
}