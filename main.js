let currentEpoch = 0
const learning_rate = .003
const epochs = 40000

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

    myChart.data.datasets[1].data = y_pred;
    myChart.update()
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
    }
    ))
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

function getWeight() {
    const isPositive = Math.random() > 0.5 ? -1 : 1

    return Math.random() * isPositive
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

function getFakeData() {
    const range = function (start, stop, step) {
        step = step || 1;
        let arr = [];

        for (let i = start; i < stop; i += step) {
            arr.push(i);
        }

        return arr;
    }

    let i = 0
    const x = range(-1, 1, .01)

    return {
        x,
        y: x.map(x => Math.pow(x, 3))
    }
}

function plotGraph() {
    const data = {
        labels: x,
        datasets: [
            {
                label: 'Real Data',
                data: y,
                borderColor: 'rgb(255, 99, 132)',
            },
            {
                label: 'Predicted',
                data: y_pred,
                borderColor: 'rgb(54, 162, 235)',
                borderDash: [1, 10]
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
