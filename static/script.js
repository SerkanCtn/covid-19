let scalerParams = null;
let modelWeights = null;

// Activation functions
const relu = (x) => tf.maximum(0, x);
const sigmoid = (x) => tf.sigmoid(x);

async function loadModelData() {
    try {
        const [scalerRes, weightsRes] = await Promise.all([
            fetch('static/scaler_params.json'),
            fetch('static/model_weights.json')
        ]);
        if (!scalerRes.ok || !weightsRes.ok) throw new Error("Could not fetch model/scaler data files.");
        
        scalerParams = await scalerRes.json();
        modelWeights = await weightsRes.json();
        console.log("Model and Scaler loaded successfully.");
    } catch (error) {
        console.error("Error loading model data:", error);
    }
}

function scaleData(input) {
    return input.map((val, i) => (val - scalerParams.mean[i]) / scalerParams.scale[i]);
}

async function predict(input) {
    if (!modelWeights) throw new Error("Model weights not loaded.");

    return tf.tidy(() => {
        let x = tf.tensor2d([input]);

        // Layer 1: Dense 64 (ReLU)
        let w1 = tf.tensor2d(modelWeights.dense1_w);
        let b1 = tf.tensor1d(modelWeights.dense1_b);
        x = relu(tf.add(tf.matMul(x, w1), b1));

        // Layer 2: Dense 32 (ReLU)
        let w2 = tf.tensor2d(modelWeights.dense2_w);
        let b2 = tf.tensor1d(modelWeights.dense2_b);
        x = relu(tf.add(tf.matMul(x, w2), b2));

        // Layer 3: Dense 16 (ReLU)
        let w3 = tf.tensor2d(modelWeights.dense3_w);
        let b3 = tf.tensor1d(modelWeights.dense3_b);
        x = relu(tf.add(tf.matMul(x, w3), b3));

        // Layer 4: Dense 1 (Sigmoid)
        let w4 = tf.tensor2d(modelWeights.dense4_w);
        let b4 = tf.tensor1d(modelWeights.dense4_b);
        x = sigmoid(tf.add(tf.matMul(x, w4), b4));

        return x.dataSync()[0];
    });
}

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!modelWeights || !scalerParams) {
        alert("Model data is still loading or failed to load. Please refresh the page.");
        return;
    }

    const submitBtn = e.target.querySelector('button');
    submitBtn.innerText = 'Analyzing in Browser...';
    submitBtn.disabled = true;

    const formData = new FormData(e.target);
    const rawData = Object.fromEntries(formData.entries());

    const features = [
        parseInt(rawData.usmer || 2),
        parseInt(rawData.medical_unit || 1),
        parseInt(rawData.sex || 1),
        parseInt(rawData.patient_type || 1),
        parseInt(rawData.intubed || 2),
        parseInt(rawData.pneumonia || 2),
        parseInt(rawData.age || 30),
        parseInt(rawData.pregnant || 2),
        parseInt(rawData.diabetes || 2),
        parseInt(rawData.copd || 2),
        parseInt(rawData.asthma || 2),
        parseInt(rawData.inmsupr || 2),
        parseInt(rawData.hipertension || 2),
        parseInt(rawData.other_disease || 2),
        parseInt(rawData.cardiovascular || 2),
        parseInt(rawData.obesity || 2),
        parseInt(rawData.renal_chronic || 2),
        parseInt(rawData.tobacco || 2),
        parseInt(rawData.classification || 3),
        parseInt(rawData.icu || 2)
    ];

    try {
        const scaledInput = scaleData(features);
        const predictionValue = await predict(scaledInput);
        
        const riskPercent = (predictionValue * 100).toFixed(1);
        
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('riskScore').innerText = `${riskPercent}%`;
        document.getElementById('riskLevel').innerText = (predictionValue > 0.5 ? 'High' : 'Low') + ` Risk Profile`;
        
        const desc = document.getElementById('riskDescription');
        if (predictionValue > 0.5) {
            desc.innerText = "High alert. The model identifies significant clinical risk factors that match patterns observed in critical patient cases. Immediate medical attention is advised.";
            document.querySelector('.result-circle').style.borderColor = '#ec4899';
        } else {
            desc.innerText = "The analysis indicates a lower probability of critical complications. However, standard precautions and monitoring should continue as per medical guidelines.";
            document.querySelector('.result-circle').style.borderColor = '#6366f1';
        }
        
        document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
        console.error('Prediction error:', error);
        alert('An error occurred during prediction: ' + error.message);
    } finally {
        submitBtn.innerText = 'Analyze Risk Profile';
        submitBtn.disabled = false;
    }
});

// Smooth scroll for nav links
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Load data on start
loadModelData();
