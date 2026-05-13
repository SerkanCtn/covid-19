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
    if (!modelWeights) return 0;

    return tf.tidy(() => {
        let x = tf.tensor2d([input]);

        // Layer 1: Dense 64 (ReLU)
        let w1 = tf.tensor2d(modelWeights.layer_1_weights);
        let b1 = tf.tensor1d(modelWeights.layer_1_bias);
        x = relu(tf.add(tf.matMul(x, w1), b1));

        // Layer 2: Dense 32 (ReLU) - Note: Dropout is ignored during inference
        let w3 = tf.tensor2d(modelWeights.layer_3_weights);
        let b3 = tf.tensor1d(modelWeights.layer_3_bias);
        x = relu(tf.add(tf.matMul(x, w3), b3));

        // Layer 3: Dense 16 (ReLU)
        let w5 = tf.tensor2d(modelWeights.layer_5_weights);
        let b5 = tf.tensor1d(modelWeights.layer_5_bias);
        x = relu(tf.add(tf.matMul(x, w5), b5));

        // Layer 4: Dense 1 (Sigmoid)
        let w6 = tf.tensor2d(modelWeights.layer_6_weights);
        let b6 = tf.tensor1d(modelWeights.layer_6_bias);
        x = sigmoid(tf.add(tf.matMul(x, w6), b6));

        return x.dataSync()[0];
    });
}

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!modelWeights || !scalerParams) {
        alert("Model data is still loading. Please try again in a few seconds.");
        return;
    }

    const submitBtn = e.target.querySelector('button');
    submitBtn.innerText = 'Analyzing in Browser...';
    submitBtn.disabled = true;

    const formData = new FormData(e.target);
    const rawData = Object.fromEntries(formData.entries());

    // Order must match the model's training order exactly:
    // 0: USMER, 1: MEDICAL_UNIT, 2: SEX, 3: PATIENT_TYPE, 4: INTUBED, 
    // 5: PNEUMONIA, 6: AGE, 7: PREGNANT, 8: DIABETES, 9: COPD, 
    // 10: ASTHMA, 11: INMSUPR, 12: HIPERTENSION, 13: OTHER_DISEASE, 14: CARDIOVASCULAR, 
    // 15: OBESITY, 16: RENAL_CHRONIC, 17: TOBACCO, 18: CLASIFFICATION_FINAL, 19: ICU
    
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
        alert('An error occurred during prediction.');
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
