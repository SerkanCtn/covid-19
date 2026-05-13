document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = e.target.querySelector('button');
    submitBtn.innerText = 'Analyzing...';
    submitBtn.disabled = true;

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (result.success) {
            const riskPercent = (result.prediction * 100).toFixed(1);
            
            document.getElementById('result').classList.remove('hidden');
            document.getElementById('riskScore').innerText = `${riskPercent}%`;
            document.getElementById('riskLevel').innerText = `${result.risk_level} Risk Profile`;
            
            const desc = document.getElementById('riskDescription');
            if (result.prediction > 0.5) {
                desc.innerText = "High alert. The model identifies significant clinical risk factors that match patterns observed in critical patient cases. Immediate medical attention is advised.";
                document.querySelector('.result-circle').style.borderColor = '#ec4899';
            } else {
                desc.innerText = "The analysis indicates a lower probability of critical complications. However, standard precautions and monitoring should continue as per medical guidelines.";
                document.querySelector('.result-circle').style.borderColor = '#6366f1';
            }
            
            // Scroll to result
            document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Could not connect to the prediction server.');
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
