// Get DOM elements
const video = document.getElementById('webcam-video');
const captureButton = document.getElementById('capture-button');
const overlayMessage = document.getElementById('overlay-message');

let session;
let modelLoaded = false;
let isPredicting = false;

// Labels for the quality classes
const labels = ['good_quality', 'blurry', 'too_dark', 'poor_framing'];

// Step 1: Initialize ONNX.js and load the model
async function initializeModel() {
    try {
        session = new onnx.InferenceSession();
        await session.loadModel('efficientnet_quality_model.onnx');
        modelLoaded = true;
        console.log("ONNX model loaded successfully!");
        startWebcam();
    } catch (e) {
        console.error("Failed to load model:", e);
        overlayMessage.textContent = "Error loading model. Check console.";
        overlayMessage.style.display = 'block';
    }
}

// Step 2: Start the webcam feed
function startWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    startPredictionLoop();
                };
            })
            .catch(error => {
                console.error("Webcam access denied:", error);
                overlayMessage.textContent = "Webcam access denied. Please enable it.";
                overlayMessage.style.display = 'block';
            });
    }
}

// Step 3: The real-time prediction loop
async function startPredictionLoop() {
    if (!modelLoaded || isPredicting) {
        return;
    }
    isPredicting = true;
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;

    const runInference = async () => {
        try {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = context.getImageData(0, 0, canvas.width, canvas.height);

            // Preprocess the image for the model
            const inputTensor = preprocess(imageData);
            
            // Run the model prediction
            const outputMap = await session.run([inputTensor]);
            const outputTensor = outputMap.values().next().value;
            
            // Get the predicted class
            const prediction = argmax(outputTensor.data);
            const predictedLabel = labels[prediction];

            // Display feedback and enable/disable button
            updateUI(predictedLabel);

        } catch (e) {
            console.error("Prediction failed:", e);
        }

        // Loop the prediction
        requestAnimationFrame(runInference);
    };

    runInference();
}

// Step 4: Final Corrected Preprocessing function
function preprocess(imageData) {
    const { data, width, height } = imageData;
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    // Create a new Float32Array to store the preprocessed data
    const tensorData = new Float32Array(3 * width * height);
    
    let offset = 0;
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const index = (h * width + w) * 4;
            // Normalize the pixel value and place it in the tensor
            tensorData[offset++] = (data[index + 0] / 255.0 - mean[0]) / std[0]; // Red
            tensorData[offset++] = (data[index + 1] / 255.0 - mean[1]) / std[1]; // Green
            tensorData[offset++] = (data[index + 2] / 255.0 - mean[2]) / std[2]; // Blue
        }
    }
    
    // Create and return the ONNX tensor
    return new onnx.Tensor(tensorData, 'float32', [1, 3, height, width]);
}


// Helper to find the index of the max value
function argmax(array) {
    let max = array[0];
    let maxIndex = 0;
    for (let i = 1; i < array.length; i++) {
        if (array[i] > max) {
            max = array[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

// Step 5: Update the UI based on the prediction
function updateUI(predictedLabel) {
    if (predictedLabel === 'good_quality') {
        overlayMessage.style.display = 'none';
        captureButton.disabled = false;
        captureButton.textContent = 'Take Picture';
    } else {
        overlayMessage.textContent = `🚫 ${predictedLabel.replace('_', ' ')} detected!`;
        overlayMessage.style.display = 'block';
        captureButton.disabled = true;
    }
}

// Initialize the application
window.onload = initializeModel;