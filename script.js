// Get DOM elements
const video = document.getElementById('webcam-video');
const captureButton = document.getElementById('capture-button');
const overlayMessage = document.getElementById('overlay-message');

let session;
let modelLoaded = false;
let isPredicting = false;

// Labels for the quality classes
const labels = ['good_quality', 'blurry', 'too_dark', 'poor_framing'];

// New variables for the timer logic
const TIMEOUT_SECONDS = 20;
let poorQualityStartTime = null;

// Step 1: Initialize ONNX.js
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
    
    const runInference = async () => {
        try {
            // Use tf.browser.fromPixels to get a tensor directly from the video frame
            const imageTensor = tf.browser.fromPixels(video)
                .resizeBilinear([224, 224])
                .toFloat()
                .div(tf.scalar(255.0))
                .sub(tf.tensor1d([0.485, 0.456, 0.406]))
                .div(tf.tensor1d([0.229, 0.224, 0.225]))
                .expandDims();
                
            // Transpose the tensor from [1, 224, 224, 3] to [1, 3, 224, 224] for ONNX
            const transposedTensor = imageTensor.transpose([0, 3, 1, 2]);

            // Convert the tf.js tensor to a raw Float32Array
            const rawTensorData = await transposedTensor.data();
            
            // Create the ONNX tensor
            const onnxTensor = new onnx.Tensor(rawTensorData, 'float32', [1, 3, 224, 224]);
            
            // Run the model prediction
            const outputMap = await session.run([onnxTensor]);
            const outputTensor = outputMap.values().next().value;
            
            // Get the predicted class
            const prediction = argmax(outputTensor.data);
            const predictedLabel = labels[prediction];

            // Display feedback and enable/disable button
            updateUI(predictedLabel);

        } catch (e) {
            console.error("Prediction failed:", e);
        }

        // Clean up the tensors to prevent memory leaks
        tf.dispose();

        // Loop the prediction
        requestAnimationFrame(runInference);
    };

    runInference();
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

// Step 4: Updated UI logic with the new timer feature
function updateUI(predictedLabel) {
    // If the image is good, reset the timer and display no message
    if (predictedLabel === 'good_quality') {
        overlayMessage.style.display = 'none';
        captureButton.textContent = 'Take Picture';
        captureButton.disabled = false;
        poorQualityStartTime = null;
    } else {
        // If the image is not good, show the message
        overlayMessage.textContent = `🚫 ${predictedLabel.replace('_', ' ')} detected!`;
        overlayMessage.style.display = 'block';
        captureButton.textContent = 'Take Anyway';
        
        // Check for the timer
        if (poorQualityStartTime === null) {
            // Start the timer when the first poor quality frame is detected
            poorQualityStartTime = Date.now();
        }

        const elapsedSeconds = (Date.now() - poorQualityStartTime) / 1000;
        
        if (elapsedSeconds >= TIMEOUT_SECONDS) {
            // After the timeout, remove the message and enable the button
            overlayMessage.textContent = 'You can take the picture now.';
            captureButton.disabled = false;
        } else {
            // Before the timeout, keep the button disabled
            captureButton.disabled = true;
        }
    }
}

// Initialize the application
window.onload = initializeModel;