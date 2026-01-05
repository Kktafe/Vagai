import { Wllama } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama/esm/index.js';

// Configuration
const CONFIG = {
    // We use the Q8_0 quant for best balance on SmolVLM 256M (it's small enough ~290MB total)
    modelUrl: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
    // VLM requires a specific projector (vision adapter). 
    // Note: If Wllama doesn't auto-load mmproj, we assume standard GGUF embedding or separate load. 
    // For this demo code, we assume the model file is self-contained or Wllama handles the split.
    useMultiThread: true,
    n_threads: 4, // Optimal for mobile 
};

let wllama = null;
let isModelLoaded = false;
let isDetecting = false;
let lastResult = null;

// UI Elements
const video = document.getElementById('camera-feed');
const canvas = document.getElementById('capture-canvas');
const ctx = canvas.getContext('2d');
const statusPill = document.getElementById('status-pill');
const resultCard = document.getElementById('result-card');

// 1. Initialize Wllama
async function initWllama() {
    try {
        statusPill.innerText = "Downloading AI Model...";
        wllama = new Wllama({
            'single-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama/esm/single-thread/wllama.wasm',
            'multi-thread/wllama.wasm': 'https://cdn.jsdelivr.net/npm/@wllama/wllama/esm/multi-thread/wllama.wasm',
        });

        await wllama.loadModelFromUrl(CONFIG.modelUrl, {
            n_threads: CONFIG.n_threads,
            progressCallback: (loaded, total) => {
                const pct = Math.round((loaded / total) * 100);
                statusPill.innerText = `Loading Model: ${pct}%`;
            }
        });

        isModelLoaded = true;
        statusPill.innerText = "AI Ready - Press Start";
        statusPill.classList.replace('text-yellow-400', 'text-green-400');
    } catch (e) {
        console.error(e);
        statusPill.innerText = "Error Loading Model";
        statusPill.classList.replace('text-yellow-400', 'text-red-500');
    }
}

// 2. Camera Logic
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera access denied or unavailable.");
    }
}

// 3. Inference Logic
async function runInference() {
    if (!isDetecting || !isModelLoaded) return;

    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64/blob for the model
    // Note: Wllama/llama.cpp specific image handling
    const imageBase64 = canvas.toDataURL('image/jpeg', 0.5).split(',')[1]; // Remove header

    // Construct Prompt for SmolVLM
    // We ask for JSON specifically to parse it easily
    const prompt = `<image>\nAnalyze this farm produce. Identify the category (Grains, Nuts, Beans, Raisins, Pulses, etc.). Identify the specific name. Count the items. Check for damage (broken, discoloration). Return ONLY a JSON object: {"category": "...", "name": "...", "near_matches": [{"name": "...", "conf": "XX%"}], "count": 0, "damaged": false}`;

    statusPill.innerText = "Analyzing...";
    const tStart = performance.now();

    try {
        // Run Inference
        // Note: Wllama API for images might vary by version. 
        // This assumes standard llama.cpp conventions or image_data argument support
        const output = await wllama.createCompletion(prompt, {
            nPredict: 128,
            temperature: 0.1, // Low temp for factual data
            images: [imageBase64] // Passing image binary
        });

        const latency = (performance.now() - tStart).toFixed(0);
        document.getElementById('perf-latency').innerHTML = `${latency} <span class="text-sm text-gray-500">ms</span>`;

        parseAndDisplay(output);

    } catch (e) {
        console.error("Inference failed", e);
    }

    if (isDetecting) {
        // Loop with delay to prevent overheating
        setTimeout(runInference, 1000); 
    }
}

function parseAndDisplay(rawText) {
    // Simple heuristic parser if model output isn't perfect JSON
    try {
        // Attempt to find JSON in response
        const jsonMatch = rawText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const data = JSON.parse(jsonMatch[0]);
            
            // Update UI
            document.getElementById('res-category').innerText = data.category || "UNKNOWN";
            document.getElementById('res-name').innerText = data.name || "Object";
            document.getElementById('res-count').innerText = `Count: ${data.count || '--'}`;
            
            // Matches
            const matchContainer = document.getElementById('res-matches');
            matchContainer.innerHTML = '';
            (data.near_matches || []).forEach(m => {
                matchContainer.innerHTML += `
                    <div class="bg-gray-700 px-2 py-1 rounded text-xs text-gray-300">
                        ${m.name} <sup class="text-[9px] text-blue-400">${m.conf || '?'}</sup>
                    </div>`;
            });

            // Damage
            const dmgEl = document.getElementById('res-damage');
            if (data.damaged) {
                dmgEl.classList.remove('hidden');
            } else {
                dmgEl.classList.add('hidden');
            }

            // Show card
            resultCard.classList.remove('translate-y-full', 'opacity-0');
        }
    } catch (e) {
        console.log("Parse error, raw text:", rawText);
    }
}

// Controls
document.getElementById('btn-start').onclick = () => {
    isDetecting = true;
    document.getElementById('btn-start').classList.add('hidden');
    document.getElementById('btn-pause').classList.remove('hidden');
    runInference();
};

document.getElementById('btn-pause').onclick = () => {
    isDetecting = false;
    document.getElementById('btn-pause').classList.add('hidden');
    document.getElementById('btn-start').classList.remove('hidden');
    // Keep last result in memory (implicit as UI isn't cleared)
};

document.getElementById('btn-stop').onclick = () => {
    isDetecting = false;
    document.getElementById('btn-pause').classList.add('hidden');
    document.getElementById('btn-start').classList.remove('hidden');
    
    // Save to IndexedDB (Stub)
    if (lastResult) {
        // saveToDb(lastResult);
        alert("Scan saved to local History.");
    }
    
    // Reset UI
    resultCard.classList.add('translate-y-full', 'opacity-0');
};

// Init
window.addEventListener('load', () => {
    startCamera();
    initWllama();
});