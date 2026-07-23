// ==========================================================================
// CodeLLM UI Script - Frontend Logic & Client-side Simulation
// ==========================================================================

const API_BASE = window.location.origin === "file://" || window.location.origin.includes("null") 
    ? "http://localhost:5000" 
    : window.location.origin;

// State management
let backendConnected = false;
let isGenerating = false;
let isPaused = false;
let stepTriggered = false;
let availableModels = {};

// Cache DOM Elements
const elements = {
    backendBadge: document.getElementById('backend-status-badge'),
    modelSelect: document.getElementById('model-select'),
    modelDetails: document.getElementById('model-details'),
    
    // Sliders
    tempSlider: document.getElementById('temp-slider'),
    tempVal: document.getElementById('temp-val'),
    topkSlider: document.getElementById('topk-slider'),
    topkVal: document.getElementById('topk-val'),
    toppSlider: document.getElementById('topp-slider'),
    toppVal: document.getElementById('topp-val'),
    tokensSlider: document.getElementById('tokens-slider'),
    tokensVal: document.getElementById('tokens-val'),
    repSlider: document.getElementById('rep-slider'),
    repVal: document.getElementById('rep-val'),
    
    // Buttons & Textarea
    btnThemeToggle: document.getElementById('btn-theme-toggle'),
    stepControls: document.getElementById('step-controls'),
    btnPauseResume: document.getElementById('btn-pause-resume'),
    btnStepForward: document.getElementById('btn-step-forward'),
    btnStop: document.getElementById('btn-stop'),
    promptInput: document.getElementById('prompt-input'),
    btnClearPrompt: document.getElementById('btn-clear-prompt'),
    btnGenerate: document.getElementById('btn-generate'),
    btnCopyCode: document.getElementById('btn-copy-code'),
    codeOutput: document.getElementById('code-output'),
    lineNumbers: document.getElementById('line-numbers'),
    
    // Stats
    statTime: document.getElementById('stat-time'),
    statTokens: document.getElementById('stat-tokens'),
    statSpeed: document.getElementById('stat-speed'),
    statDevice: document.getElementById('stat-device'),
    
    // Tabs & Panels
    tabs: document.querySelectorAll('.vis-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),
    attentionCanvas: document.getElementById('attention-canvas'),
    tokenChipsContainer: document.getElementById('token-chips-container'),
    terminalBody: document.getElementById('terminal-body'),
    btnClearLogs: document.getElementById('btn-clear-logs'),
};

// Canvas drawing state
const canvasCtx = elements.attentionCanvas.getContext('2d');

// Templates for Local Client-side Simulation (When backend is offline)
const MOCK_GENERATIONS = {
    palindrome: `def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome, ignoring casing and non-alphanumeric characters.
    """
    # Clean the input string
    clean_chars = [c.lower() for c in s if c.isalnum()]
    
    # Compare with its reverse
    return clean_chars == clean_chars[::-1]

# Example execution:
# print(is_palindrome("A man, a plan, a canal: Panama")) # True`,

    vowels: `def count_vowels(text: str) -> int:
    """
    Counts and returns the number of vowels in a given input string.
    """
    vowels = set("aeiouAEIOU")
    count = 0
    
    for char in text:
        if char in vowels:
            count += 1
            
    return count

# Example execution:
# print(count_vowels("Hello World")) # 3`,

    reverselist: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head: ListNode) -> ListNode:
    """
    Reverses a singly linked list in-place and returns the new head.
    """
    prev = None
    curr = head
    
    while curr is not None:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
        
    return prev`,

    binarysearch: `def binary_search(arr: list, target) -> int:
    """
    Perform a binary search on a sorted array.
    Returns the index of the target if found, else -1.
    """
    low = 0
    high = len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        
        if guess == target:
            return mid
        if guess > target:
            high = mid - 1
        else: low = mid + 1
        
    return -1`,

    fibonacci: `def generate_fibonacci(n: int) -> list:
    """
    Generate the Fibonacci sequence up to n elements.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
        
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
        
    return sequence`
};

const MOCK_TOKENS_DICT = [
    "def", " ", "is", "_", "palindrome", "(", "s", ":", " ", "str", ")", " ->", " bool", ":",
    "\\n", "    ", "\"\"\"", "\\n", "    ", "Check", " if", " a", " string", " is", " a", " palindrome",
    "\\n", "    ", "\"\"\"", "\\n", "    ", "clean", " =", " [", "c", ".", "lower", "()", " for",
    " c", " in", " s", " if", " c", ".", "isalnum", "()]", "\\n", "    ", "return", " clean", " ==",
    " clean", "[", "::", "-1", "]"
];

// ==========================================================================
// Initialization & Event Listeners
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupEventListeners();
    checkBackendStatus();
    setInterval(checkBackendStatus, 15000); // Check status every 15s
    drawPlaceholderAttention();
});

function setupEventListeners() {
    // Slider binds
    bindSlider(elements.tempSlider, elements.tempVal, (v) => parseFloat(v).toFixed(2));
    bindSlider(elements.topkSlider, elements.topkVal);
    bindSlider(elements.toppSlider, elements.toppVal, (v) => parseFloat(v).toFixed(2));
    bindSlider(elements.tokensSlider, elements.tokensVal);
    bindSlider(elements.repSlider, elements.repVal, (v) => parseFloat(v).toFixed(2));

    // Input handlers
    elements.btnClearPrompt.addEventListener('click', () => {
        elements.promptInput.value = '';
        logConsole("Prompt cleared.", "info");
    });

    elements.btnGenerate.addEventListener('click', startGeneration);

    elements.btnCopyCode.addEventListener('click', copyCodeToClipboard);

    // Theme Toggle
    elements.btnThemeToggle.addEventListener('click', toggleTheme);

    // Step-by-Step Generation Controls
    elements.btnPauseResume.addEventListener('click', togglePauseResume);
    elements.btnStepForward.addEventListener('click', triggerStep);
    elements.btnStop.addEventListener('click', stopGeneration);

    elements.btnClearLogs.addEventListener('click', () => {
        elements.terminalBody.innerHTML = '';
        logConsole("Logs console cleared.", "system");
    });

    // Template Tags
    document.querySelectorAll('.template-tag').forEach(tag => {
        tag.addEventListener('click', (e) => {
            const promptText = e.target.getAttribute('data-prompt');
            elements.promptInput.value = promptText;
            logConsole(`Loaded prompt template: "${e.target.innerText}"`, "info");
            
            // Highlight tag momentarily
            e.target.style.transform = "scale(0.95)";
            setTimeout(() => e.target.style.transform = "", 150);
        });
    });

    // Model select change handler
    elements.modelSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        logConsole(`Model target switched to: ${val}`, "info");
        
        if (backendConnected && availableModels[val]) {
            const modelInfo = availableModels[val];
            elements.modelDetails.innerText = modelInfo.exists 
                ? "Status: Available (Loaded)" 
                : "Status: " + modelInfo.details;
        } else {
            if (val === 'gpt_subword') {
                elements.modelDetails.innerText = "Subword BPE: Requires code_bpe.model tokenizer";
            } else {
                elements.modelDetails.innerText = "Char-level: Requires char_vocab.json & checkpoint";
            }
        }
    });

    // Visualizer tabs switching
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            const targetPane = e.target.getAttribute('data-tab');
            elements.tabs.forEach(t => t.classList.remove('active'));
            elements.tabPanes.forEach(p => p.classList.remove('active'));
            
            e.target.classList.add('active');
            document.getElementById(`vis-${targetPane}`).classList.add('active');
        });
    });
}

function bindSlider(slider, valDisplay, formatter = (v) => v) {
    slider.addEventListener('input', (e) => {
        valDisplay.innerText = formatter(e.target.value);
    });
}

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
    }
}

function toggleTheme() {
    const isLight = document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    logConsole(`UI Theme switched to ${isLight ? 'Light' : 'Dark'} Mode`, "system");
    
    // Redraw attention placeholder with correct theme colors
    drawPlaceholderAttention();
}

function togglePauseResume() {
    if (!isGenerating) return;
    isPaused = !isPaused;
    if (isPaused) {
        elements.btnPauseResume.innerHTML = `<i class="fa-solid fa-play"></i>`;
        elements.btnPauseResume.title = "Resume Generation";
        logConsole("Generation paused by user.", "warn");
    } else {
        elements.btnPauseResume.innerHTML = `<i class="fa-solid fa-pause"></i>`;
        elements.btnPauseResume.title = "Pause Generation";
        logConsole("Generation resumed.", "info");
    }
}

function triggerStep() {
    if (!isGenerating || !isPaused) return;
    stepTriggered = true;
    logConsole("Stepped forward 1 token.", "info");
}

function stopGeneration() {
    if (!isGenerating) return;
    isGenerating = false;
    isPaused = false;
    logConsole("Generation aborted by user.", "error");
}

// ==========================================================================
// Status Verification (Heartbeat)
// ==========================================================================

async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        if (!response.ok) throw new Error("Status API returned failure status");
        
        const data = await response.json();
        backendConnected = true;
        availableModels = data.models_available || {};
        
        // Update status badge
        elements.backendBadge.className = "status-badge status-online";
        elements.backendBadge.querySelector('.status-text').innerText = "Connected to PyTorch API";
        
        // Update specific selected model status
        const activeModel = elements.modelSelect.value;
        if (availableModels[activeModel]) {
            const info = availableModels[activeModel];
            elements.modelDetails.innerText = info.exists ? "Ready to run inference" : info.details;
        }
        
        // Update stats
        elements.statDevice.innerText = data.gpu_name ? `GPU (${data.gpu_name})` : (data.cuda_available ? "CUDA" : "CPU");
        
    } catch (error) {
        if (backendConnected) {
            logConsole("Connection to PyTorch backend lost. Switching to Simulated Mode.", "warn");
        }
        backendConnected = false;
        elements.backendBadge.className = "status-badge status-offline";
        elements.backendBadge.querySelector('.status-text').innerText = "Simulated Mode (Local API Offline)";
        elements.modelDetails.innerText = "Simulating: No PyTorch installation required";
        elements.statDevice.innerText = "Browser VM";
    }
}

// ==========================================================================
// Logging utility
// ==========================================================================

function logConsole(message, type = "info") {
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `terminal-line log-${type}`;
    
    let prefix = "[INFO]";
    if (type === "system") prefix = "[SYS]";
    if (type === "warn") prefix = "[WARN]";
    if (type === "error") prefix = "[ERR]";
    if (type === "success") prefix = "[OK]";
    
    line.innerText = `${timestamp} ${prefix} ${message}`;
    elements.terminalBody.appendChild(line);
    elements.terminalBody.scrollTop = elements.terminalBody.scrollHeight;
}

// ==========================================================================
// Code Generation Controller
// ==========================================================================

async function startGeneration() {
    if (isGenerating) return;
    
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
        logConsole("Failed: Prompt input cannot be empty.", "error");
        alert("Please enter a natural language prompt first.");
        return;
    }
    
    isGenerating = true;
    isPaused = false;
    stepTriggered = false;
    elements.btnPauseResume.innerHTML = `<i class="fa-solid fa-pause"></i>`;
    elements.btnPauseResume.title = "Pause Generation";
    elements.stepControls.classList.remove('hidden');

    elements.btnGenerate.classList.add('generating');
    elements.btnGenerate.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;
    elements.codeOutput.innerHTML = '';
    elements.lineNumbers.innerHTML = '<span>1</span>';
    
    logConsole("Initializing generation pipeline...", "system");
    
    if (backendConnected) {
        await runRealGeneration(prompt);
    } else {
        await runSimulatedGeneration(prompt);
    }
    
    isGenerating = false;
    isPaused = false;
    elements.stepControls.classList.add('hidden');
    elements.btnGenerate.classList.remove('generating');
    elements.btnGenerate.innerHTML = `<i class="fa-solid fa-play"></i> Generate Code`;
}

// ==========================================================================
// 1. Real API Inference
// ==========================================================================

async function runRealGeneration(prompt) {
    const params = {
        model: elements.modelSelect.value,
        prompt: prompt,
        temperature: parseFloat(elements.tempSlider.value),
        top_k: parseInt(elements.topkSlider.value),
        top_p: parseFloat(elements.toppSlider.value),
        max_tokens: parseInt(elements.tokensSlider.value),
        repetition_penalty: parseFloat(elements.repSlider.value)
    };

    logConsole(`Executing request on backend using '${params.model}' model...`, "info");
    
    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Backend returned HTTP error");
        }
        
        // Print logs returned by backend
        if (data.logs && Array.isArray(data.logs)) {
            data.logs.forEach(log => {
                if (log.includes("[ERROR]")) logConsole(log.replace("[ERROR] ", ""), "error");
                else if (log.includes("[WARN]")) logConsole(log.replace("[WARN] ", ""), "warn");
                else logConsole(log.replace("[INFO] ", ""), "info");
            });
        }
        
        // Stream the output code to the editor
        await animateTokenStreaming(data.text, data.stats);
        logConsole("Generation successful.", "success");
        
    } catch (err) {
        logConsole(`Inference failed: ${err.message}`, "error");
        // Fallback to simulation if backend failed during generation
        logConsole("Falling back to client-side simulated generation...", "warn");
        await runSimulatedGeneration(prompt);
    }
}

// ==========================================================================
// 2. Simulated Generation (Offline mode)
// ==========================================================================

async function runSimulatedGeneration(prompt) {
    logConsole("[SIMULATOR] Parsing prompt token relevance...", "info");
    
    // Match prompt against mock templates
    let matchedTemplate = 'palindrome';
    const lowerPrompt = prompt.toLowerCase();
    
    if (lowerPrompt.includes('vowel')) matchedTemplate = 'vowels';
    else if (lowerPrompt.includes('reverse') && (lowerPrompt.includes('list') || lowerPrompt.includes('link'))) matchedTemplate = 'reverselist';
    else if (lowerPrompt.includes('binary') || lowerPrompt.includes('search')) matchedTemplate = 'binarysearch';
    else if (lowerPrompt.includes('fibonacci')) matchedTemplate = 'fibonacci';
    else if (lowerPrompt.includes('palindrome') || lowerPrompt.includes('palin')) matchedTemplate = 'palindrome';
    else {
        // Create custom mock code matching prompt
        matchedTemplate = 'custom';
    }
    
    let targetCode = "";
    if (matchedTemplate === 'custom') {
        // Build a dynamic generic function matching prompt intent
        const words = prompt.replace(/[^\w\s]/gi, '').split(/\s+/).filter(w => w.length > 2);
        const name = words.length > 0 ? words.slice(-2).join('_').toLowerCase() : 'process_data';
        targetCode = `def ${name}(data):\n    """\n    Automated response to: "${prompt}"\n    """\n    # TODO: Implement algorithm logic\n    result = []\n    for item in data:\n        # Syntax validation\n        processed = item\n        result.append(processed)\n        \n    return result\n\n# Simulated run completed.`;
    } else {
        targetCode = MOCK_GENERATIONS[matchedTemplate];
    }
    
    // Simulate pipeline latency
    await delay(600);
    logConsole("[SIMULATOR] Tokenizing prompt into sentencepiece IDs...", "info");
    await delay(350);
    logConsole(`[SIMULATOR] Encoding: 21 tokens matched. Found vocabulary references.`, "info");
    await delay(400);
    logConsole("[SIMULATOR] Sampling logits using Attention projection...", "info");
    
    // Stream tokens
    const totalTokens = Math.floor(targetCode.length / 4) + 12;
    const startTime = Date.now();
    
    await animateTokenStreaming(targetCode, {
        time_taken_sec: (targetCode.length * 0.0035).toFixed(3),
        tokens_generated: totalTokens,
        tokens_per_sec: (totalTokens / (targetCode.length * 0.0035)).toFixed(1),
        device: "Browser JS Engine"
    });
    
    logConsole("[SIMULATOR] Met <|end|> token. Decoding final output stream.", "success");
}

// ==========================================================================
// Token-by-Token Typing Animation & Visualizer Updates
// ==========================================================================

async function animateTokenStreaming(codeText, stats) {
    // Generate token boundaries (chunks of characters to simulate tokens)
    const tokens = chunkTextIntoTokens(codeText);
    
    // Clear and build token visualizer chips
    elements.tokenChipsContainer.innerHTML = '';
    
    let currentText = '';
    const startTimestamp = Date.now();
    
    // Setup attention matrix dimensions
    const maxAttnDim = Math.min(tokens.length, 32);
    let attnWeights = generateAttentionMatrix(maxAttnDim);
    
    let totalPauseDurationMs = 0;

    for (let i = 0; i < tokens.length; i++) {
        if (!isGenerating) {
            break;
        }

        // Pause/Step check
        if (isPaused) {
            const pauseStart = Date.now();
            elements.btnStepForward.disabled = false;
            while (isPaused && isGenerating) {
                if (stepTriggered) {
                    stepTriggered = false;
                    break;
                }
                await delay(30);
            }
            elements.btnStepForward.disabled = true;
            totalPauseDurationMs += (Date.now() - pauseStart);
        }

        if (!isGenerating) {
            break;
        }

        currentText += tokens[i];
        
        // Highlight code
        const highlighted = highlightPython(currentText);
        elements.codeOutput.innerHTML = highlighted;
        
        // Update line numbers
        updateLineNumbers(currentText);
        
        // Add token chip to the visualization
        addTokenChip(tokens[i], i);
        
        // Draw dynamically updating attention layers
        drawAttentionGrid(attnWeights, i, maxAttnDim);
        
        // Calculate real-time speed stats
        const elapsedSec = (Date.now() - startTimestamp - totalPauseDurationMs) / 1000;
        elements.statTime.innerText = `${elapsedSec.toFixed(2)}s`;
        elements.statTokens.innerText = `${i + 1} / ${tokens.length}`;
        elements.statSpeed.innerText = `${((i + 1) / Math.max(elapsedSec, 0.01)).toFixed(1)} tok/s`;
        
        // Scroll containers to bottom
        elements.codeOutput.scrollIntoView({ block: 'end' });
        elements.tokenChipsContainer.scrollTop = elements.tokenChipsContainer.scrollHeight;
        
        // Dynamic speed control based on temperature setting
        const temp = parseFloat(elements.tempSlider.value);
        const delayMs = Math.max(15, Math.min(100, (temp * 30)));
        await delay(delayMs);
    }
    
    // Set final strict stats
    elements.statTime.innerText = `${stats.time_taken_sec}s`;
    elements.statTokens.innerText = stats.tokens_generated;
    elements.statSpeed.innerText = `${stats.tokens_per_sec} tok/s`;
}

// Chunking algorithm to simulate code splitting
function chunkTextIntoTokens(text) {
    const tokens = [];
    let i = 0;
    while (i < text.length) {
        // Pick variable chunk sizes to simulate subword tokens
        let chunkSize = Math.floor(Math.random() * 3) + 2; 
        if (text[i] === ' ' || text[i] === '\n') {
            chunkSize = 1;
        } else if (text.substr(i, 3) === 'def') {
            chunkSize = 3;
        }
        tokens.push(text.substring(i, i + chunkSize));
        i += chunkSize;
    }
    return tokens;
}

function updateLineNumbers(text) {
    const lines = text.split('\n').length;
    let linesHtml = '';
    for (let i = 1; i <= lines; i++) {
        linesHtml += `<span>${i}</span>`;
    }
    elements.lineNumbers.innerHTML = linesHtml;
}

function addTokenChip(tokenVal, index) {
    if (elements.tokenChipsContainer.querySelector('.empty-chips')) {
        elements.tokenChipsContainer.innerHTML = '';
    }
    
    // Escape whitespace characters for visualization
    const displayVal = tokenVal
        .replace(/\n/g, '\\n')
        .replace(/\t/g, '\\t')
        .replace(/ /g, '•');
        
    const chip = document.createElement('span');
    chip.className = 'token-chip';
    
    // Assign a deterministic background color based on token value
    let hash = 0;
    for (let i = 0; i < tokenVal.length; i++) {
        hash = tokenVal.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    chip.style.backgroundColor = `hsla(${hue}, 70%, 25%, 0.4)`;
    chip.style.borderColor = `hsla(${hue}, 80%, 40%, 0.6)`;
    chip.innerText = `${displayVal} [${(1000 + (index % 8999))}]`;
    
    elements.tokenChipsContainer.appendChild(chip);
}

// ==========================================================================
// Attention Matrix Visualizer (Canvas Rendering)
// ==========================================================================

function generateAttentionMatrix(dim) {
    const matrix = [];
    for (let i = 0; i < dim; i++) {
        matrix[i] = [];
        for (let j = 0; j < dim; j++) {
            // Self-attention is causal (lower-triangular mask)
            if (j > i) {
                matrix[i][j] = 0;
            } else {
                // Diagonal self-attends highly, with some random key weightings
                if (i === j) {
                    matrix[i][j] = 0.6 + Math.random() * 0.4;
                } else {
                    matrix[i][j] = Math.random() * (j / i) * 0.5;
                }
            }
        }
    }
    return matrix;
}

function drawPlaceholderAttention() {
    const isLight = document.body.classList.contains('light-theme');
    canvasCtx.fillStyle = isLight ? '#f8fafc' : '#080c14';
    canvasCtx.fillRect(0, 0, elements.attentionCanvas.width, elements.attentionCanvas.height);
    
    canvasCtx.strokeStyle = isLight ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)';
    canvasCtx.lineWidth = 1;
    for (let i = 0; i < elements.attentionCanvas.width; i += 20) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(i, 0);
        canvasCtx.lineTo(i, elements.attentionCanvas.height);
        canvasCtx.stroke();
        
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, i);
        canvasCtx.lineTo(elements.attentionCanvas.width, i);
        canvasCtx.stroke();
    }
    
    canvasCtx.fillStyle = isLight ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.2)';
    canvasCtx.font = '11px Outfit';
    canvasCtx.textAlign = 'center';
    canvasCtx.fillText("Waiting for generation...", elements.attentionCanvas.width / 2, elements.attentionCanvas.height / 2);
}

function drawAttentionGrid(matrix, currentTokenIndex, maxDim) {
    const canvas = elements.attentionCanvas;
    const width = canvas.width;
    const height = canvas.height;
    const cellW = width / maxDim;
    const cellH = height / maxDim;
    
    const isLight = document.body.classList.contains('light-theme');
    canvasCtx.fillStyle = isLight ? '#f8fafc' : '#05070c';
    canvasCtx.fillRect(0, 0, width, height);
    
    for (let i = 0; i < maxDim; i++) {
        for (let j = 0; j < maxDim; j++) {
            if (j > i) continue; // Masked causal attention
            
            let weight = matrix[i][j];
            if (i > currentTokenIndex) {
                // Not generated yet
                continue;
            }
            
            // Highlight the row currently being sampled
            if (i === currentTokenIndex) {
                canvasCtx.fillStyle = `rgba(6, 182, 212, ${weight * 0.95})`;
            } else {
                // Older layers: Indigo transparency
                canvasCtx.fillStyle = `rgba(99, 102, 241, ${weight * 0.7})`;
            }
            
            canvasCtx.fillRect(j * cellW, i * cellH, cellW - 1, cellH - 1);
        }
    }
}

// ==========================================================================
// Regex-based Code Highlighter (Python Syntax Engine)
// ==========================================================================

function highlightPython(code) {
    // Simple sanitization
    let escaped = code
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    // List of regex patterns for basic python components
    const patterns = [
        { regex: /("(?:\\"|[^"])*"|'(?:\\'|[^'])*')/g, class: 'hl-str' }, // Strings
        { regex: /(#.*)/g, class: 'hl-comment' }, // Comments
        { regex: /\b(def|class|return|if|else|elif|while|for|in|import|from|as|try|except|raise|with|lambda|and|or|not|is|pass|assert|global|nonlocal|break|continue)\b/g, class: 'hl-keyword' }, // Keywords
        { regex: /\b(self|cls)\b/g, class: 'hl-def' }, // Native objects
        { regex: /\b(\w+)(?=\()/g, class: 'hl-name' }, // Function names
        { regex: /\b(\d+)\b/g, class: 'hl-num' }, // Numbers
        { regex: /([+\-*/%&|^~<>!=]=?)/g, class: 'hl-op' } // Operators
    ];
    
    // We mask comments and strings to protect them from keyword highlighting
    const maskedText = [];
    let tempCode = escaped;
    
    // Mask Strings & Comments
    let maskCounter = 0;
    const stringCommentRegex = /("(?:\\"|[^"])*"|'(?:\\'|[^'])*'|#.*)/g;
    
    tempCode = tempCode.replace(stringCommentRegex, (match) => {
        const mask = `___MASK_TOKEN_${maskCounter}___`;
        const isComment = match.startsWith('#');
        maskedText.push({
            mask: mask,
            original: match,
            class: isComment ? 'hl-comment' : 'hl-str'
        });
        maskCounter++;
        return mask;
    });

    // Apply regex highlighting on the remaining unmasked structure
    patterns.forEach(p => {
        if (p.class !== 'hl-str' && p.class !== 'hl-comment') {
            tempCode = tempCode.replace(p.regex, `<span class="${p.class}">$1</span>`);
        }
    });

    // Unmask strings and comments back, wrapping them in their highlighted span
    maskedText.forEach(m => {
        tempCode = tempCode.replace(m.mask, `<span class="${m.class}">${m.original}</span>`);
    });

    return tempCode;
}

// ==========================================================================
// Copy Code Controller
// ==========================================================================

function copyCodeToClipboard() {
    const textToCopy = elements.codeOutput.innerText;
    if (!textToCopy || textToCopy.startsWith("# Output will be")) {
        return;
    }
    
    navigator.clipboard.writeText(textToCopy).then(() => {
        elements.btnCopyCode.innerHTML = `<i class="fa-solid fa-check" style="color: var(--accent-green)"></i> Copied!`;
        logConsole("Copied generated code to clipboard.", "info");
        setTimeout(() => {
            elements.btnCopyCode.innerHTML = `<i class="fa-regular fa-copy"></i> Copy`;
        }, 2000);
    }).catch(err => {
        logConsole(`Copy failed: ${err.message}`, "error");
    });
}

// Helper utilities
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
