// OCR Engine Web UI

const API_URL = '/api/v1';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const resultSection = document.getElementById('resultSection');
const resultText = document.getElementById('resultText');
const resultJson = document.getElementById('resultJson');
const resultMeta = document.getElementById('resultMeta');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorText = document.getElementById('errorText');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const spellCheck = document.getElementById('spellCheck');

// State
let currentFile = null;
let currentResult = null;

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
clearBtn.addEventListener('click', clearAll);
copyBtn.addEventListener('click', copyText);
downloadBtn.addEventListener('click', downloadResult);

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Drag & Drop handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// File processing
async function processFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Desteklenmeyen dosya tipi. Lutfen PNG, JPG, BMP, TIFF veya WEBP yukleyin.');
        return;
    }

    currentFile = file;
    previewSection.style.display = 'none'; // cevap gelene kadar gizle
    await performOCR(file);
}

async function performOCR(file) {
    showLoading(true);
    hideError();
    resultSection.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('spell_check', spellCheck.checked);
        
        const response = await fetch(`${API_URL}/ocr`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'OCR islemi basarisiz');
        }
        
        const result = await response.json();
        currentResult = result;
        
        displayResult(result);
        
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

function displayResult(result) {
    // Sunucudan gelen bounding box'lı görseli direkt göster
    if (result.visualized_image) {
        previewImage.src = result.visualized_image;
    }
    previewSection.style.display = 'block';

    // Text tab
    resultText.value = result.text;
    
    // JSON tab — visualized_image çok uzun, özetini göster
    const summary = {...result, visualized_image: result.visualized_image ? '[base64 PNG]' : null};
    resultJson.textContent = JSON.stringify(summary, null, 2);
    
    // Meta info
    resultMeta.innerHTML = `
        <strong>Isleme suresi:</strong> ${result.processing_time.toFixed(3)}s | 
        <strong>Gorsel boyutu:</strong> ${result.image_size.width}x${result.image_size.height} | 
        <strong>Tespit edilen blok:</strong> ${result.blocks.length}
    `;
    
    resultSection.style.display = 'block';
}
    
function switchTab(tabId) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}Content`);
    });
}

function clearAll() {
    currentFile = null;
    currentResult = null;
    fileInput.value = '';
    previewImage.src = '';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    hideError();
}

function copyText() {
    const text = resultText.value;
    navigator.clipboard.writeText(text).then(() => {
        copyBtn.textContent = 'Kopyalandi!';
        setTimeout(() => {
            copyBtn.textContent = 'Kopyala';
        }, 2000);
    });
}

function downloadResult() {
    if (!currentResult) return;
    
    const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ocr_result.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

function showError(message) {
    errorText.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

// Health check on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('OCR Engine Status:', data);
    } catch (err) {
        console.warn('Health check failed:', err);
    }
}

// Initialize
checkHealth();