// OCR Engine Web UI

const API_URL = '/api/v1';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const overlayCanvas = document.getElementById('overlayCanvas');
const resultSection = document.getElementById('resultSection');
const resultText = document.getElementById('resultText');
const resultJson = document.getElementById('resultJson');
const resultTable = document.getElementById('resultTable');
const resultMeta = document.getElementById('resultMeta');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorText = document.getElementById('errorText');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const detectTables = document.getElementById('detectTables');
const spellCheck = document.getElementById('spellCheck');
const tableTab = document.getElementById('tableTab');

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
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Desteklenmeyen dosya tipi. Lutfen PNG, JPG, BMP, TIFF veya WEBP yukleyin.');
        return;
    }

    currentFile = file;
    
    // Show preview
    showPreview(file);
    
    // Perform OCR
    await performOCR(file);
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        
        // Wait for image to load
        previewImage.onload = () => {
            // Resize canvas to match image
            overlayCanvas.width = previewImage.width;
            overlayCanvas.height = previewImage.height;
        };
    };
    reader.readAsDataURL(file);
}

async function performOCR(file) {
    showLoading(true);
    hideError();
    resultSection.style.display = 'none';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('detect_tables', detectTables.checked);
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
        drawBoxes(result.blocks);
        
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

function displayResult(result) {
    // Text tab
    resultText.value = result.text;
    
    // JSON tab
    resultJson.textContent = JSON.stringify(result, null, 2);
    
    // Table tab
    if (result.tables && result.tables.length > 0) {
        tableTab.style.display = 'block';
        renderTable(result.tables[0]);
    } else {
        tableTab.style.display = 'none';
    }
    
    // Meta info
    resultMeta.innerHTML = `
        <strong>Isleme suresi:</strong> ${result.processing_time.toFixed(3)}s | 
        <strong>Gorsel boyutu:</strong> ${result.image_size.width}x${result.image_size.height} | 
        <strong>Tespit edilen blok:</strong> ${result.blocks.length}
    `;
    
    resultSection.style.display = 'block';
}

function renderTable(table) {
    let html = '<table><thead><tr>';
    
    // Header (first row)
    for (let c = 0; c < table.cols; c++) {
        const cell = table.cells.find(cell => cell.row === 0 && cell.col === c);
        html += `<th>${cell ? cell.text : ''}</th>`;
    }
    html += '</tr></thead><tbody>';
    
    // Body rows
    for (let r = 1; r < table.rows; r++) {
        html += '<tr>';
        for (let c = 0; c < table.cols; c++) {
            const cell = table.cells.find(cell => cell.row === r && cell.col === c);
            html += `<td>${cell ? cell.text : ''}</td>`;
        }
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    resultTable.innerHTML = html;
}

function drawBoxes(blocks) {
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    
    // Scale factor
    const scaleX = previewImage.width / currentResult.image_size.width;
    const scaleY = previewImage.height / currentResult.image_size.height;
    
    blocks.forEach((block, index) => {
        const box = block.bounding_box;
        
        // Draw rectangle
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            box.x1 * scaleX,
            box.y1 * scaleY,
            (box.x2 - box.x1) * scaleX,
            (box.y2 - box.y1) * scaleY
        );
        
        // Draw polygon if available
        if (box.polygon && box.polygon.length === 4) {
            ctx.beginPath();
            ctx.moveTo(box.polygon[0][0] * scaleX, box.polygon[0][1] * scaleY);
            for (let i = 1; i < box.polygon.length; i++) {
                ctx.lineTo(box.polygon[i][0] * scaleX, box.polygon[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.strokeStyle = 'rgba(37, 99, 235, 0.8)';
            ctx.stroke();
        }
    });
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
    
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
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
