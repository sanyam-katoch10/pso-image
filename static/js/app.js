document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initSegmentForm();
});


function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('imageInput');
    if (!dropzone || !fileInput) return;

    const previewContainer = dropzone.querySelector('.dropzone-preview');
    const previewImg = previewContainer ? previewContainer.querySelector('img') : null;
    const fileInfo = dropzone.querySelector('.dropzone-file-info');
    const uploadText = dropzone.querySelector('.dropzone-text');
    const uploadHint = dropzone.querySelector('.dropzone-hint');
    const MAX_SIZE = 16 * 1024 * 1024;
    const ALLOWED = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff', 'image/webp'];

    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('.dropzone-file-info')) return;
        fileInput.click();
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!ALLOWED.includes(file.type)) {
            showError('Invalid file type. Please upload PNG, JPG, BMP, TIFF, or WebP.');
            return;
        }
        if (file.size > MAX_SIZE) {
            showError('File too large. Maximum size is 16 MB.');
            return;
        }

        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        dropzone.classList.add('has-file');

        if (previewImg && previewContainer) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewContainer.classList.add('active');
            };
            reader.readAsDataURL(file);
        }

        if (fileInfo) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            fileInfo.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                ${file.name} (${sizeMB} MB)
            `;
            fileInfo.classList.add('active');
        }

        if (uploadText) uploadText.style.display = 'none';
        if (uploadHint) uploadHint.style.display = 'none';

        const submitBtn = document.getElementById('segmentBtn');
        if (submitBtn) submitBtn.disabled = false;
    }
}


function initSegmentForm() {
    const form = document.getElementById('segmentForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const fileInput = document.getElementById('imageInput');

        if (!fileInput || !fileInput.files.length) {
            showError('Please select an image first.');
            return;
        }

        const submitBtn = document.getElementById('segmentBtn');
        if (submitBtn) submitBtn.disabled = true;

        showProgress();

        try {
            const response = await fetch('/segment', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Segmentation failed');
            }

            const data = await response.json();
            listenProgress(data.job_id);

        } catch (err) {
            hideProgress();
            showError(err.message);
            if (submitBtn) submitBtn.disabled = false;
        }
    });
}

function listenProgress(jobId) {
    const source = new EventSource(`/progress/${jobId}`);

    source.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === 'complete') {
            source.close();
            updateProgress(100, 'Segmentation complete!');
            setTimeout(() => {
                window.location.href = `/result/${jobId}`;
            }, 600);
            return;
        }

        if (data.status === 'error') {
            source.close();
            hideProgress();
            showError(data.error || 'An error occurred during segmentation.');
            return;
        }

        if (data.progress !== undefined) {
            const costText = data.cost ? ` | MSE: ${data.cost}` : '';
            updateProgress(data.progress, `Optimizing swarm particles...${costText}`);
        }
    };

    source.onerror = () => {
        source.close();
    };
}


function showProgress() {
    const overlay = document.getElementById('progressOverlay');
    if (overlay) overlay.classList.add('active');
    updateProgress(0, 'Initializing PSO engine...');
}

function hideProgress() {
    const overlay = document.getElementById('progressOverlay');
    if (overlay) overlay.classList.remove('active');
}

function updateProgress(percent, statusText) {
    const fill = document.getElementById('progressBarFill');
    const percentEl = document.getElementById('progressPercent');
    const statusEl = document.getElementById('progressStatus');
    const spinnerFill = document.getElementById('spinnerFill');

    if (fill) fill.style.width = `${percent}%`;
    if (percentEl) percentEl.textContent = `${Math.round(percent)}%`;
    if (statusEl) statusEl.textContent = statusText || '';

    if (spinnerFill) {
        const circumference = 201;
        const offset = circumference - (percent / 100) * circumference;
        spinnerFill.style.strokeDashoffset = offset;
    }
}


function showError(message) {
    let toast = document.getElementById('errorToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'errorToast';
        toast.style.cssText = `
            position: fixed; bottom: 24px; right: 24px; z-index: 2000;
            background: rgba(239, 68, 68, 0.95); color: white;
            padding: 14px 24px; border-radius: 12px;
            font-size: 14px; font-weight: 500; max-width: 400px;
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
            backdrop-filter: blur(8px);
            animation: fadeInUp 0.3s ease-out;
        `;
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.display = 'block';

    setTimeout(() => {
        if (toast) toast.style.display = 'none';
    }, 5000);
}
