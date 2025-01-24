// app/static/js/scripts.js
document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const fileInput = document.querySelector('#file');
    const progressBar = document.querySelector('.progress-bar > div');
    const statusText = document.getElementById('status');

    form.addEventListener('submit', (event) => {
        if (!fileInput.files.length) {
            alert('Please select a file.');
            event.preventDefault();
            return;
        }

        // Disable the button and show status
        document.querySelector('button[type="submit"]').disabled = true;
        statusText.innerText = 'Uploading...';

        // Simulate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += 20;
            progressBar.style.width = `${progress}%`;

            if (progress >= 100) {
                clearInterval(interval);
                statusText.innerText = 'Processing...';
            }
        }, 500);
    });
});
