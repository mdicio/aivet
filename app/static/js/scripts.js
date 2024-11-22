// Add JS functionality if needed (like form validation or feedback)
document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function () {
        document.getElementById('uploadBtn').disabled = true;
        document.getElementById('status').innerText = "Processing...";
    });
});
