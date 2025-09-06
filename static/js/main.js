// Main JavaScript for Coconut Leaf Disease Predictor

document.addEventListener('DOMContentLoaded', function() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in-up');
        }, index * 100);
    });

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        }, 5000);
    });

    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading state to form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
                
                // Re-enable after 30 seconds as failsafe
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 30000);
            }
        });
    });

    // File input validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Check file size (16MB limit)
                const maxSize = 16 * 1024 * 1024; // 16MB in bytes
                if (file.size > maxSize) {
                    alert('File size is too large. Please select a file smaller than 16MB.');
                    this.value = '';
                    return;
                }

                // Check file type
                const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif'];
                if (!allowedTypes.includes(file.type)) {
                    alert('Invalid file type. Please select a PNG, JPG, JPEG, or GIF image.');
                    this.value = '';
                    return;
                }
            }
        });
    });

    // Tooltip initialization (if using Bootstrap tooltips)
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });

    // Progress bar animation
    const progressBars = document.querySelectorAll('.progress-bar');
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const progressObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target;
                const width = progressBar.style.width;
                progressBar.style.width = '0%';
                setTimeout(() => {
                    progressBar.style.width = width;
                }, 100);
            }
        });
    }, observerOptions);

    progressBars.forEach(bar => {
        progressObserver.observe(bar);
    });
});

// Image preview function (global scope for inline usage)
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        const preview = document.getElementById('preview');
        const previewContainer = document.getElementById('imagePreview');
        
        reader.onload = function(e) {
            preview.src = e.target.result;
            previewContainer.style.display = 'block';
            preview.classList.add('fade-in-up');
        };
        
        reader.onerror = function() {
            alert('Error reading file. Please try again.');
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

// Confidence level styling
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'bg-success';
    if (confidence >= 0.6) return 'bg-warning';
    return 'bg-danger';
}

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Dashboard chart initialization (if Chart.js is available)
function initializeDashboardChart(healthyCount, diseasedCount) {
    const ctx = document.getElementById('healthChart');
    if (ctx && typeof Chart !== 'undefined') {
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Healthy', 'Diseased'],
                datasets: [{
                    data: [healthyCount, diseasedCount],
                    backgroundColor: ['#28a745', '#ffc107'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Health Distribution',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 1000
                }
            }
        });
    }
}

// Utility function to show notifications
function showNotification(message, type = 'info', duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after duration
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, duration);
}

// Handle offline/online status
window.addEventListener('online', function() {
    showNotification('Connection restored!', 'success');
});

window.addEventListener('offline', function() {
    showNotification('Connection lost. Some features may not work properly.', 'warning');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search or navigate to predict page
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const predictLink = document.querySelector('a[href*="predict"]');
        if (predictLink) {
            predictLink.click();
        }
    }
    
    // Escape to close modals or go back
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        if (modals.length === 0 && window.history.length > 1) {
            window.history.back();
        }
    }
});
