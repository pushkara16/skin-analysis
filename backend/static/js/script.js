document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.querySelector('.upload-area');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('uploadForm');
  
    // Handle file selection
    fileInput.addEventListener('change', function() {
      if (this.files && this.files[0]) {
        const file = this.files[0];
        fileName.textContent = file.name;
        
        // Validate file type
        const fileType = file.type.split('/')[0];
        if (fileType !== 'video') {
          fileName.innerHTML = '<span style="color: #d9534f;">Please select a video file.</span>';
          this.value = ''; // Clear the input
        }
      }
    });
  
    // Handle drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      uploadArea.classList.add('active');
    });
  
    uploadArea.addEventListener('dragleave', function() {
      uploadArea.classList.remove('active');
    });
  
    uploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      uploadArea.classList.remove('active');
      
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        
        // Trigger change event manually
        const event = new Event('change');
        fileInput.dispatchEvent(event);
      }
    });
  
    // Make the whole upload area clickable
    uploadArea.addEventListener('click', function() {
      fileInput.click();
    });
  
    // Show loading state when form is submitted
    uploadForm.addEventListener('submit', function() {
      const submitButton = document.querySelector('.btn-upload');
      submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
      submitButton.disabled = true;
    });
  
    // Close alert messages
    const alertCloseButtons = document.querySelectorAll('.alert .btn-close');
    alertCloseButtons.forEach(button => {
      button.addEventListener('click', function() {
        this.parentElement.classList.add('d-none');
      });
    });
  });
  