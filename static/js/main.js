// main.js
function openModal() {
    document.getElementById("creator-modal").style.display = "block";
  }
  
  function closeModal() {
    document.getElementById("creator-modal").style.display = "none";
  }
  
  window.onclick = function(event) {
    const modal = document.getElementById("creator-modal");
    if (event.target == modal) {
      modal.style.display = "none";
    }
  };
  
  // Close modal on ESC
  window.addEventListener("keydown", function(e) {
    if (e.key === "Escape") {
      closeModal();
    }
  });
  
  // Fade-in on scroll
  const faders = document.querySelectorAll('.fade-in');
  
  const appearOptions = {
    threshold: 0.3,
    rootMargin: "0px 0px -50px 0px"
  };
  
  const appearOnScroll = new IntersectionObserver(function(entries, observer) {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("appear");
      observer.unobserve(entry.target);
    });
  }, appearOptions);
  
  faders.forEach(fader => {
    appearOnScroll.observe(fader);
  });
  
  // Profile picture upload
document.addEventListener('DOMContentLoaded', function() {
  const profilePicUpload = document.getElementById('profile-pic-upload');
  if (profilePicUpload) {
      profilePicUpload.addEventListener('change', function() {
          const form = document.getElementById('profile-pic-form');
          const formData = new FormData(form);
          
          fetch('/upload_profile_pic', {
              method: 'POST',
              body: formData
          })
          .then(response => response.json())
          .then(data => {
              if (data.success) {
                  window.location.reload();
              } else {
                  alert(data.error || 'Failed to upload profile picture');
              }
          })
          .catch(error => {
              console.error('Error:', error);
              alert('An error occurred while uploading the picture');
          });
      });
  }
});