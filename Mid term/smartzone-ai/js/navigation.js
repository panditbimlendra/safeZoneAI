// Handle navigation between pages
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-links a');
    const contentFrame = document.getElementById('contentFrame');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = link.getAttribute('href');
            
            // Update active state
            navLinks.forEach(a => a.classList.remove('active'));
            link.classList.add('active');
            
            // Load content
            contentFrame.src = targetPage;
            
            // Update browser history
            history.pushState(null, '', targetPage);
        });
    });
    
    // Handle browser back/forward
    window.addEventListener('popstate', function() {
        const path = window.location.pathname.split('/').pop() || 'home.html';
        contentFrame.src = path;
    });
});