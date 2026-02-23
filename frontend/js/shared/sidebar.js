document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.querySelector(".mobile-nav-toggle");
    const sidebar = document.querySelector(".input-column");
    const overlay = document.querySelector(".mobile-overlay");

    if (!toggleBtn || !sidebar || !overlay) return;

    function toggleSidebar() {
        sidebar.classList.toggle("open");
        overlay.classList.toggle("show");
        toggleBtn.classList.toggle("hidden");
    }

    function closeSidebar() {
        sidebar.classList.remove("open");
        overlay.classList.remove("show");
        toggleBtn.classList.remove("hidden");
    }

    toggleBtn.addEventListener("click", toggleSidebar);
    overlay.addEventListener("click", closeSidebar);

    // Close sidebar on link click (useful for landing page)
    const links = sidebar.querySelectorAll("a");
    links.forEach(link => {
        link.addEventListener("click", closeSidebar);
    });
});