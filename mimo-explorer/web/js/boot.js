// Runs before the page paints (loaded blocking in <head>): apply the saved theme so there's no flash.
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* private mode */ }
