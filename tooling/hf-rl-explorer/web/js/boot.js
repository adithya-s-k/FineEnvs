// Runs before the page paints (loaded blocking in <head>): apply the saved theme so there's no flash, and mark that
// scripts run, so the page's server-written text (for crawlers, app/seo.py) stays hidden while the app starts.
document.documentElement.classList.add("js");
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* private mode */ }
