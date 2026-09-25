/**
 * Mermaid Zoom Integration - Version Optimisée
 * Système léger pour beaucoup de diagrammes Mermaid
 */

console.log("🚀 Mermaid Zoom Script v16.0 loaded - OPTIMIZED for many diagrams");

// Cache pour les images générées
const imageCache = new Map();

// Fonction pour appliquer les styles Mermaid au SVG selon le thème
function applyMermaidStylesToSvg(svgElement) {
    try {
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";

        // Couleurs selon le thème
        const colors = isDark ? {
            nodeFill: '#1a1a1a',
            nodeStroke: '#ffffff',
            clusterFill: '#2a2a2a',
            clusterStroke: '#ffffff',
            pathStroke: '#ffffff',
            textColor: '#ffffff'
        } : {
            nodeFill: '#ffffff',
            nodeStroke: '#333333',
            clusterFill: '#f9f9f9',
            clusterStroke: '#333333',
            pathStroke: '#333333',
            textColor: '#333333'
        };

        // Appliquer border-radius et couleurs
        const rects = svgElement.querySelectorAll('rect:not(.flowchart-link), .node rect, .nodeLabel rect');
        rects.forEach(rect => {
            rect.setAttribute('rx', '8');
            rect.setAttribute('ry', '8');
            rect.setAttribute('fill', colors.nodeFill);
            rect.setAttribute('stroke', colors.nodeStroke);
        });

        const clusterRects = svgElement.querySelectorAll('.cluster rect');
        clusterRects.forEach(rect => {
            rect.setAttribute('rx', '8');
            rect.setAttribute('ry', '8');
            rect.setAttribute('fill', colors.clusterFill);
            rect.setAttribute('stroke', colors.clusterStroke);
        });

        const paths = svgElement.querySelectorAll('.edgePath');
        paths.forEach(path => {
            path.setAttribute('stroke', colors.pathStroke);
        });

        const textElements = svgElement.querySelectorAll('text, .nodeLabel text, .edgeLabel text');
        textElements.forEach(text => {
            text.setAttribute('fill', colors.textColor);
        });

    } catch (error) {
        console.error('❌ Error applying styles to SVG:', error);
    }
}

// Fonction pour générer une clé de cache unique
function getCacheKey(svgElement, theme) {
    const svgContent = svgElement.outerHTML;
    const hash = svgContent.length + svgContent.slice(0, 100);
    return `${hash}-${theme}`;
}

// Fonction pour convertir SVG en image (avec cache)
function convertSvgToImageCached(svgElement, wrapper, theme) {
    const cacheKey = getCacheKey(svgElement, theme);

    // Vérifier le cache
    if (imageCache.has(cacheKey)) {
        console.log(`📦 Using cached image for theme: ${theme}`);
        return imageCache.get(cacheKey);
    }

    try {
        const wrapperRect = wrapper.getBoundingClientRect();
        const wrapperWidth = Math.round(wrapperRect.width);
        const wrapperHeight = Math.round(wrapperRect.height);

        // Cloner le SVG
        const clonedSvg = svgElement.cloneNode(true);
        applyMermaidStylesToSvg(clonedSvg);

        // Créer une image 2x plus grande pour le zoom
        const zoomFactor = 2;
        const imageWidth = wrapperWidth * zoomFactor;
        const imageHeight = wrapperHeight * zoomFactor;

        clonedSvg.setAttribute('width', imageWidth);
        clonedSvg.setAttribute('height', imageHeight);

        // Créer l'URL de l'image
        const svgData = new XMLSerializer().serializeToString(clonedSvg);
        const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
        const svgUrl = URL.createObjectURL(svgBlob);

        // Créer l'élément image
        const imgElement = document.createElement('img');
        imgElement.src = svgUrl;
        imgElement.style.width = `${wrapperWidth}px`;
        imgElement.style.height = `${wrapperHeight}px`;
        imgElement.style.display = 'block';
        imgElement.setAttribute('data-zoomable', '1');
        imgElement.classList.add('mermaid-zoom-image');

        const result = {
            imgElement,
            svgUrl,
            dimensions: { wrapperWidth, wrapperHeight, imageWidth, imageHeight }
        };

        // Mettre en cache
        imageCache.set(cacheKey, result);
        console.log(`💾 Cached image for theme: ${theme}`);

        return result;

    } catch (error) {
        console.error("❌ Error converting SVG to image:", error);
        return null;
    }
}

// Fonction pour initialiser le zoom sur un diagramme Mermaid
function initMermaidZoom(mermaidEl, index) {
    // Vérifier si déjà wrappé
    if (mermaidEl.parentElement?.classList.contains("mermaid-zoom-wrapper")) {
        return;
    }

    console.log(`📦 Setting up zoom for Mermaid ${index}`);

    // Créer le wrapper
    const wrapper = document.createElement("div");
    wrapper.className = "mermaid-zoom-wrapper";
    wrapper.setAttribute("data-zoomable", "1");
    wrapper.style.display = "block";
    wrapper.style.width = "100%";
    wrapper.style.maxWidth = "100%";

    // Insérer le wrapper avant l'élément Mermaid
    mermaidEl.parentNode.insertBefore(wrapper, mermaidEl);
    wrapper.appendChild(mermaidEl);

    // Ajouter l'événement de clic (conversion à la demande)
    wrapper.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();

        const svgElement = mermaidEl.querySelector("svg");
        if (!svgElement) return;

        const currentTheme = document.documentElement.getAttribute("data-theme");
        console.log(`🖱️ Mermaid ${index} clicked, converting for zoom`);

        // Convertir l'image (avec cache)
        const result = convertSvgToImageCached(svgElement, wrapper, currentTheme);
        if (!result) return;

        // Remplacer le contenu par l'image
        wrapper.innerHTML = '';
        wrapper.appendChild(result.imgElement);

        // Initialiser medium-zoom
        const isDark = currentTheme === "dark";
        const background = isDark ? "rgba(0,0,0,.9)" : "rgba(0,0,0,.85)";

        const zoomInstance = window.mediumZoom(result.imgElement, {
            background,
            margin: 24,
            scrollOffset: 0,
        });

        console.log(`🎉 Zoom initialized for Mermaid ${index}`);

        // Forcer les z-index élevés
        const forceZIndex = () => {
            const overlay = document.querySelector('.medium-zoom-overlay');
            const zoomedImage = document.querySelector('.medium-zoom-image--opened');

            if (overlay) {
                overlay.style.zIndex = '9999999';
                overlay.style.setProperty('z-index', '9999999', 'important');
            }

            if (zoomedImage) {
                zoomedImage.style.zIndex = '10000000';
                zoomedImage.style.setProperty('z-index', '10000000', 'important');
                zoomedImage.style.filter = 'none';
                zoomedImage.style.setProperty('filter', 'none', 'important');
            }
        };

        // Écouter les événements de zoom
        result.imgElement.addEventListener('zoom:open', forceZIndex);
        result.imgElement.addEventListener('zoom:opened', forceZIndex);

        // Observer pour forcer les styles
        const observer = new MutationObserver(forceZIndex);
        observer.observe(document.body, { childList: true, subtree: true });

        // Nettoyer l'observer après 5 secondes
        setTimeout(() => observer.disconnect(), 5000);
    });
}

// Fonction principale optimisée
function setupMermaidZoomOptimized() {
    const mermaidElements = document.querySelectorAll(".mermaid");
    console.log(`🔍 Found ${mermaidElements.length} Mermaid elements`);

    mermaidElements.forEach((mermaidEl, index) => {
        initMermaidZoom(mermaidEl, index);
    });
}

// Observer global optimisé (un seul observer pour tout)
function setupGlobalObserver() {
    const observer = new MutationObserver((mutations) => {
        let shouldUpdate = false;

        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                // Vérifier si de nouveaux diagrammes Mermaid ont été ajoutés
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) { // Element node
                        if (node.classList?.contains('mermaid') ||
                            node.querySelector?.('.mermaid')) {
                            shouldUpdate = true;
                        }
                    }
                });
            }
        });

        if (shouldUpdate) {
            console.log(`🔄 New Mermaid diagrams detected, updating...`);
            setTimeout(setupMermaidZoomOptimized, 100);
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    console.log(`👁️ Global observer started`);
}

// Initialisation optimisée
function initMermaidZoomOptimized() {
    console.log("🎯 Initializing optimized Mermaid zoom");

    // Attendre que mediumZoom soit disponible
    const checkMediumZoom = () => {
        if (window.mediumZoom) {
            console.log("✅ mediumZoom found, initializing optimized Mermaid zoom");
            setupMermaidZoomOptimized();
            setupGlobalObserver();
        } else {
            console.log("⏳ Waiting for mediumZoom...");
            setTimeout(checkMediumZoom, 100);
        }
    };

    checkMediumZoom();
}

// Auto-initialisation
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMermaidZoomOptimized);
} else {
    initMermaidZoomOptimized();
}

// Aussi après le chargement complet
window.addEventListener("load", () => {
    setTimeout(initMermaidZoomOptimized, 1000);
});

// Nettoyer le cache périodiquement (éviter les fuites mémoire)
setInterval(() => {
    if (imageCache.size > 20) {
        console.log(`🧹 Cleaning image cache (${imageCache.size} items)`);
        imageCache.clear();
    }
}, 60000); // Toutes les minutes
