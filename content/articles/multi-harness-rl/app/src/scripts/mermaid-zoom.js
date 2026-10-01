/**
 * Mermaid Zoom Integration
 * Provides medium-zoom functionality for Mermaid diagrams
 */

// Configuration - Désactiver le zoom Mermaid
const MERMAID_ZOOM_ENABLED = false;

console.log("🚀 Mermaid Zoom Script v20.0 loaded - DISABLED");

// Si désactivé, ne rien faire
if (!MERMAID_ZOOM_ENABLED) {
    console.log("🚫 Mermaid zoom is disabled, skipping initialization");
    // Export vide pour éviter les erreurs
    window.mermaidZoom = {
        init: () => { },
        cleanup: () => { },
        convertSvgToImage: () => { },
        openZoom: () => { },
        closeZoom: () => { }
    };
    // Arrêter l'exécution ici sans lancer d'erreur
    // Le reste du code ne s'exécutera pas car tout est dans le bloc if ci-dessous
} else {

    // Fonction pour appliquer les styles Mermaid au SVG selon le thème
    function applyMermaidStylesToSvg(svgElement) {
        try {
            const isDark = document.documentElement.getAttribute("data-theme") === "dark";
            console.log(`🎨 Applying Mermaid styles for theme: ${isDark ? 'dark' : 'light'}`);

            // Couleurs selon le thème - avec plus de contraste pour le zoom
            const colors = isDark ? {
                nodeFill: '#1a1a1a',
                nodeStroke: '#ffffff',
                nodeStrokeWidth: '2', // Plus épais pour plus de visibilité
                clusterFill: '#2a2a2a',
                clusterStroke: '#ffffff',
                clusterStrokeWidth: '2',
                pathStroke: '#ffffff',
                pathStrokeWidth: '2', // Plus épais pour les flèches
                textColor: '#ffffff',
                linkColor: '#ffffff' // Couleur spécifique pour les liens
            } : {
                nodeFill: '#ffffff',
                nodeStroke: '#000000', // Noir pur pour plus de contraste
                nodeStrokeWidth: '2',
                clusterFill: '#f9f9f9',
                clusterStroke: '#000000', // Noir pur
                clusterStrokeWidth: '2',
                pathStroke: '#000000', // Noir pur pour les flèches
                pathStrokeWidth: '2',
                textColor: '#000000', // Noir pur
                linkColor: '#000000' // Noir pur pour les liens
            };

            // Appliquer border-radius aux rectangles
            const rects = svgElement.querySelectorAll('rect:not(.flowchart-link), .node rect, .nodeLabel rect');
            rects.forEach(rect => {
                rect.setAttribute('rx', '8');
                rect.setAttribute('ry', '8');
                rect.setAttribute('fill', colors.nodeFill);
                rect.setAttribute('stroke', colors.nodeStroke);
                rect.setAttribute('stroke-width', colors.nodeStrokeWidth);
            });

            // Appliquer border-radius et couleurs aux clusters
            const clusterRects = svgElement.querySelectorAll('.cluster rect');
            clusterRects.forEach(rect => {
                rect.setAttribute('rx', '8');
                rect.setAttribute('ry', '8');
                rect.setAttribute('fill', colors.clusterFill);
                rect.setAttribute('stroke', colors.clusterStroke);
                rect.setAttribute('stroke-width', colors.clusterStrokeWidth);
            });

            // Appliquer les couleurs aux nœuds
            const nodes = svgElement.querySelectorAll('.node');
            nodes.forEach(node => {
                const rect = node.querySelector('rect');
                if (rect) {
                    rect.setAttribute('fill', colors.nodeFill);
                    rect.setAttribute('stroke', colors.nodeStroke);
                    rect.setAttribute('stroke-width', colors.nodeStrokeWidth);
                }
            });

            // Appliquer les couleurs aux clusters
            const clusters = svgElement.querySelectorAll('.cluster');
            clusters.forEach(cluster => {
                const rect = cluster.querySelector('rect');
                if (rect) {
                    rect.setAttribute('fill', colors.clusterFill);
                    rect.setAttribute('stroke', colors.clusterStroke);
                    rect.setAttribute('stroke-width', colors.clusterStrokeWidth);
                }
            });

            // Appliquer les couleurs aux chemins et flèches
            const paths = svgElement.querySelectorAll('.edgePath, path, .flowchart-link');
            paths.forEach(path => {
                path.setAttribute('stroke', colors.pathStroke);
                path.setAttribute('stroke-width', colors.pathStrokeWidth);
            });

            // Appliquer les couleurs aux liens spécifiquement
            const links = svgElement.querySelectorAll('.flowchart-link, .edgeLabel');
            links.forEach(link => {
                link.setAttribute('stroke', colors.linkColor);
                link.setAttribute('fill', colors.linkColor);
            });

            // Appliquer les couleurs au texte
            const textElements = svgElement.querySelectorAll('text, .nodeLabel text, .edgeLabel text');
            textElements.forEach(text => {
                text.setAttribute('fill', colors.textColor);
            });

            // Appliquer les couleurs aux marqueurs de flèches
            const markers = svgElement.querySelectorAll('marker, marker path');
            markers.forEach(marker => {
                marker.setAttribute('fill', colors.pathStroke);
                marker.setAttribute('stroke', colors.pathStroke);
            });

            console.log(`🎨 Applied Mermaid styles: ${rects.length} rects, ${clusters.length} clusters, ${paths.length} paths, ${textElements.length} text elements`);

        } catch (error) {
            console.error('❌ Error applying styles to SVG:', error);
        }
    }

    // Fonction pour extraire les styles CSS appliqués au diagramme Mermaid
    function getComputedStylesForMermaid(mermaidElement) {
        try {
            // Styles CSS essentiels pour Mermaid (hardcodés pour éviter les problèmes CORS)
            const essentialStyles = `
      .mermaid rect:not(.flowchart-link), 
      .mermaid .node rect, 
      .mermaid .nodeLabel rect {
        rx: 8px !important;
        ry: 8px !important;
      }
      
      .mermaid .cluster rect {
        rx: 8px !important;
        ry: 8px !important;
      }
      
      .mermaid .nodeLabel p {
        color: black !important;
      }
      
      .mermaid .edgeLabel {
        color: black !important;
      }
      
      .mermaid .edgePath {
        stroke: #333 !important;
      }
      
      .mermaid .node {
        fill: #fff !important;
        stroke: #333 !important;
      }
      
      .mermaid .cluster {
        fill: #f9f9f9 !important;
        stroke: #333 !important;
      }
    `;

            console.log(`🎨 Using essential CSS styles for Mermaid`);
            return essentialStyles;

        } catch (error) {
            console.error('❌ Error getting CSS styles:', error);
            return null;
        }
    }

    // Fonction pour forcer Mermaid à se re-rendre avec le bon thème
    function forceMermaidThemeUpdate(mermaidElement) {
        try {
            // Obtenir le thème actuel
            const currentTheme = document.documentElement.getAttribute("data-theme");
            console.log(`🎨 Forcing Mermaid theme update to: ${currentTheme}`);

            // Trouver le diagramme Mermaid original
            const mermaidCode = mermaidElement.textContent || mermaidElement.innerText;
            if (!mermaidCode) {
                console.log(`❌ No Mermaid code found to re-render`);
                return false;
            }

            // Forcer le re-rendu de Mermaid avec le nouveau thème
            if (window.mermaid) {
                // Utiliser l'API Mermaid pour re-rendre avec le bon thème
                const config = {
                    theme: currentTheme === 'dark' ? 'dark' : 'neutral',
                    autoTheme: true
                };

                console.log(`🔄 Re-rendering Mermaid with config:`, config);

                // Re-rendre le diagramme
                mermaid.init(undefined, mermaidElement);

                return true;
            } else {
                console.log(`❌ Mermaid API not available`);
                return false;
            }
        } catch (error) {
            console.error(`❌ Error forcing Mermaid theme update:`, error);
            return false;
        }
    }

    // Fonction pour convertir SVG en image en préservant EXACTEMENT les dimensions
    function convertSvgToImagePreservingDimensions(svgElement, wrapper, originalMermaid) {
        console.log(`🔄 Converting SVG to image with EXACT dimension preservation`);

        try {
            // Obtenir les dimensions EXACTES du wrapper actuel
            const wrapperRect = wrapper.getBoundingClientRect();
            const wrapperWidth = Math.round(wrapperRect.width);
            const wrapperHeight = Math.round(wrapperRect.height);

            console.log(`📏 Wrapper dimensions:`, { width: wrapperWidth, height: wrapperHeight });

            // Cloner le SVG
            const clonedSvg = svgElement.cloneNode(true);

            // Appliquer les styles Mermaid
            applyMermaidStylesToSvg(clonedSvg);

            // Créer une image PLUS GRANDE pour permettre un vrai zoom (2x la taille)
            const zoomFactor = 2;
            const imageWidth = wrapperWidth * zoomFactor;
            const imageHeight = wrapperHeight * zoomFactor;

            // Forcer les dimensions plus grandes sur le SVG
            clonedSvg.setAttribute('width', imageWidth);
            clonedSvg.setAttribute('height', imageHeight);
            clonedSvg.style.width = `${imageWidth}px`;
            clonedSvg.style.height = `${imageHeight}px`;

            console.log(`🔍 Creating zoomable image:`, {
                original: { width: wrapperWidth, height: wrapperHeight },
                zoomed: { width: imageWidth, height: imageHeight },
                factor: zoomFactor
            });

            // Créer une URL data pour le SVG
            const svgData = new XMLSerializer().serializeToString(clonedSvg);
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const svgUrl = URL.createObjectURL(svgBlob);

            // Créer un élément img avec les dimensions du wrapper (affichage normal)
            const imgElement = document.createElement('img');
            imgElement.src = svgUrl;
            imgElement.style.width = `${wrapperWidth}px`;  // Affichage normal
            imgElement.style.height = `${wrapperHeight}px`; // Affichage normal
            imgElement.style.display = 'block';
            imgElement.setAttribute('data-zoomable', '1');
            imgElement.classList.add('mermaid-zoom-image'); // Classe spécifique pour Mermaid

            // Ajouter les attributs pour medium-zoom (dimensions réelles de l'image)
            imgElement.setAttribute('data-zoom-width', imageWidth);
            imgElement.setAttribute('data-zoom-height', imageHeight);

            console.log(`🖼️ Image created with exact dimensions:`, { width: wrapperWidth, height: wrapperHeight });

            // Remplacer le contenu du wrapper par l'image
            wrapper.innerHTML = '';
            wrapper.appendChild(imgElement);

            // Retirer la classe "converting" pour restaurer l'opacité normale
            wrapper.classList.remove("converting");

            // Attendre que l'image soit chargée puis initialiser medium-zoom
            imgElement.onload = () => {
                console.log(`✅ Image loaded, initializing REAL medium-zoom`);

                const isDark = document.documentElement.getAttribute("data-theme") === "dark";
                const background = isDark ? "rgba(0,0,0,.9)" : "rgba(0,0,0,.85)";

                // Utiliser VRAI medium-zoom
                const zoomInstance = window.mediumZoom(imgElement, {
                    background,
                    margin: 24,
                    scrollOffset: 0,
                });

                console.log(`🎉 REAL medium-zoom initialized with exact dimensions!`);

                // Forcer les bonnes couleurs en JavaScript (contournement du CSS global)
                const forceCorrectColors = () => {
                    // Trouver l'image zoomée dans le DOM
                    const zoomedImage = document.querySelector('.medium-zoom-image--opened');
                    if (zoomedImage && zoomedImage.classList.contains('mermaid-zoom-image')) {
                        console.log(`🎨 Forcing correct colors for Mermaid zoom image`);
                        zoomedImage.style.filter = 'none';
                        zoomedImage.style.setProperty('filter', 'none', 'important');
                    }

                    // Forcer les z-index élevés
                    const overlay = document.querySelector('.medium-zoom-overlay');
                    if (overlay) {
                        console.log(`🔝 Forcing high z-index for overlay`);
                        overlay.style.zIndex = '9999999';
                        overlay.style.setProperty('z-index', '9999999', 'important');
                    }

                    if (zoomedImage) {
                        console.log(`🔝 Forcing high z-index for zoomed image`);
                        zoomedImage.style.zIndex = '10000000';
                        zoomedImage.style.setProperty('z-index', '10000000', 'important');
                    }
                };

                // Écouter les événements de zoom pour appliquer les bonnes couleurs
                imgElement.addEventListener('zoom:open', forceCorrectColors);
                imgElement.addEventListener('zoom:opened', forceCorrectColors);

                // Observer pour les changements de thème
                const themeObserver = new MutationObserver(() => {
                    console.log(`🎨 Theme changed, forcing Mermaid re-render with new theme`);

                    // Forcer Mermaid à se re-rendre avec le nouveau thème
                    const currentTheme = document.documentElement.getAttribute("data-theme");
                    console.log(`🎨 Current theme: ${currentTheme}`);

                    // Attendre un peu pour que Mermaid détecte le changement de thème
                    setTimeout(() => {
                        // Forcer Mermaid à se re-rendre avec le nouveau thème
                        const themeUpdated = forceMermaidThemeUpdate(originalMermaid);

                        if (themeUpdated) {
                            // Attendre que le re-rendu soit terminé
                            setTimeout(() => {
                                // Re-obtenir le SVG mis à jour
                                const updatedSvgElement = originalMermaid.querySelector("svg");
                                if (updatedSvgElement) {
                                    console.log(`🔄 Re-converting with updated SVG for theme: ${currentTheme}`);
                                    convertSvgToImagePreservingDimensions(updatedSvgElement, wrapper, originalMermaid);
                                } else {
                                    console.log(`❌ No updated SVG found after theme change`);
                                }
                            }, 200); // Délai pour laisser Mermaid finir le re-rendu
                        } else {
                            console.log(`❌ Failed to update Mermaid theme, using current SVG`);
                            // Fallback: utiliser le SVG actuel même s'il n'est pas mis à jour
                            const currentSvgElement = originalMermaid.querySelector("svg");
                            if (currentSvgElement) {
                                convertSvgToImagePreservingDimensions(currentSvgElement, wrapper, originalMermaid);
                            }
                        }
                    }, 100); // Petit délai pour laisser Mermaid se mettre à jour
                });

                themeObserver.observe(document.documentElement, {
                    attributes: true,
                    attributeFilter: ["data-theme"],
                });

                wrapper._themeObserver = themeObserver;
            };

            imgElement.onerror = (error) => {
                console.error(`❌ Error loading SVG as image:`, error);
                // Fallback: utiliser le zoom custom
                wrapper.innerHTML = '';
                wrapper.appendChild(originalMermaid);
                wrapper.classList.remove("converting");
                wrapper.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    openMermaidZoom(wrapper, originalMermaid);
                });
            };

        } catch (error) {
            console.error("❌ Error converting SVG to image:", error);
            // Fallback: utiliser le zoom custom
            wrapper.classList.remove("converting");
            wrapper.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                openMermaidZoom(wrapper, originalMermaid);
            });
        }
    }

    // Fonction pour initialiser medium-zoom directement sur le SVG
    function initializeMediumZoomOnSvg(svgElement, wrapper, originalMermaid) {
        try {
            // S'assurer que le SVG a des dimensions fixes pour éviter le flicker
            const bbox = svgElement.getBBox();
            const width = bbox.width || svgElement.clientWidth || 800;
            const height = bbox.height || svgElement.clientHeight || 600;

            // Fixer les dimensions du SVG pour éviter les changements
            svgElement.setAttribute('width', width);
            svgElement.setAttribute('height', height);
            svgElement.style.width = '100%';
            svgElement.style.height = 'auto';
            svgElement.style.display = 'block';

            // Retirer la classe "converting" pour restaurer l'opacité normale
            wrapper.classList.remove("converting");

            console.log(`📏 SVG dimensions fixed:`, { width, height });

            // Initialiser medium-zoom directement sur le wrapper SVG
            const isDark = document.documentElement.getAttribute("data-theme") === "dark";
            const background = isDark ? "rgba(0,0,0,.9)" : "rgba(0,0,0,.85)";

            // Utiliser medium-zoom sur le wrapper qui contient le SVG
            window.mediumZoom(wrapper, {
                background,
                margin: 24,
                scrollOffset: 0,
            });

            console.log(`🎉 Medium-zoom initialized directly on SVG wrapper!`);

            // Observer pour les changements de thème
            const themeObserver = new MutationObserver(() => {
                console.log(`🎨 Theme changed, updating medium-zoom background`);
                // Re-initialiser medium-zoom avec le nouveau thème
                initializeMediumZoomOnSvg(svgElement, wrapper, originalMermaid);
            });

            themeObserver.observe(document.documentElement, {
                attributes: true,
                attributeFilter: ["data-theme"],
            });

            // Stocker l'observer pour le nettoyer plus tard
            wrapper._themeObserver = themeObserver;

        } catch (error) {
            console.error("❌ Error initializing medium-zoom on SVG:", error);
            // Fallback: utiliser le zoom custom
            wrapper.classList.remove("converting");
            wrapper.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                openMermaidZoom(wrapper, originalMermaid);
            });
        }
    }

    // Fonction pour attendre que Mermaid soit stable (évite le flicker)
    function waitForMermaidStable(mermaidEl, wrapper, index) {
        // Approche simplifiée : attendre un délai fixe puis convertir
        console.log(`⏳ Waiting for Mermaid ${index} to stabilize...`);

        setTimeout(() => {
            const svgElement = mermaidEl.querySelector("svg");
            if (svgElement) {
                console.log(`🎯 Converting Mermaid ${index} to image`);
                convertSvgToImageForMediumZoom(svgElement, wrapper, mermaidEl);
            } else {
                console.log(`❌ No SVG found in Mermaid element ${index}`);
            }
        }, 2000); // Attendre 2 secondes puis convertir
    }

    // Fonction pour convertir SVG en image SANS canvas (pour éviter l'erreur de sécurité)
    function convertSvgToImageForMediumZoom(svgElement, wrapper, originalMermaid) {
        console.log(`🔄 Converting SVG to image for REAL medium-zoom`);

        try {
            // Cloner le SVG
            const clonedSvg = svgElement.cloneNode(true);

            // Appliquer directement les styles SVG pour préserver l'apparence
            applyMermaidStylesToSvg(clonedSvg);

            // S'assurer que le SVG a des dimensions
            const bbox = clonedSvg.getBBox();
            const width = bbox.width || clonedSvg.clientWidth || 800;
            const height = bbox.height || clonedSvg.clientHeight || 600;

            console.log(`📏 SVG dimensions:`, { width, height });

            // Créer une image directement à partir du SVG (sans canvas)
            const svgData = new XMLSerializer().serializeToString(clonedSvg);

            // Ajouter les dimensions au SVG
            clonedSvg.setAttribute("width", width);
            clonedSvg.setAttribute("height", height);
            const svgWithDimensions = new XMLSerializer().serializeToString(clonedSvg);

            // Créer une URL data pour le SVG
            const svgBlob = new Blob([svgWithDimensions], {
                type: "image/svg+xml;charset=utf-8",
            });
            const svgUrl = URL.createObjectURL(svgBlob);

            // Créer un élément img avec le SVG
            const imgElement = document.createElement("img");
            imgElement.src = svgUrl;
            imgElement.style.width = "100%";
            imgElement.style.height = "auto";
            imgElement.style.display = "block";
            imgElement.setAttribute("data-zoomable", "1");

            // Remplacer le contenu du wrapper par l'image
            wrapper.innerHTML = "";
            wrapper.appendChild(imgElement);

            // Retirer la classe "converting" pour restaurer l'opacité normale
            wrapper.classList.remove("converting");

            console.log(`🖼️ SVG converted to img element`);

            // Attendre que l'image soit chargée puis initialiser medium-zoom
            imgElement.onload = () => {
                console.log(`✅ Image loaded, initializing REAL medium-zoom`);

                const isDark = document.documentElement.getAttribute("data-theme") === "dark";
                const background = isDark ? "rgba(0,0,0,.9)" : "rgba(0,0,0,.85)";

                // Utiliser VRAI medium-zoom
                window.mediumZoom(imgElement, {
                    background,
                    margin: 24,
                    scrollOffset: 0,
                });

                console.log(`🎉 REAL medium-zoom initialized on Mermaid!`);

                // Observer pour les changements de thème - reconvertir l'image
                const themeObserver = new MutationObserver(() => {
                    console.log(`🎨 Theme changed, reconverting Mermaid image`);
                    // Reconverter l'image avec le nouveau thème
                    convertSvgToImageForMediumZoom(svgElement, wrapper, originalMermaid);
                });

                themeObserver.observe(document.documentElement, {
                    attributes: true,
                    attributeFilter: ["data-theme"],
                });

                // Stocker l'observer pour le nettoyer plus tard
                wrapper._themeObserver = themeObserver;
            };

            imgElement.onerror = (error) => {
                console.error(`❌ Error loading SVG as image:`, error);
                // Fallback: utiliser le zoom custom
                wrapper.innerHTML = "";
                wrapper.appendChild(originalMermaid);
                wrapper.addEventListener("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    openMermaidZoom(wrapper, originalMermaid);
                });
            };

        } catch (error) {
            console.error("❌ Error converting SVG to image:", error);
            // Fallback: utiliser le zoom custom
            wrapper.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                openMermaidZoom(wrapper, originalMermaid);
            });
        }
    }

    // Fonction pour ouvrir le zoom custom des diagrammes Mermaid (comme medium-zoom)
    function openMermaidZoom(wrapper, mermaidElement) {
        // Créer l'overlay (comme medium-zoom)
        const overlay = document.createElement("div");
        overlay.className = "medium-zoom-overlay";
        overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.85);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
    opacity: 0;
    transition: opacity 0.3s ease;
  `;

        // Créer l'image zoomée (comme medium-zoom)
        const zoomImage = document.createElement("div");
        zoomImage.className = "medium-zoom-image";
        zoomImage.style.cssText = `
    max-width: 90%;
    max-height: 90%;
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    transform: scale(0.8);
    transition: transform 0.3s ease;
    overflow: auto;
  `;

        // Cloner le diagramme Mermaid
        const clonedMermaid = mermaidElement.cloneNode(true);
        clonedMermaid.style.cssText = `
    width: 100%;
    height: auto;
    max-width: none;
    display: block;
  `;

        zoomImage.appendChild(clonedMermaid);
        overlay.appendChild(zoomImage);

        // Ajouter l'overlay au body
        document.body.appendChild(overlay);

        // Animation d'ouverture (comme medium-zoom)
        requestAnimationFrame(() => {
            overlay.style.opacity = "1";
            zoomImage.style.transform = "scale(1)";
        });

        // Fermer au clic sur l'overlay
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) {
                closeMermaidZoom(overlay);
            }
        });

        // Fermer avec Escape
        const handleEscape = (e) => {
            if (e.key === "Escape") {
                closeMermaidZoom(overlay);
                document.removeEventListener("keydown", handleEscape);
            }
        };
        document.addEventListener("keydown", handleEscape);

        // Fermer au scroll (comme medium-zoom)
        const handleScroll = () => {
            closeMermaidZoom(overlay);
        };
        window.addEventListener("wheel", handleScroll, { passive: true });
        window.addEventListener("touchmove", handleScroll, { passive: true });
        window.addEventListener("scroll", handleScroll, { passive: true });

        // Stocker les handlers pour les nettoyer
        overlay._handlers = { handleEscape, handleScroll };
    }

    function closeMermaidZoom(overlay) {
        // Animation de fermeture
        overlay.style.opacity = "0";
        const zoomImage = overlay.querySelector(".medium-zoom-image");
        if (zoomImage) {
            zoomImage.style.transform = "scale(0.8)";
        }

        // Nettoyer les event listeners
        if (overlay._handlers) {
            document.removeEventListener("keydown", overlay._handlers.handleEscape);
            window.removeEventListener("wheel", overlay._handlers.handleScroll);
            window.removeEventListener("touchmove", overlay._handlers.handleScroll);
            window.removeEventListener("scroll", overlay._handlers.handleScroll);
        }

        // Supprimer l'overlay après l'animation
        setTimeout(() => {
            if (document.body.contains(overlay)) {
                document.body.removeChild(overlay);
            }
        }, 300);
    }

    // Fonction pour nettoyer les observers
    function cleanupMermaidObservers(wrapper) {
        if (wrapper._themeObserver) {
            wrapper._themeObserver.disconnect();
            wrapper._themeObserver = null;
        }
    }

    // Fonction principale pour initialiser le zoom Mermaid
    function setupMermaidZoom() {
        console.log(`🎯 setupMermaidZoom called`);
        const mermaidElements = document.querySelectorAll(".mermaid");
        console.log(`🔍 Found ${mermaidElements.length} Mermaid elements`);

        let processedCount = 0;
        mermaidElements.forEach((mermaidEl, index) => {
            // Vérifier si déjà wrappé
            if (
                mermaidEl.parentElement &&
                mermaidEl.parentElement.classList.contains("mermaid-zoom-wrapper")
            ) {
                console.log(`📦 Mermaid ${index} already wrapped`);
                processedCount++;
                return;
            }

            console.log(`📦 Wrapping Mermaid element ${index}`);

            // Créer le wrapper
            const wrapper = document.createElement("div");
            wrapper.className = "mermaid-zoom-wrapper";
            wrapper.setAttribute("data-zoomable", "1");

            // Insérer le wrapper avant l'élément Mermaid
            mermaidEl.parentNode.insertBefore(wrapper, mermaidEl);

            // Déplacer l'élément Mermaid dans le wrapper
            wrapper.appendChild(mermaidEl);

            // S'assurer que le wrapper a des dimensions
            wrapper.style.display = "block";
            wrapper.style.width = "100%";
            wrapper.style.maxWidth = "100%";

            // Ajouter la classe "converting" pour masquer le flicker
            wrapper.classList.add("converting");

            // Convertir SVG en image pour utiliser VRAI medium-zoom
            console.log(`✅ Converting Mermaid to image for REAL medium-zoom`);

            // Utiliser le VRAI medium-zoom avec conversion SVG → image SANS flicker
            console.log(`✅ Setting up REAL medium-zoom for Mermaid ${index}`);

            // Attendre que Mermaid soit stable puis convertir avec préservation des dimensions
            setTimeout(() => {
                const svgElement = mermaidEl.querySelector("svg");
                if (svgElement) {
                    console.log(`🎯 Converting SVG to image with dimension preservation`);
                    convertSvgToImagePreservingDimensions(svgElement, wrapper, mermaidEl);
                } else {
                    console.log(`❌ No SVG found in Mermaid element ${index}`);
                }
            }, 1000); // Réduit de 2000ms à 1000ms pour être plus rapide
        });

        console.log(`✅ Processed ${processedCount} already wrapped, ${mermaidElements.length - processedCount} new diagrams`);
    }

    // Observer global pour forcer les bonnes couleurs et z-index sur les images Mermaid zoomées
    function setupGlobalMermaidColorFix() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    // Vérifier si une image Mermaid est zoomée
                    const zoomedMermaidImage = document.querySelector('.medium-zoom-image--opened.mermaid-zoom-image');
                    if (zoomedMermaidImage) {
                        console.log(`🎨 Global fix: Found zoomed Mermaid image, forcing correct colors and z-index`);
                        zoomedMermaidImage.style.filter = 'none';
                        zoomedMermaidImage.style.setProperty('filter', 'none', 'important');
                        zoomedMermaidImage.style.zIndex = '10000000';
                        zoomedMermaidImage.style.setProperty('z-index', '10000000', 'important');
                    }

                    // Vérifier si l'overlay medium-zoom existe
                    const overlay = document.querySelector('.medium-zoom-overlay');
                    if (overlay) {
                        console.log(`🔝 Global fix: Found medium-zoom overlay, forcing high z-index`);
                        overlay.style.zIndex = '9999999';
                        overlay.style.setProperty('z-index', '9999999', 'important');
                    }
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        console.log(`🎨 Global Mermaid color and z-index fix observer started`);
    }

    // Fonction pour initialiser le zoom Mermaid avec retry
    function initMermaidZoom() {
        console.log("🎯 initMermaidZoom called");

        // Attendre que mediumZoom soit disponible
        const checkMediumZoom = () => {
            if (window.mediumZoom) {
                console.log("✅ mediumZoom found, initializing Mermaid zoom");
                setupMermaidZoom();
            } else {
                console.log("⏳ Waiting for mediumZoom...");
                setTimeout(checkMediumZoom, 100);
            }
        };

        checkMediumZoom();
    }

    // Export des fonctions pour utilisation globale
    window.MermaidZoom = {
        init: initMermaidZoom,
        setup: setupMermaidZoom,
        cleanup: cleanupMermaidObservers,
        convertSvgToImage: convertSvgToImageForMediumZoom,
        openZoom: openMermaidZoom,
        closeZoom: closeMermaidZoom
    };

    // Auto-initialisation si le DOM est déjà chargé
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            initMermaidZoom();
            setupGlobalMermaidColorFix();
        });
    } else {
        initMermaidZoom();
        setupGlobalMermaidColorFix();
    }

    // Aussi après le chargement complet
    window.addEventListener("load", () => {
        setTimeout(() => {
            initMermaidZoom();
            setupGlobalMermaidColorFix();
        }, 1000);
    });

    // Observer simple pour les nouveaux diagrammes Mermaid (avec debounce)
    let resizeTimeout;
    const observer = new MutationObserver(() => {
        // Debounce pour éviter les appels multiples
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            // Vérifier s'il y a vraiment de nouveaux diagrammes
            const mermaidElements = document.querySelectorAll(".mermaid");
            const wrappedElements = document.querySelectorAll(".mermaid-zoom-wrapper");

            // Seulement si il y a plus de diagrammes que de wrappers
            if (mermaidElements.length > wrappedElements.length) {
                console.log(`🔄 New Mermaid diagrams detected: ${mermaidElements.length} total, ${wrappedElements.length} wrapped`);
                initMermaidZoom();
            }
        }, 500); // Délai plus long pour éviter les appels fréquents
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
    });

    console.log("🚀 Mermaid Zoom Script v19.0 loaded - DEBOUNCED observer to prevent resize loops");
} // Fin du bloc else (MERMAID_ZOOM_ENABLED)
