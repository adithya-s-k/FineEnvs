# Mermaid Zoom Integration

Ce module fournit une intégration du zoom medium-zoom pour les diagrammes Mermaid dans le projet Astro.

## Fonctionnalités

- ✅ **Zoom medium-zoom identique aux images** : Utilise la vraie bibliothèque medium-zoom
- ✅ **Conversion SVG → Image** : Convertit les diagrammes Mermaid en images pour le zoom
- ✅ **Adaptation au thème** : Reconversion automatique lors des changements de thème
- ✅ **Fallback robuste** : Zoom custom si la conversion échoue
- ✅ **Auto-initialisation** : Détection automatique des nouveaux diagrammes

## Architecture

### Fichiers

- `src/scripts/mermaid-zoom.js` : Module principal avec toute la logique
- `src/pages/index.astro` : Import du module (ligne 209)
- `src/styles/components/_mermaid.css` : Styles CSS pour les wrappers

### Fonctions principales

- `convertSvgToImageForMediumZoom()` : Convertit SVG en image pour medium-zoom
- `setupMermaidZoom()` : Wrappe les diagrammes et initialise le zoom
- `initMermaidZoom()` : Fonction principale d'initialisation
- `openMermaidZoom()` / `closeMermaidZoom()` : Zoom custom de fallback

### API globale

```javascript
window.MermaidZoom = {
  init: initMermaidZoom,           // Initialiser le zoom
  setup: setupMermaidZoom,         // Wrapper et conversion
  cleanup: cleanupMermaidObservers, // Nettoyer les observers
  convertSvgToImage: convertSvgToImageForMediumZoom,
  openZoom: openMermaidZoom,       // Ouvrir zoom custom
  closeZoom: closeMermaidZoom      // Fermer zoom custom
};
```

## Comment ça marche

1. **Détection** : Le script détecte tous les éléments `.mermaid`
2. **Wrapper** : Crée un `div.mermaid-zoom-wrapper` autour de chaque diagramme
3. **Conversion** : Convertit le SVG en image `<img>` avec `data-zoomable="1"`
4. **Medium-zoom** : Initialise `window.mediumZoom()` sur l'image générée
5. **Thème** : Observer pour reconvertir lors des changements de thème

## Avantages de cette approche

- **Vrai medium-zoom** : Comportement identique aux images existantes
- **Pas d'erreur canvas** : Évite l'erreur "Tainted canvases" en utilisant des URLs blob
- **Réactivité au thème** : Les diagrammes s'adaptent automatiquement
- **Code modulaire** : Facile à maintenir et déboguer
- **Performance** : Une seule conversion par diagramme

## Debug

Le module inclut des logs détaillés :
- `🚀 Mermaid Zoom Script loaded`
- `🔍 Found X Mermaid elements`
- `📦 Wrapping Mermaid element X`
- `🖼️ SVG converted to img element`
- `🎉 REAL medium-zoom initialized on Mermaid!`
- `🎨 Theme changed, reconverting Mermaid image`
