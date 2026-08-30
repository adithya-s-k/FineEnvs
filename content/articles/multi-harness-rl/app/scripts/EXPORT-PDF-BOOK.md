# 📚 Export PDF Book avec Paged.js

Système de génération de PDF professionnel avec mise en page type livre, propulsé par **Paged.js**.

## ✨ Fonctionnalités

### Mise en page professionnelle
- ✅ **Pagination automatique** avec Paged.js
- ✅ **Running headers** : titres de chapitres en haut de page
- ✅ **Numérotation des pages** : alternée gauche/droite
- ✅ **Marges asymétriques** : optimisées pour reliure (recto/verso)
- ✅ **Gestion veuves et orphelines** : évite les lignes isolées
- ✅ **Typographie professionnelle** : justification, césure automatique

### Éléments de livre
- 📖 Compteurs automatiques : chapitres, figures, tableaux
- 📑 Notes de bas de page (si implémentées)
- 🔢 Numérotation hiérarchique (1.2.3, etc.)
- 📊 Support complet des visualisations D3/Plotly
- 🖼️ Figures avec légendes numérotées
- 📝 Citations et références

## 🚀 Utilisation

### Commande de base

```bash
npm run export:pdf:book
```

Cette commande va :
1. Builder le site Astro (si nécessaire)
2. Démarrer un serveur preview
3. Charger la page et injecter Paged.js
4. Paginer le contenu automatiquement
5. Générer le PDF dans `dist/article-book.pdf`

### Options disponibles

```bash
# Thème sombre
npm run export:pdf:book -- --theme=dark

# Format personnalisé
npm run export:pdf:book -- --format=Letter

# Nom de fichier custom
npm run export:pdf:book -- --filename=mon-livre

# Combinaison d'options
npm run export:pdf:book -- --theme=light --format=A4 --filename=thesis
```

#### Options détaillées

| Option | Valeurs | Défaut | Description |
|--------|---------|--------|-------------|
| `--theme` | `light`, `dark` | `light` | Thème de couleur |
| `--format` | `A4`, `Letter`, `Legal`, `A3`, `Tabloid` | `A4` | Format de page |
| `--filename` | `string` | `article-book` | Nom du fichier de sortie |
| `--wait` | `full`, `images`, `plotly`, `d3` | `full` | Stratégie d'attente |

## 📐 Format de page

Le système utilise des marges optimisées pour l'impression livre :

### Pages de droite (recto)
- Marge gauche : **25mm** (reliure)
- Marge droite : **20mm**
- Header droite : titre de section
- Footer droite : numéro de page

### Pages de gauche (verso)
- Marge gauche : **20mm**
- Marge droite : **25mm** (reliure)
- Header gauche : titre de chapitre
- Footer gauche : numéro de page

### Première page
- Marges augmentées (40mm haut/bas)
- Pas de headers/footers
- Centrée

## 🎨 Personnalisation CSS

Le style livre est défini dans :
```
app/src/styles/_print-book.css
```

### Modifier les marges

```css
@page {
  margin-top: 20mm;
  margin-bottom: 25mm;
  /* ... */
}

@page :left {
  margin-left: 20mm;
  margin-right: 25mm;
}

@page :right {
  margin-left: 25mm;
  margin-right: 20mm;
}
```

### Modifier la typographie

```css
body {
  font-family: "Georgia", "Palatino", "Times New Roman", serif;
  font-size: 11pt;
  line-height: 1.6;
}

h2 {
  font-size: 18pt;
  /* ... */
}
```

### Personnaliser les running headers

```css
@page :left {
  @top-left {
    content: string(chapter-title);
    font-size: 9pt;
    font-style: italic;
    /* ... */
  }
}

@page :right {
  @top-right {
    content: string(section-title);
    /* ... */
  }
}
```

### Ajouter un logo/filigrane

```css
@page {
  background-image: url('/logo.png');
  background-position: bottom center;
  background-size: 20mm;
  background-repeat: no-repeat;
}
```

## 🔧 Configuration Paged.js avancée

### Hooks JavaScript personnalisés

Vous pouvez ajouter des hooks Paged.js dans le script `export-pdf-book.mjs` :

```javascript
// Après l'injection de Paged.js
await page.evaluate(() => {
  class BookHooks extends window.Paged.Handler {
    beforeParsed(content) {
      // Modifier le contenu avant pagination
    }
    
    afterParsed(parsed) {
      // Après l'analyse
    }
    
    afterRendered(pages) {
      // Après le rendu de toutes les pages
      console.log(`Rendered ${pages.length} pages`);
    }
  }
  
  window.Paged.registerHandlers(BookHooks);
});
```

### Forcer des sauts de page

Dans votre MDX :

```mdx
## Chapitre 1

Contenu...

<div style="break-after: page;"></div>

## Chapitre 2 (commence sur une nouvelle page)
```

Ou avec une classe CSS :

```css
.chapter-break {
  break-after: page;
}
```

## 📊 Visualisations

Les graphiques D3 et Plotly sont automatiquement :
- ✅ Redimensionnés pour le format livre
- ✅ Rendus en haute qualité
- ✅ Évitent les coupures de page
- ✅ Conservent l'interactivité dans le HTML source

## 🐛 Dépannage

### Le PDF est vide ou incomplet

```bash
# Augmenter le temps d'attente
npm run export:pdf:book -- --wait=full
```

### Les images ne s'affichent pas

Vérifiez que les chemins d'images sont **absolus** dans le HTML :
```html
<!-- ✅ Bon -->
<img src="/images/photo.jpg">

<!-- ❌ Mauvais -->
<img src="images/photo.jpg">
```

### Les graphiques sont coupés

Ajoutez dans `_print-book.css` :
```css
.your-chart-class {
  max-height: 180mm !important;
  break-inside: avoid;
}
```

### Erreur "Paged.js not found"

```bash
# Réinstaller Paged.js
cd app
npm install pagedjs
```

### Le serveur ne démarre pas

```bash
# Port déjà utilisé ? Changer le port
PREVIEW_PORT=8081 npm run export:pdf:book
```

## 📚 Ressources Paged.js

- **Documentation officielle** : https://pagedjs.org/documentation/
- **Spécifications CSS Paged Media** : https://www.w3.org/TR/css-page-3/
- **Exemples** : https://pagedjs.org/examples/

## 🆚 Différences avec export:pdf standard

| Fonctionnalité | `export:pdf` | `export:pdf:book` |
|----------------|--------------|-------------------|
| Pagination | Navigateur standard | Paged.js professionnel |
| Running headers | ❌ | ✅ |
| Marges reliure | ❌ | ✅ |
| Numérotation avancée | ❌ | ✅ |
| Compteurs automatiques | ❌ | ✅ |
| Gestion veuves/orphelines | Basique | Avancée |
| Notes de bas de page | ❌ | ✅ (si activées) |
| Contrôle typographique | Standard | Professionnel |
| Table des matières | Manuelle | Automatique (avec CSS) |

## 💡 Conseils pour un résultat optimal

1. **Structurez votre contenu** avec des `<h2>` pour les chapitres
2. **Utilisez des `<h3>` pour les sections** (apparaissent dans les headers)
3. **Ajoutez des IDs** aux titres pour les références croisées
4. **Optimisez les images** : résolution 300 DPI pour l'impression
5. **Testez le rendu** avant l'impression finale
6. **Évitez les couleurs vives** en mode print (privilégier les niveaux de gris)

## 🎯 Cas d'usage

Ce système est idéal pour :
- 📘 **Thèses et mémoires**
- 📗 **Livres techniques**
- 📕 **Rapports académiques**
- 📙 **Documentation longue**
- 📓 **E-books premium**
- 📔 **Revues scientifiques**

## 🔮 Améliorations futures

- [ ] Génération automatique de table des matières
- [ ] Support des index
- [ ] Références croisées automatiques
- [ ] Export en EPUB
- [ ] Templates de livre préconfigurés
- [ ] Mode "two-up" pour visualisation double page

---

**Créé avec ❤️ par votre équipe template**

