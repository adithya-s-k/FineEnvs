# 📚 Export PDF Livre - Guide Complet

Système de génération de PDF professionnel avec mise en page type livre pour votre template d'article scientifique.

## 🎯 Objectif

Créer des PDFs de qualité professionnelle avec :
- Typographie soignée (Georgia, justification, césure)
- Marges asymétriques pour reliure
- Running headers avec titres de chapitres
- Numérotation de pages gauche/droite
- Gestion veuves et orphelines
- Style livre académique/éditorial

## 📦 Ce qui a été créé

### Fichiers créés

```
app/
├── scripts/
│   ├── export-pdf-book.mjs          ← Version avec Paged.js (avancée, en cours)
│   ├── export-pdf-book-simple.mjs   ← Version simple (RECOMMANDÉE ✅)
│   └── EXPORT-PDF-BOOK.md           ← Documentation détaillée
└── src/
    └── styles/
        └── _print-book.css          ← Styles CSS Paged Media
```

### Commandes npm ajoutées

```json
{
  "export:pdf:book": "Version Paged.js (expérimentale)",
  "export:pdf:book:simple": "Version simple (stable ✅)"
}
```

## 🚀 Utilisation

### Commande recommandée

```bash
npm run export:pdf:book:simple
```

Le PDF sera généré dans :
- `dist/article-book.pdf`
- `public/article-book.pdf` (copie automatique)

### Options disponibles

```bash
# Thème sombre
npm run export:pdf:book:simple -- --theme=dark

# Format Letter
npm run export:pdf:book:simple -- --format=Letter

# Nom personnalisé
npm run export:pdf:book:simple -- --filename=ma-these

# Combinaison
npm run export:pdf:book:simple -- --theme=light --format=A4 --filename=livre
```

## 🎨 Caractéristiques du style livre

### Marges

```
Pages droites (recto)  │  Pages gauches (verso)
                       │
   20mm ──┐            │            ┌── 25mm
          │            │            │
  ┌───────┴──────┐     │     ┌──────┴───────┐
  │              │     │     │              │
  │   CONTENU    │     │     │   CONTENU    │
  │              │     │     │              │
  └──────────────┘     │     └──────────────┘
        25mm           │           20mm
     (reliure)         │        (reliure)
```

### Typographie

- **Police** : Georgia, Palatino (serif)
- **Taille** : 11pt
- **Interlignage** : 1.6
- **Alignement** : Justifié avec césure automatique
- **Retrait** : 5mm pour les paragraphes suivants

### Titres

```css
H2 (Chapitres)     → 18pt, numérotés (1. 2. 3.)
H3 (Sections)      → 14pt, numérotés (1.1, 1.2)  
H4 (Sous-sections) → 12pt
```

### Compteurs automatiques

- Chapitres : 1, 2, 3...
- Sections : 1.1, 1.2, 2.1...
- Figures : Figure 1.1, Figure 1.2...
- Tableaux : idem

## 📐 Configuration CSS

Le fichier `_print-book.css` contient tous les styles. Vous pouvez personnaliser :

### Changer les polices

```css
body {
  font-family: "Baskerville", "Georgia", serif;
  font-size: 12pt;
}
```

### Ajuster les marges

```css
@page {
  margin-top: 25mm;
  margin-bottom: 30mm;
}

@page :left {
  margin-left: 18mm;
  margin-right: 30mm;
}
```

### Personnaliser les headers

```css
@page :left {
  @top-left {
    content: string(chapter-title);
    font-size: 10pt;
    color: #333;
  }
}
```

### Forcer un saut de page

Dans votre MDX :
```mdx
## Chapitre 1

Contenu...

---

## Chapitre 2 (nouvelle page)
```

Ou avec CSS :
```css
.new-chapter {
  break-before: page;
}
```

## 🆚 Comparaison des versions

| Fonctionnalité | Simple | Paged.js |
|----------------|--------|----------|
| **Stabilité** | ✅ Excellente | ⚠️ En cours |
| **Vitesse** | ✅ Rapide | ⏱️ Plus lent |
| **Setup** | ✅ Aucun | 📦 Paged.js requis |
| **Marges reliure** | ✅ | ✅ |
| **Running headers** | ⚠️ Limité | ✅ Avancé |
| **Notes de bas de page** | ❌ | ✅ |
| **Table matières auto** | ❌ | ✅ |
| **Qualité typo** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Quand utiliser quelle version ?

**Version Simple** (recommandée) :
- ✅ Pour la plupart des cas d'usage
- ✅ Stabilité prioritaire
- ✅ Génération rapide
- ✅ Résultats prévisibles

**Version Paged.js** (expérimentale) :
- 🔬 Pour tester les fonctionnalités avancées
- 📚 Si vous avez besoin de notes de bas de page
- 📖 Pour des tables des matières générées automatiquement
- ⚠️ Nécessite plus de tests

## 🐛 Dépannage

### Le PDF est vide

```bash
# Reconstruire d'abord
npm run build
npm run export:pdf:book:simple
```

### Les images manquent

Vérifiez que les chemins sont absolus :
```html
<!-- ✅ Bon -->
<img src="/images/photo.jpg">

<!-- ❌ Mauvais -->
<img src="images/photo.jpg">
```

### Les graphiques sont coupés

Dans `_print-book.css`, ajoutez :
```css
.your-chart {
  max-height: 200mm;
  break-inside: avoid;
}
```

### Port 8080 déjà utilisé

```bash
PREVIEW_PORT=8081 npm run export:pdf:book:simple
```

## 🎓 Prochaines étapes

### Améliorations possibles

1. **Finaliser Paged.js** pour les fonctionnalités avancées
2. **Table des matières automatique** avec numéros de page
3. **Index** généré automatiquement
4. **Références croisées** (Voir Figure 2.3, etc.)
5. **Templates prédéfinis** :
   - Thèse académique
   - Rapport technique
   - Livre scientifique
   - Documentation

### Contribuer

Les styles sont dans `_print-book.css`. Pour proposer des améliorations :

1. Testez avec votre contenu
2. Modifiez le CSS
3. Générez le PDF
4. Partagez vos modifications !

## 📚 Ressources

### CSS Paged Media

- [W3C Spec](https://www.w3.org/TR/css-page-3/)
- [CSS Tricks Guide](https://css-tricks.com/css-paged-media-guide/)
- [Print CSS Documentation](https://www.smashingmagazine.com/2015/01/designing-for-print-with-css/)

### Paged.js

- [Documentation](https://pagedjs.org/documentation/)
- [Exemples](https://pagedjs.org/examples/)
- [W3C Paged Media](https://www.w3.org/TR/css-page-3/)

### Typographie de livre

- [Butterick's Practical Typography](https://practicaltypography.com/)
- [The Elements of Typographic Style](http://webtypography.net/)

## 💡 Cas d'usage

Ce système est idéal pour :

- 📘 **Thèses de doctorat**
- 📗 **Mémoires de master**
- 📕 **Rapports de recherche**
- 📙 **Documentation technique**
- 📓 **Livres blancs**
- 📔 **Livres auto-publiés**
- 📚 **Collections d'articles**

## 🎉 Résultat

Avec ce système, vous obtenez :

✅ **PDF prêt pour l'impression**
- Marges correctes pour reliure
- Typographie professionnelle
- Mise en page cohérente

✅ **Qualité éditoriale**
- Numérotation automatique
- Gestion des veuves/orphelines
- Césure propre

✅ **Workflow moderne**
- Écriture en MDX
- Build automatisé
- Un seul fichier source

---

**Créé avec ❤️ pour le Research Article Template**

*Profitez de votre nouveau système d'export PDF livre !* 📚✨

