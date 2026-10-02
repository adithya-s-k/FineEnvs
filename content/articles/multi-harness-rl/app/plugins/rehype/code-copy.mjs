// Minimal rehype plugin to wrap code blocks with a copy button
// Exported as a standalone module to keep astro.config.mjs lean
export default function rehypeCodeCopy() {
  return (tree) => {
    // Walk the tree; lightweight visitor to find <pre><code>
    const visit = (node, parent) => {
      if (!node || typeof node !== 'object') return;
      const children = Array.isArray(node.children) ? node.children : [];
      if (node.tagName === 'pre' && children.some(c => c.tagName === 'code')) {
        // Find code child
        const code = children.find(c => c.tagName === 'code');
        // Determine if single-line block: prefer Shiki lines, then text content
        const countLinesFromShiki = () => {
          const isLineEl = (el) => el && el.type === 'element' && el.tagName === 'span' && Array.isArray(el.properties?.className) && el.properties.className.includes('line');
          const hasNonWhitespaceText = (node) => {
            if (!node) return false;
            if (node.type === 'text') return /\S/.test(String(node.value || ''));
            const kids = Array.isArray(node.children) ? node.children : [];
            return kids.some(hasNonWhitespaceText);
          };
          const collectLines = (node, acc) => {
            if (!node || typeof node !== 'object') return;
            if (isLineEl(node)) acc.push(node);
            const kids = Array.isArray(node.children) ? node.children : [];
            kids.forEach((k) => collectLines(k, acc));
          };
          const lines = [];
          collectLines(code, lines);
          const nonEmpty = lines.filter((ln) => hasNonWhitespaceText(ln)).length;
          return nonEmpty || 0;
        };
        const countLinesFromText = () => {
          // Parse raw text content of the <code> node including nested spans
          const extractText = (node) => {
            if (!node) return '';
            if (node.type === 'text') return String(node.value || '');
            const kids = Array.isArray(node.children) ? node.children : [];
            return kids.map(extractText).join('');
          };
          const raw = extractText(code);
          if (!raw || !/\S/.test(raw)) return 0;
          return raw.split('\n').filter(line => /\S/.test(line)).length;
        };
        const lines = countLinesFromShiki() || countLinesFromText();
        const isSingleLine = lines <= 1;
        // Also treat code blocks shorter than a threshold as single-line (defensive)
        if (!isSingleLine) {
          const approxChars = (() => {
            const extract = (n) => Array.isArray(n?.children) ? n.children.map(extract).join('') : (n?.type === 'text' ? String(n.value||'') : '');
            return extract(code).length;
          })();
          if (approxChars < 6) {
            node.__forceSingle = true;
          }
        }
        // Replace <pre> with wrapper div.code-card containing button + pre
        const wrapper = {
          type: 'element',
          tagName: 'div',
          properties: { className: ['code-card'].concat((isSingleLine || node.__forceSingle) ? ['no-copy'] : []) },
          children: (isSingleLine || node.__forceSingle) ? [ node ] : [
            {
              type: 'element',
              tagName: 'button',
              properties: { className: ['code-copy', 'button--ghost'], type: 'button', 'aria-label': 'Copy code' },
              children: [
                {
                  type: 'element',
                  tagName: 'svg',
                  properties: { viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false' },
                  children: [
                    { type: 'element', tagName: 'path', properties: { d: 'M16 1H4c-1.1 0-2 .9-2 2v12h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z' }, children: [] }
                  ]
                }
              ]
            },
            node
          ]
        };
        if (parent && Array.isArray(parent.children)) {
          const idx = parent.children.indexOf(node);
          if (idx !== -1) parent.children[idx] = wrapper;
        }
        return; // don't visit nested
      }
      children.forEach((c) => visit(c, node));
    };
    visit(tree, null);
  };
}




