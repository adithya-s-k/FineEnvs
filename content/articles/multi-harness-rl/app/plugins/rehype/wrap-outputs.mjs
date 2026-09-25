// Wrap plain-text content inside <section class="code-outputs"> into a <pre>
export default function rehypeWrapOutput() {
  return (tree) => {
    const isWhitespace = (value) => typeof value === 'string' && !/\S/.test(value);
    const extractText = (node) => {
      if (!node) return '';
      if (node.type === 'text') return String(node.value || '');
      const kids = Array.isArray(node.children) ? node.children : [];
      return kids.map(extractText).join('');
    };
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      const children = Array.isArray(node.children) ? node.children : [];
      if (node.type === 'element' && node.tagName === 'section') {
        const className = node.properties?.className || [];
        const classes = Array.isArray(className) ? className : [className].filter(Boolean);
        if (classes.includes('code-output')) {
          const meaningful = children.filter((c) => !(c.type === 'text' && isWhitespace(c.value)));
          if (meaningful.length === 1) {
            const only = meaningful[0];
            const isPlainParagraph = only.type === 'element' && only.tagName === 'p' && (only.children || []).every((c) => c.type === 'text');
            const isPlainText = only.type === 'text';
            if (isPlainParagraph || isPlainText) {
              const text = isPlainText ? String(only.value || '') : extractText(only);
              node.children = [
                { type: 'element', tagName: 'pre', properties: {}, children: [ { type: 'text', value: text } ] }
              ];
            }
          }
        }
      }
      children.forEach(visit);
    };
    visit(tree);
  };
}


