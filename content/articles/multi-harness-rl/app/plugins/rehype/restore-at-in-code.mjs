// Rehype plugin to restore '@' inside code nodes after rehype-citation ran
export default function rehypeRestoreAtInCode() {
  return (tree) => {
    const restoreInNode = (node) => {
      if (!node || typeof node !== 'object') return;
      const isText = node.type === 'text';
      if (isText && typeof node.value === 'string' && node.value.includes('__AT_SENTINEL__')) {
        node.value = node.value.replace(/__AT_SENTINEL__/g, '@');
      }
      const isCodeEl = node.type === 'element' && node.tagName === 'code';
      const children = Array.isArray(node.children) ? node.children : [];
      if (isCodeEl && children.length) {
        children.forEach(restoreInNode);
        return;
      }
      children.forEach(restoreInNode);
    };
    restoreInNode(tree);
  };
}


