// Remark plugin to ignore citations inside code (block and inline)
export default function remarkIgnoreCitationsInCode() {
  return (tree) => {
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      const type = node.type;
      if (type === 'code' || type === 'inlineCode') {
        if (typeof node.value === 'string' && node.value.includes('@')) {
          // Use a sentinel to avoid rehype-citation, will be restored later in rehype
          node.value = node.value.replace(/@/g, '__AT_SENTINEL__');
        }
        return; // do not traverse into code
      }
      const children = Array.isArray(node.children) ? node.children : [];
      children.forEach(visit);
    };
    visit(tree);
  };
}


