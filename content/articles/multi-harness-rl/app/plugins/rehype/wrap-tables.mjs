// rehype plugin: wrap bare <table> elements in a <div class="table-scroll"> container
// so that tables stay width:100% while enabling horizontal scroll when content overflows

export default function rehypeWrapTables() {
  return (tree) => {
    const isElement = (n) => n && typeof n === 'object' && n.type === 'element';
    const getChildren = (n) => (Array.isArray(n?.children) ? n.children : []);

    const walk = (node, parent, fn) => {
      if (!node || typeof node !== 'object') return;
      fn && fn(node, parent);
      const kids = getChildren(node);
      for (const child of kids) walk(child, node, fn);
    };

    const ensureArray = (v) => (Array.isArray(v) ? v : v != null ? [v] : []);
    const hasClass = (el, name) => ensureArray(el?.properties?.className).map(String).includes(name);

    const wrapTable = (tableNode, parent) => {
      if (!parent || !Array.isArray(parent.children)) return;
      // Don't double-wrap if already inside .table-scroll
      if (parent.tagName === 'div' && hasClass(parent, 'table-scroll')) return;

      const wrapper = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-scroll'] },
        children: [tableNode]
      };

      const idx = parent.children.indexOf(tableNode);
      if (idx >= 0) parent.children.splice(idx, 1, wrapper);
    };

    walk(tree, null, (node, parent) => {
      if (!isElement(node)) return;
      if (node.tagName !== 'table') return;
      wrapTable(node, parent);
    });
  };
}


