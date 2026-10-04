// The accessible label always holds the final count. Only the visual number
// animates, once per change, and detached views stop requesting frames.
export function countUp(node, target, format, from = 0) {
  node.textContent = format(target);
  if (from === target || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  node.textContent = format(from);
  const start = performance.now();
  const frame = (now) => {
    if (!node.isConnected) return;
    const t = Math.min(1, (now - start) / 750);
    node.textContent = format(Math.round(from + (target - from) * (1 - (1 - t) ** 3)));
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}
