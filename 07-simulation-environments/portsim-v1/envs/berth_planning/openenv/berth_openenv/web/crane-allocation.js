// Rail allocation is independent of rendering: preserve the requested crane count,
// keep gantries ordered, and park spare cranes with their booms raised.
export function craneTargets(moored, N, lay) {
  const MIN = 29;
  const lo = (lay.railX0 ?? lay.X0) + 16, hi = (lay.railX1 ?? lay.X1) - 16;
  const out = [];
  const clear = x => x >= lo && x <= hi && out.every(t => Math.abs(t.x - x) >= MIN);
  const ships = [...moored].sort((a, b) => a.x1 - b.x1);
  for (const s of ships) {
    const want = s.want == null ? Math.max(1, Math.round((s.x2 - s.x1) / 95)) : Math.max(0, s.want);
    const available = s.bays?.length ? s.bays : Array.from({ length: Math.max(1, Math.floor((s.x2 - s.x1) / MIN)) }, (_, k) => ({ x: s.x1 + 15 + k * MIN, top: s.cargoTop }));
    for (let k = 0; k < want && out.length < N; k++) {
      const ideal = s.x1 + (k + 0.5) * (s.x2 - s.x1) / want;
      const bay = available.filter(b => clear(b.x)).sort((a, b) => Math.abs(a.x - ideal) - Math.abs(b.x - ideal))[0];
      if (!bay) continue;
      out.push({ x: bay.x, working: true, zNear: s.zNear, zFar: s.zFar, deckY: s.deckY, cargoTop: bay.top, ship: s.id });
    }
  }
  while (out.length < N) {
    const positions = [lo - MIN, ...out.map(t => t.x).sort((a, b) => a - b), hi + MIN];
    let x = null, room = -1;
    for (let i = 1; i < positions.length; i++) {
      const left = Math.max(lo, positions[i - 1] + MIN), right = Math.min(hi, positions[i] - MIN);
      if (right >= left && right - left > room) { room = right - left; x = (left + right) / 2; }
    }
    if (x == null) {
      // A malformed/custom fleet larger than the rail capacity must not stack gantries.
      out.push({ x: hi + MIN * (out.length + 1), working: false, hidden: true });
    } else out.push({ x, working: false });
  }
  return out.sort((a, b) => a.x - b.x);
}
