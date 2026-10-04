// Asset paths are relative to the Space, never the explorer. Resolve only
// same-origin HTTPS paths; arbitrary URL schemes and foreign hosts are rejected.
export function spaceMedia(data, host) {
  if (!data || typeof data !== "object") return data;
  if (Array.isArray(data)) return data.map((v) => spaceMedia(v, host));
  const out = Object.fromEntries(Object.entries(data).map(([k, v]) => [k, spaceMedia(v, host)]));
  if (typeof out.asset_path === "string" && out.asset_path.startsWith("/assets/") && /^(audio|image|video)\//.test(out.mime || "")) {
    try {
      const base = new URL(host), asset = new URL(out.asset_path, base);
      if (base.protocol === "https:" && asset.origin === base.origin && !asset.username && !asset.password)
        out.media_url = asset.href;
    } catch { /* malformed source: keep its metadata visible */ }
  }
  return out;
}
