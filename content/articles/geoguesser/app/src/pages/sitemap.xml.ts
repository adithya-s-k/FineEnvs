/**
 * The sitemap, emitted at build time from `site`.
 *
 * Written by hand rather than by `@astrojs/sitemap`: the version that resolves
 * here reads a routes shape Astro 4.16 does not pass and throws as soon as
 * `site` is set, which is exactly when a sitemap becomes possible. This is two
 * pages, so an endpoint is less machinery than a patched dependency.
 */

import type { APIRoute } from "astro";

/** Every page worth indexing, with how often it is likely to change. */
const PAGES: { path: string; priority: string; changefreq: string }[] = [
  { path: "", priority: "1.0", changefreq: "weekly" },
  { path: "dataviz/", priority: "0.4", changefreq: "monthly" },
];

export const GET: APIRoute = ({ site }) => {
  const origin = (site ?? new URL("https://fineenvs-geoguesser-article.hf.space")).origin;
  const today = new Date().toISOString().slice(0, 10);

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${PAGES.map(
  (p) => `  <url>
    <loc>${origin}/${p.path}</loc>
    <lastmod>${today}</lastmod>
    <changefreq>${p.changefreq}</changefreq>
    <priority>${p.priority}</priority>
  </url>`,
).join("\n")}
</urlset>
`;

  return new Response(body, {
    headers: { "Content-Type": "application/xml; charset=utf-8" },
  });
};
