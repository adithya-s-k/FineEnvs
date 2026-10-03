import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import svelte from '@astrojs/svelte';
import mermaid from 'astro-mermaid';
import compressor from 'astro-compressor';
import generateLlmsTxt from './plugins/astro/generate-llms-txt.mjs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkFootnotes from 'remark-footnotes';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeCitation from 'rehype-citation';
import rehypeCodeCopy from './plugins/rehype/code-copy.mjs';
import rehypeReferencesAndFootnotes from './plugins/rehype/post-citation.mjs';
import remarkIgnoreCitationsInCode from './plugins/remark/ignore-citations-in-code.mjs';
import remarkUnwrapCitationLinks from './plugins/remark/unwrap-citation-links.mjs';
import remarkDirective from 'remark-directive';
import remarkOutputContainer from './plugins/remark/output-container.mjs';
import rehypeRestoreAtInCode from './plugins/rehype/restore-at-in-code.mjs';
import rehypeWrapTables from './plugins/rehype/wrap-tables.mjs';
import rehypeWrapOutput from './plugins/rehype/wrap-outputs.mjs';
// Built-in Shiki (dual themes) — no rehype-pretty-code

// Plugins moved to app/plugins/*

// The absolute origin every canonical, og:url, og:image and sitemap entry is
// built from. It has to resolve at *build* time, and a Space's Docker build does
// not always carry SPACE_ID, which is how this article shipped with
// `<link rel="canonical" href="http://localhost:4321/">` on the live page: a
// canonical pointing at localhost, and social cards whose image cannot load.
//
// So: an explicit PUBLIC_SITE_URL wins, SPACE_ID is the convenience, and the
// deployed URL is the fallback rather than `undefined`.
const spaceId = process.env.SPACE_ID; // e.g. "FineEnvs/geoguesser-article"
const siteUrl =
  process.env.PUBLIC_SITE_URL ||
  (spaceId ? `https://${spaceId.replace('/', '-').toLowerCase()}.hf.space` : null) ||
  'https://fineenvs-geoguesser-article.hf.space';

export default defineConfig({
  site: siteUrl,
  // No `sitemap()` integration: the version resolved here reads a routes shape
  // Astro 4.16 does not pass and throws in `astro:build:done` the moment `site`
  // is set. `src/pages/sitemap.xml.ts` emits the same file from the same origin.
  output: 'static',
  integrations: [
    mermaid({ theme: 'neutral', autoTheme: true }),
    mdx(),
    svelte(),
    generateLlmsTxt(),
    // Precompress output with Gzip only (Brotli disabled due to server module mismatch)
    compressor({ brotli: false, gzip: true })
  ],
  devToolbar: {
    enabled: false
  },
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      },
      defaultColor: false,
      wrap: false,
      langAlias: {
        // Map MDX fences to TSX for better JSX tokenization
        mdx: 'tsx'
      }
    },
    remarkPlugins: [
      remarkUnwrapCitationLinks,
      remarkIgnoreCitationsInCode,
      remarkMath,
      [remarkFootnotes, { inlineNotes: true }],
      remarkDirective,
      remarkOutputContainer
    ],
    rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      [rehypeKatex, {
        trust: true,
      }],
      [rehypeCitation, {
        bibliography: 'src/content/bibliography.bib',
        linkCitations: true,
        csl: "apa",
        noCite: false,
        suppressBibliography: false,
      }],
      rehypeReferencesAndFootnotes,
      rehypeRestoreAtInCode,
      rehypeCodeCopy,
      rehypeWrapOutput,
      rehypeWrapTables
    ]
  }
});


