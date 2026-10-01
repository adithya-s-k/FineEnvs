/**
 * Astro integration: generate /llms.txt at build time.
 *
 * Reads article.mdx, resolves chapter imports, strips JSX/components,
 * and outputs a clean Markdown file with structured metadata header
 * following the llms.txt convention (https://llmstxt.org/).
 */
import { promises as fs } from "node:fs";
import { resolve } from "node:path";

// ---------------------------------------------------------------------------
// MDX helpers (shared logic with export-latex.mjs)
// ---------------------------------------------------------------------------

async function readFile(filePath) {
  try {
    return await fs.readFile(filePath, "utf-8");
  } catch {
    return "";
  }
}

function stripQuotes(s) {
  return String(s || "")
    .replace(/^["']|["']$/g, "")
    .trim();
}

function extractFrontmatter(content) {
  const m = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!m) return { frontmatter: {}, body: content };

  const fm = {};
  const lines = m[1].split("\n");
  let key = null;
  let val = "";
  let multi = false;

  for (const line of lines) {
    if (line.match(/^[a-zA-Z_]\w*\s*:/) && !multi) {
      if (key) fm[key] = val.trim();
      const [k, ...rest] = line.split(":");
      key = k.trim();
      val = rest.join(":").trim();
      if (val === ">" || val === "|") {
        multi = true;
        val = "";
      }
    } else if (key) {
      val += (multi ? " " : "\n") + line;
    }
  }
  if (key) fm[key] = val.trim();

  return { frontmatter: fm, body: content.replace(m[0], "") };
}

function parseAuthors(raw) {
  if (!raw) return [];
  const names = [];
  const re = /name:\s*["']?([^"'\n]+)["']?/g;
  let hit;
  while ((hit = re.exec(raw)) !== null) names.push(hit[1].trim());
  if (names.length) return names;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function mdxToMarkdown(content) {
  let out = content;
  out = out.replace(/^import .+?;?\s*$/gm, "");
  out = out.replace(/<[A-Z][a-zA-Z0-9]*\s*\/>/g, "");
  out = out.replace(
    /<Sidenote>([\s\S]*?)<\/Sidenote>/g,
    (_, inner) => {
      const aside = inner.match(
        /<Fragment slot="aside">([\s\S]*?)<\/Fragment>/,
      );
      const main = inner
        .replace(/<Fragment slot="aside">[\s\S]*?<\/Fragment>/, "")
        .trim();
      return aside ? `${main}\n\n> ${aside[1].trim()}` : main;
    },
  );
  out = out.replace(
    /<Note[^>]*>([\s\S]*?)<\/Note>/g,
    (_, inner) => `\n> ${inner.trim()}\n`,
  );
  out = out.replace(/<(Wide|FullWidth)>([\s\S]*?)<\/\1>/g, "$2");
  out = out.replace(
    /<HtmlEmbed\s+src="([^"]*)"[^>]*\/>/g,
    "*[Interactive visualization: $1]*",
  );
  out = out.replace(
    /<Fragment[^>]*>([\s\S]*?)<\/Fragment>/g,
    "$1",
  );
  out = out.replace(
    /<[A-Z][a-zA-Z0-9]*[^>]*>([\s\S]*?)<\/[A-Z][a-zA-Z0-9]*>/g,
    "$1",
  );
  out = out.replace(/className="[^"]*"/g, "");
  out = out.replace(/<br\s*\/?>/g, "\n");
  out = out.replace(/<details>\s*([\s\S]*?)<\/details>/g, "$1");
  out = out.replace(/<summary>([\s\S]*?)<\/summary>/g, "**$1**\n");
  // Convert <a href="url">text</a> to [text](url)
  out = out.replace(
    /<a\s+href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g,
    "[$2]($1)",
  );
  // Strip remaining HTML tags but keep their text content
  out = out.replace(/<[^>]+>/g, "");
  // Remove JSX comments {/* ... */}
  out = out.replace(/\{\/\*[\s\S]*?\*\/\}/g, "");
  // Collapse multiple spaces and blank lines
  out = out.replace(/ {2,}/g, " ");
  out = out.replace(/\n{3,}/g, "\n\n");
  return out.trim();
}

async function resolveChapters(content, contentDir) {
  let out = content;
  const importRe =
    /import\s+(\w+)\s+from\s+["']\.\/chapters\/([^"']+)["'];?/g;
  const imports = new Map();
  let m;
  while ((m = importRe.exec(content)) !== null) {
    imports.set(m[1], m[2]);
  }
  out = out.replace(importRe, "");

  for (const [name, path] of imports) {
    const re = new RegExp(`<${name}\\s*\\/>`, "g");
    const raw = await readFile(resolve(contentDir, "chapters", path));
    const { body } = extractFrontmatter(raw);
    out = out.replace(re, mdxToMarkdown(body));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Build the llms.txt header from frontmatter
// ---------------------------------------------------------------------------

function buildHeader(fm) {
  const title = stripQuotes(
    (fm.title || "Untitled")
      .replace(/\\n/g, " ")
      .replace(/\n/g, " ")
      .replace(/ {2,}/g, " "),
  );

  const parts = [`# ${title}\n`];

  const desc = stripQuotes(fm.description || fm.subtitle || "");
  if (desc) parts.push(`> ${desc}\n`);

  const meta = [];
  const authors = parseAuthors(fm.authors);
  if (authors.length) meta.push(`- **Authors**: ${authors.join(", ")}`);
  if (fm.published) meta.push(`- **Published**: ${stripQuotes(fm.published)}`);
  if (fm.doi) meta.push(`- **DOI**: https://doi.org/${fm.doi}`);
  if (fm.template) meta.push(`- **Template**: ${stripQuotes(fm.template)}`);
  if (meta.length) parts.push(meta.join("\n") + "\n");

  parts.push("---\n");
  return parts.join("\n");
}

// ---------------------------------------------------------------------------
// Astro integration
// ---------------------------------------------------------------------------

export default function generateLlmsTxt() {
  return {
    name: "generate-llms-txt",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        const cwd = process.cwd();
        const contentDir = resolve(cwd, "src/content");
        const articleFile = resolve(contentDir, "article.mdx");

        let raw;
        try {
          raw = await readFile(articleFile);
        } catch {
          console.warn("[llms.txt] article.mdx not found, skipping");
          return;
        }
        if (!raw) {
          console.warn("[llms.txt] article.mdx is empty, skipping");
          return;
        }

        const { frontmatter, body } = extractFrontmatter(raw);
        const withChapters = await resolveChapters(body, contentDir);
        const markdown = mdxToMarkdown(withChapters);

        const header = buildHeader(frontmatter);
        const output = header + "\n" + markdown + "\n";

        const outPath = new URL("llms.txt", dir);
        await fs.writeFile(outPath, output, "utf-8");

        const kb = (Buffer.byteLength(output) / 1024).toFixed(1);
        console.log(`[llms.txt] Generated ${outPath.pathname} (${kb} KB)`);
      },
    },
  };
}
