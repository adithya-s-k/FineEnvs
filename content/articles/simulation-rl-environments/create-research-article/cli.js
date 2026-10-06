#!/usr/bin/env node

import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { createInterface } from "readline";

// =============================================================================
// Terminal formatting (zero deps)
// =============================================================================

const fmt = {
  bold: (s) => `\x1b[1m${s}\x1b[22m`,
  green: (s) => `\x1b[32m${s}\x1b[39m`,
  cyan: (s) => `\x1b[36m${s}\x1b[39m`,
  red: (s) => `\x1b[31m${s}\x1b[39m`,
  dim: (s) => `\x1b[2m${s}\x1b[22m`,
  yellow: (s) => `\x1b[33m${s}\x1b[39m`,
};

const TEMPLATE_REPO =
  "https://huggingface.co/spaces/tfrere/research-article-template";

// =============================================================================
// Prompt helpers
// =============================================================================

function createPrompt() {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  const ask = (question, defaultValue) =>
    new Promise((resolve) => {
      const hint = defaultValue ? ` ${fmt.dim(`[${defaultValue}]`)}` : "";
      rl.question(`  ${question}${hint}: `, (answer) => {
        resolve(answer.trim() || defaultValue || "");
      });
    });

  const select = (question, options) =>
    new Promise((resolve) => {
      console.log(`\n  ${question}\n`);
      options.forEach((opt, i) => {
        const marker = i === 0 ? fmt.cyan(">") : " ";
        const hint = opt.hint ? fmt.dim(` - ${opt.hint}`) : "";
        console.log(`  ${marker} ${fmt.bold(String(i + 1))}. ${opt.label}${hint}`);
      });
      const tryRead = () => {
        rl.question(`\n  Choice ${fmt.dim(`[1-${options.length}]`)}: `, (answer) => {
          const idx = parseInt(answer, 10) - 1;
          if (idx >= 0 && idx < options.length) {
            resolve(options[idx].value);
          } else if (answer.trim() === "") {
            resolve(options[0].value);
          } else {
            tryRead();
          }
        });
      };
      tryRead();
    });

  const confirm = (question, defaultYes = true) =>
    new Promise((resolve) => {
      const hint = defaultYes ? fmt.dim(" [Y/n]") : fmt.dim(" [y/N]");
      rl.question(`  ${question}${hint}: `, (answer) => {
        const a = answer.trim().toLowerCase();
        if (a === "") resolve(defaultYes);
        else resolve(a === "y" || a === "yes");
      });
    });

  return { rl, ask, select, confirm, close: () => rl.close() };
}

// =============================================================================
// Content generators
// =============================================================================

function todayFormatted() {
  const d = new Date();
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[d.getMonth()]}. ${String(d.getDate()).padStart(2, "0")}, ${d.getFullYear()}`;
}

function generateArticleMdx(cfg) {
  const authorLine = cfg.authorUrl
    ? `  - name: "${cfg.authorName}"\n    url: "${cfg.authorUrl}"\n    affiliations: [1]`
    : `  - name: "${cfg.authorName}"\n    affiliations: [1]`;

  const affiliationLine = cfg.affiliationUrl
    ? `  - name: "${cfg.affiliationName}"\n    url: "${cfg.affiliationUrl}"`
    : `  - name: "${cfg.affiliationName}"`;

  const isArticle = cfg.layout === "article";

  return `---
title: "${cfg.title}"
description: "${cfg.description}"
authors:
${authorLine}
affiliations:
${affiliationLine}
published: "${todayFormatted()}"
template: "${cfg.layout}"
tableOfContentsAutoCollapse: ${isArticle}
showPdf: ${isArticle}
---

import Introduction from "./chapters/introduction.mdx";

<Introduction />
`;
}

function generateIntroChapter(layout) {
  if (layout === "article") {
    return `## Abstract

Write your abstract here. This section should briefly summarize the key points of your research.

## Introduction

Start writing your article here. You can use all the features of the template:

- **Markdown** for text formatting
- **KaTeX** for math: $E = mc^2$
- **Citations** using BibTeX references [@example2025]
- **D3.js embeds** for interactive visualizations
- **Images** with automatic figure numbering

See the [template documentation](https://huggingface.co/spaces/tfrere/research-article-template) for more details.
`;
  }

  return `## Introduction

Start writing your paper here. You can use all the features of the template:

- **Markdown** for text formatting
- **KaTeX** for math: $E = mc^2$
- **Citations** using BibTeX references [@example2025]
- **D3.js embeds** for interactive visualizations
- **Images** with automatic figure numbering

See the [template documentation](https://huggingface.co/spaces/tfrere/research-article-template) for more details.
`;
}

function generateBibliography() {
  return `@article{example2025,
  title={Example Reference},
  author={Author, Example},
  journal={Journal of Examples},
  year={2025}
}
`;
}

function spaceIdToUrl(spaceId) {
  return `https://${spaceId.replace("/", "-").toLowerCase()}.hf.space`;
}

function generateReadme(name, layout, title, spaceId) {
  const layoutLabel = layout === "article" ? "Research Article" : "Research Paper";
  const safeTitle = title || name;
  const thumbLine = spaceId
    ? `\nthumbnail: ${spaceIdToUrl(spaceId)}/thumb.auto.jpg`
    : "";
  return `---
title: "${safeTitle}"
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
header: mini
pinned: false
tags:
  - research-article-template${thumbLine}
---

# ${name}

A ${layoutLabel.toLowerCase()} built with [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template).

## Quick start

\`\`\`bash
cd app
npm run dev           # dev server at http://localhost:4321
\`\`\`

## Deploy to Hugging Face

\`\`\`bash
# 1. Create a Space at huggingface.co/new-space (select Docker SDK)
# 2. Push to it:
git remote add space git@hf.co:spaces/<your-username>/<your-space>
git push space main
\`\`\`

That's it. The Dockerfile and nginx config are included.

## Edit your content

| File | What |
|------|------|
| \`app/src/content/article.mdx\` | Main article (metadata + chapter imports) |
| \`app/src/content/chapters/\` | Your chapters (one .mdx per section) |
| \`app/src/content/bibliography.bib\` | BibTeX references |
| \`app/src/content/embeds/\` | D3.js HTML visualizations |
| \`app/src/content/assets/data/\` | CSV/JSON data files |
| \`app/src/content/assets/image/\` | Images |

## Commands

| Command | Description |
|---------|-------------|
| \`npm run dev\` | Dev server |
| \`npm run build\` | Production build |
| \`npm run export:pdf\` | Export as PDF |
| \`npm run export:latex\` | Export as LaTeX |
| \`npm run sync:template\` | Pull latest template updates |

## License

CC-BY-4.0
`;
}

// =============================================================================
// Cleanup - remove demo content from the cloned template
// =============================================================================

const CLEANUP_DIRS = [
  ".git",
  ".cursor/rules",
  ".vscode",
  ".playwright-mcp",
  "app/.claude",
  "app/venv",
  "app/screenshots",
  "app/src/content/chapters/demo",
  "app/src/content/embeds/demo",
  "app/src/content/embeds/arxiv",
  "app/src/content/embeds/smol-playbook",
  "app/src/content/embeds/typography",
  "app/src/content/assets/audio",
  "app/src/content/assets/sprites",
  "app/src/components/demo",
];

const CLEANUP_FILES = [
  "AUDIT-2026.md",
  "ROADMAP-2026.md",
  "CHANGELOG.md",
  "CONTRIBUTING.md",
  "prototype-project-page.html",
  "app/scripts/fetch-hf-citations.py",
];

const CLEANUP_GLOBS_IN = {
  "app/src/content/embeds": (name) =>
    name !== "banner.html" && name.endsWith(".html"),
  "app/public": (name) => name.endsWith(".pdf"),
};

function cleanupProject(dir) {
  // Remove known directories
  for (const p of CLEANUP_DIRS) {
    const full = path.join(dir, p);
    if (fs.existsSync(full)) fs.rmSync(full, { recursive: true, force: true });
  }

  // Remove known files
  for (const p of CLEANUP_FILES) {
    const full = path.join(dir, p);
    if (fs.existsSync(full)) fs.rmSync(full, { force: true });
  }

  // Remove matching files inside specific directories
  for (const [relDir, matcher] of Object.entries(CLEANUP_GLOBS_IN)) {
    const full = path.join(dir, relDir);
    if (!fs.existsSync(full)) continue;
    for (const entry of fs.readdirSync(full)) {
      if (matcher(entry)) {
        fs.rmSync(path.join(full, entry), { recursive: true, force: true });
      }
    }
  }

  // Clear data dir but keep it
  const dataDir = path.join(dir, "app/src/content/assets/data");
  if (fs.existsSync(dataDir)) {
    for (const f of fs.readdirSync(dataDir)) {
      if (f === ".gitkeep") continue;
      fs.rmSync(path.join(dataDir, f), { recursive: true, force: true });
    }
    fs.writeFileSync(path.join(dataDir, ".gitkeep"), "");
  }

  // Clear ALL images (LFS pointers are not valid image files)
  const imageDir = path.join(dir, "app/src/content/assets/image");
  if (fs.existsSync(imageDir)) {
    for (const f of fs.readdirSync(imageDir)) {
      fs.rmSync(path.join(imageDir, f), { recursive: true, force: true });
    }
    // Generate a minimal real placeholder PNG (1x1 white pixel)
    // so the project has a working example image
    const pngData = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB" +
        "Nl7BcQAAAABJRU5ErkJggg==",
      "base64",
    );
    fs.writeFileSync(path.join(imageDir, "placeholder.png"), pngData);
  }

  // Remove all auto-generated and LFS thumbnails from public
  const publicDir = path.join(dir, "app/public");
  if (fs.existsSync(publicDir)) {
    for (const f of fs.readdirSync(publicDir)) {
      if (
        f.endsWith(".pdf") ||
        f.startsWith("thumb.auto") ||
        f === "thumb.png"
      ) {
        fs.rmSync(path.join(publicDir, f), { force: true });
      }
    }
  }

  // Remove LFS-tracked .gitattributes entries that won't apply to new project
  const gitattrs = path.join(dir, ".gitattributes");
  if (fs.existsSync(gitattrs)) {
    fs.rmSync(gitattrs, { force: true });
  }
}

// =============================================================================
// Main
// =============================================================================

async function main() {
  const args = process.argv.slice(2);

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
  ${fmt.bold("create-research-article")}

  Scaffold a new research paper or article.

  ${fmt.bold("Usage:")}
    npx create-research-article [project-name] [options]

  ${fmt.bold("Options:")}
    --template=article|paper   Skip template selection prompt
    --help, -h                 Show this help message
`);
    process.exit(0);
  }

  const positionalName = args.find((a) => !a.startsWith("-"));
  const templateFlag = args
    .find((a) => a.startsWith("--template="))
    ?.split("=")[1]
    ?.toLowerCase();

  console.log("");
  console.log(
    `  ${fmt.bold("create-research-article")} ${fmt.dim("v0.2.0")}`,
  );
  console.log(fmt.dim("  Scaffold a new research paper or article\n"));

  // Check git is available
  try {
    execSync("git --version", { stdio: "ignore" });
  } catch {
    console.error(fmt.red("  Error: git is required but not installed.\n"));
    process.exit(1);
  }

  const prompt = createPrompt();

  try {
    // 1. Project name
    const name =
      positionalName ||
      (await prompt.ask("Project name", "my-research-article"));

    const targetDir = path.resolve(process.cwd(), name);
    if (fs.existsSync(targetDir)) {
      console.error(
        fmt.red(`\n  Error: directory "${name}" already exists.\n`),
      );
      prompt.close();
      process.exit(1);
    }

    // 2. Layout
    let layout;
    if (templateFlag && ["article", "paper"].includes(templateFlag)) {
      layout = templateFlag;
      console.log(fmt.cyan(`  Using template: ${layout}`));
    } else {
      if (templateFlag) {
        console.log(fmt.yellow(`  Warning: unknown template "${templateFlag}", showing picker.\n`));
      }
      layout = await prompt.select("Choose a layout:", [
        {
          value: "article",
          label: "Research Article",
          hint: "Full layout with banner, TOC, DOI, citations, PDF export",
        },
        {
          value: "paper",
          label: "Research Paper",
          hint: "Lighter single-column layout, blog-friendly",
        },
      ]);
    }

    // 3. Metadata
    console.log("");
    const title = await prompt.ask("Title", "My Research Article");
    const description = await prompt.ask("Short description", "");
    const authorName = await prompt.ask("Author name", "");
    const authorUrl = await prompt.ask("Author URL", "");
    const affiliationName = await prompt.ask("Affiliation", "");
    const affiliationUrl = await prompt.ask("Affiliation URL", "");

    prompt.close();

    // 4. Clone
    console.log("");
    console.log(fmt.cyan("  Cloning template..."));
    execSync(
      `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${TEMPLATE_REPO}" "${targetDir}"`,
      { stdio: "pipe" },
    );

    // 5. Cleanup
    console.log(fmt.cyan("  Cleaning up demo content..."));
    cleanupProject(targetDir);

    // 6. Generate content
    console.log(fmt.cyan("  Generating project files..."));

    fs.writeFileSync(
      path.join(targetDir, "app/src/content/article.mdx"),
      generateArticleMdx({
        layout,
        title,
        description,
        authorName,
        authorUrl,
        affiliationName,
        affiliationUrl,
      }),
    );

    // Rename chapter: remove old, create new
    const chaptersDir = path.join(targetDir, "app/src/content/chapters");
    if (fs.existsSync(chaptersDir)) {
      for (const f of fs.readdirSync(chaptersDir)) {
        fs.rmSync(path.join(chaptersDir, f), { recursive: true, force: true });
      }
    } else {
      fs.mkdirSync(chaptersDir, { recursive: true });
    }
    fs.writeFileSync(
      path.join(chaptersDir, "introduction.mdx"),
      generateIntroChapter(layout),
    );

    fs.writeFileSync(
      path.join(targetDir, "app/src/content/bibliography.bib"),
      generateBibliography(),
    );

    fs.writeFileSync(path.join(targetDir, "README.md"), generateReadme(name, layout, title));

    // 7. Init git
    execSync("git init", { cwd: targetDir, stdio: "pipe" });
    execSync("git add -A", { cwd: targetDir, stdio: "pipe" });
    execSync('git commit -m "Initial scaffold from create-research-article"', {
      cwd: targetDir,
      stdio: "pipe",
    });

    // 8. Install dependencies
    console.log(fmt.cyan("  Installing dependencies (this may take a moment)..."));
    try {
      execSync("npm install", {
        cwd: path.join(targetDir, "app"),
        stdio: "pipe",
      });
      console.log(fmt.green("  Dependencies installed."));
    } catch {
      console.log(
        fmt.yellow(
          "  Warning: npm install failed. Run it manually: cd app && npm install",
        ),
      );
    }

    // 9. Optional: deploy to Hugging Face
    let deployed = false;
    const hasHfCli = (() => {
      try {
        execSync("huggingface-cli whoami", { stdio: "pipe" });
        return true;
      } catch {
        return false;
      }
    })();

    if (hasHfCli) {
      const prompt2 = createPrompt();
      console.log("");
      const wantDeploy = await prompt2.confirm(
        "Deploy to Hugging Face Spaces?",
        true,
      );

      if (wantDeploy) {
        const hfUser = execSync("huggingface-cli whoami", { encoding: "utf8" })
          .split("\n")[0]
          .trim();
        const defaultSpace = `${hfUser}/${name}`;
        const spaceId = await prompt2.ask("Space ID", defaultSpace);
        prompt2.close();

        console.log(fmt.cyan(`  Creating Space ${spaceId}...`));
        try {
          execSync(
            `huggingface-cli repo create "${spaceId}" --repo-type space --space_sdk docker --exist-ok`,
            { stdio: "pipe" },
          );

          execSync(
            `git remote add space "git@hf.co:spaces/${spaceId}"`,
            { cwd: targetDir, stdio: "pipe" },
          );

          // Update README thumbnail URL to point to the Space's own auto-generated thumb
          const readmePath = path.join(targetDir, "README.md");
          const readmeContent = fs.readFileSync(readmePath, "utf8");
          const thumbUrl = `${spaceIdToUrl(spaceId)}/thumb.auto.jpg`;
          if (!readmeContent.includes("thumbnail:")) {
            const updated = readmeContent.replace(
              /^(tags:\n(?:\s+-[^\n]+\n?)+)/m,
              `$1thumbnail: ${thumbUrl}\n`,
            );
            fs.writeFileSync(readmePath, updated);
          } else {
            const updated = readmeContent.replace(
              /^thumbnail:.*$/m,
              `thumbnail: ${thumbUrl}`,
            );
            fs.writeFileSync(readmePath, updated);
          }
          execSync("git add README.md && git commit --amend --no-edit", {
            cwd: targetDir,
            stdio: "pipe",
          });

          console.log(fmt.cyan("  Pushing to Hugging Face..."));
          execSync("git push space main", {
            cwd: targetDir,
            stdio: "pipe",
          });

          deployed = true;
          const spaceUrl = `https://huggingface.co/spaces/${spaceId}`;
          console.log(fmt.green(`  Deployed! ${spaceUrl}`));
        } catch (e) {
          console.log(
            fmt.yellow(`  Warning: deploy failed. ${e.message}`),
          );
          console.log(
            fmt.dim("  You can deploy manually later with: git push space main"),
          );
        }
      } else {
        prompt2.close();
      }
    }

    // 10. Success
    console.log("");
    console.log(fmt.green(fmt.bold("  Done! Project created successfully.")));
    console.log("");
    console.log(`  ${fmt.bold("Next steps:")}`);
    console.log("");
    console.log(fmt.cyan(`    cd ${name}/app`));
    console.log(fmt.cyan("    npm run dev"));
    if (!deployed && !hasHfCli) {
      console.log("");
      console.log(`  ${fmt.dim("To deploy to Hugging Face:")}`);
      console.log(
        fmt.dim(
          "    pip install huggingface_hub && huggingface-cli login",
        ),
      );
      console.log(
        fmt.dim(
          `    huggingface-cli repo create ${name} --repo-type space --space_sdk docker`,
        ),
      );
      console.log(
        fmt.dim(
          `    cd ${name} && git remote add space git@hf.co:spaces/<user>/${name} && git push space main`,
        ),
      );
    } else if (!deployed) {
      console.log("");
      console.log(`  ${fmt.dim("To deploy later:")}`);
      console.log(
        fmt.dim(
          `    cd ${name} && git remote add space git@hf.co:spaces/<user>/${name} && git push space main`,
        ),
      );
    }
    console.log("");
  } catch (err) {
    prompt.close();
    console.error(fmt.red(`\n  Error: ${err.message}\n`));
    process.exit(1);
  }
}

main();
