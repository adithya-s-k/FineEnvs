#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import { resolve, dirname, basename, extname } from 'node:path';
import process from 'node:process';

async function run(command, args = [], options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', shell: false, ...options });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolvePromise(undefined);
      else reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

function parseArgs(argv) {
  const out = {};
  for (const arg of argv.slice(2)) {
    if (!arg.startsWith('--')) continue;
    const [k, v] = arg.replace(/^--/, '').split('=');
    out[k] = v === undefined ? true : v;
  }
  return out;
}

function slugify(text) {
  return String(text || '')
    .normalize('NFKD')
    .replace(/\p{Diacritic}+/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 120) || 'article';
}

async function checkPandocInstalled() {
  try {
    await run('pandoc', ['--version'], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

async function readMdxFile(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return content;
  } catch (error) {
    console.warn(`Warning: Could not read ${filePath}:`, error.message);
    return '';
  }
}

function extractFrontmatter(content) {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/);
  if (!frontmatterMatch) return { frontmatter: {}, content };

  const frontmatterText = frontmatterMatch[1];
  const contentWithoutFrontmatter = content.replace(frontmatterMatch[0], '');

  // More robust YAML parsing that handles complex structures
  const frontmatter = {};
  const lines = frontmatterText.split('\n');
  let currentKey = null;
  let currentValue = '';
  let inMultiLineValue = false;
  let multiLineOperator = null; // '>' or '|'

  for (const line of lines) {
    // Check if this is a new key
    if (line.match(/^[a-zA-Z_][a-zA-Z0-9_]*\s*:/) && !inMultiLineValue) {
      // Save previous key if exists
      if (currentKey) {
        frontmatter[currentKey] = currentValue.trim();
      }

      const [key, ...valueParts] = line.split(':');
      currentKey = key.trim();
      currentValue = valueParts.join(':').trim();

      // Check for multi-line operators
      if (currentValue.endsWith('>') || currentValue.endsWith('|')) {
        multiLineOperator = currentValue.slice(-1);
        currentValue = currentValue.slice(0, -1).trim();
        inMultiLineValue = true;
      } else if (currentValue) {
        inMultiLineValue = false;
      } else {
        inMultiLineValue = true;
      }
    } else if (currentKey && (inMultiLineValue || line.match(/^\s/))) {
      // Continuation line or nested content
      if (inMultiLineValue) {
        if (line.trim() === '' && multiLineOperator === '>') {
          // Empty line in folded style should become space
          currentValue += ' ';
        } else {
          const lineContent = line.startsWith(' ') ? line : ' ' + line;
          currentValue += lineContent;
        }
      } else {
        currentValue += '\n' + line;
      }
    }
  }

  // Save the last key
  if (currentKey) {
    frontmatter[currentKey] = currentValue.trim();
  }

  return { frontmatter, content: contentWithoutFrontmatter };
}

function cleanMdxToMarkdown(content) {
  // Remove import statements
  content = content.replace(/^import .+?;?\s*$/gm, '');

  // Remove JSX component calls like <ComponentName />
  content = content.replace(/<[A-Z][a-zA-Z0-9]*\s*\/>/g, '');

  // Convert JSX components to simpler markdown
  // Handle Sidenote components specially
  content = content.replace(/<Sidenote>([\s\S]*?)<\/Sidenote>/g, (match, innerContent) => {
    // Extract main content and aside content
    const asideMatch = innerContent.match(/<Fragment slot="aside">([\s\S]*?)<\/Fragment>/);
    const mainContent = innerContent.replace(/<Fragment slot="aside">[\s\S]*?<\/Fragment>/, '').trim();
    const asideContent = asideMatch ? asideMatch[1].trim() : '';

    let result = mainContent;
    if (asideContent) {
      result += `\n\n> **Note:** ${asideContent}`;
    }
    return result;
  });

  // Handle Note components
  content = content.replace(/<Note[^>]*>([\s\S]*?)<\/Note>/g, (match, innerContent) => {
    return `\n> **Note:** ${innerContent.trim()}\n`;
  });

  // Handle Wide and FullWidth components
  content = content.replace(/<(Wide|FullWidth)>([\s\S]*?)<\/\1>/g, '$2');

  // Handle HtmlEmbed components (convert to simple text)
  content = content.replace(/<HtmlEmbed[^>]*\/>/g, '*[Interactive content not available in LaTeX]*');

  // Remove remaining JSX fragments
  content = content.replace(/<Fragment[^>]*>([\s\S]*?)<\/Fragment>/g, '$1');
  content = content.replace(/<[A-Z][a-zA-Z0-9]*[^>]*>([\s\S]*?)<\/[A-Z][a-zA-Z0-9]*>/g, '$1');

  // Clean up className attributes
  content = content.replace(/className="[^"]*"/g, '');

  // Clean up extra whitespace
  content = content.replace(/\n{3,}/g, '\n\n');

  // Clean up characters that might cause YAML parsing issues
  // Remove any potential YAML-style markers that might interfere
  content = content.replace(/^---$/gm, '');
  content = content.replace(/^\s*&\s+/gm, ''); // Remove YAML aliases

  return content.trim();
}

async function processChapterImports(content, contentDir) {
  let processedContent = content;

  // First, extract all import statements and their corresponding component calls
  const importPattern = /import\s+(\w+)\s+from\s+["']\.\/chapters\/([^"']+)["'];?/g;
  const imports = new Map();
  let match;

  // Collect all imports
  while ((match = importPattern.exec(content)) !== null) {
    const [fullImport, componentName, chapterPath] = match;
    imports.set(componentName, { path: chapterPath, importStatement: fullImport });
  }

  // Remove all import statements
  processedContent = processedContent.replace(importPattern, '');

  // Process each component call
  for (const [componentName, { path: chapterPath }] of imports) {
    const componentCallPattern = new RegExp(`<${componentName}\\s*\\/>`, 'g');

    try {
      const chapterFile = resolve(contentDir, 'chapters', chapterPath);
      const chapterContent = await readMdxFile(chapterFile);
      const { content: chapterMarkdown } = extractFrontmatter(chapterContent);
      const cleanChapter = cleanMdxToMarkdown(chapterMarkdown);

      processedContent = processedContent.replace(componentCallPattern, cleanChapter);
      console.log(`✅ Processed chapter: ${chapterPath}`);
    } catch (error) {
      console.warn(`Warning: Could not process chapter ${chapterPath}:`, error.message);
      processedContent = processedContent.replace(componentCallPattern, `\n*[Chapter ${chapterPath} could not be loaded]*\n`);
    }
  }

  return processedContent;
}

function createLatexPreamble(frontmatter) {
  const title = frontmatter.title ? frontmatter.title.replace(/\n/g, ' ') : 'Untitled Article';
  const subtitle = frontmatter.subtitle || '';
  const authors = frontmatter.authors || '';
  const date = frontmatter.published || '';

  return `\\documentclass[11pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{amsmath,amsfonts,amssymb}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{array}
\\usepackage{multirow}
\\usepackage{wrapfig}
\\usepackage{float}
\\usepackage{colortbl}
\\usepackage{pdflscape}
\\usepackage{tabu}
\\usepackage{threeparttable}
\\usepackage{threeparttablex}
\\usepackage{ulem}
\\usepackage{makecell}
\\usepackage{xcolor}
\\usepackage{listings}
\\usepackage{fancyvrb}
\\usepackage{geometry}
\\geometry{margin=1in}

\\title{${title}${subtitle ? `\\\\\\large ${subtitle}` : ''}}
${authors ? `\\author{${authors}}` : ''}
${date ? `\\date{${date}}` : ''}

\\begin{document}
\\maketitle
\\tableofcontents
\\newpage

`;
}

async function main() {
  const cwd = process.cwd();
  const args = parseArgs(process.argv);

  // Check if pandoc is installed
  const hasPandoc = await checkPandocInstalled();
  if (!hasPandoc) {
    console.error('❌ Pandoc is not installed. Please install it first:');
    console.error('   macOS: brew install pandoc');
    console.error('   Ubuntu: apt-get install pandoc');
    console.error('   Windows: choco install pandoc');
    process.exit(1);
  }

  const contentDir = resolve(cwd, 'src/content');
  const articleFile = resolve(contentDir, 'article.mdx');

  // Check if article.mdx exists
  try {
    await fs.access(articleFile);
  } catch {
    console.error(`❌ Could not find article.mdx at ${articleFile}`);
    process.exit(1);
  }

  console.log('> Reading article content...');
  const articleContent = await readMdxFile(articleFile);
  const { frontmatter, content } = extractFrontmatter(articleContent);

  console.log('> Processing chapters...');
  const processedContent = await processChapterImports(content, contentDir);

  console.log('> Converting MDX to Markdown...');
  const markdownContent = cleanMdxToMarkdown(processedContent);

  // Generate output filename
  const title = frontmatter.title ? frontmatter.title.replace(/\n/g, ' ') : 'article';
  const outFileBase = args.filename ? String(args.filename).replace(/\.(tex|pdf)$/i, '') : slugify(title);

  // Create temporary markdown file (ensure it's pure markdown without YAML frontmatter)
  const tempMdFile = resolve(cwd, 'temp-article.md');

  // Clean the markdown content to ensure no YAML frontmatter remains
  let cleanMarkdown = markdownContent;
  // Remove any potential YAML frontmatter that might have leaked through
  cleanMarkdown = cleanMarkdown.replace(/^---\n[\s\S]*?\n---\n/, '');
  // Remove any standalone YAML blocks that might cause issues
  cleanMarkdown = cleanMarkdown.replace(/^---\n([\s\S]*?)\n---$/gm, '');

  await fs.writeFile(tempMdFile, cleanMarkdown);


  console.log('> Converting to LaTeX with Pandoc...');
  const outputLatex = resolve(cwd, 'dist', `${outFileBase}.tex`);

  // Ensure dist directory exists
  await fs.mkdir(resolve(cwd, 'dist'), { recursive: true });

  // Pandoc conversion arguments
  const pandocArgs = [
    tempMdFile,
    '-o', outputLatex,
    '--from=markdown-yaml_metadata_block', // Explicitly exclude YAML metadata parsing
    '--to=latex',
    '--standalone',
    '--toc',
    '--number-sections',
    '--highlight-style=tango',
    '--listings'
  ];

  // Add bibliography if it exists
  const bibFile = resolve(contentDir, 'bibliography.bib');
  try {
    await fs.access(bibFile);
    pandocArgs.push('--bibliography', bibFile);
    pandocArgs.push('--citeproc');
    console.log('✅ Found bibliography file, including citations');
  } catch {
    console.log('ℹ️  No bibliography file found');
  }

  try {
    await run('pandoc', pandocArgs);
    console.log(`✅ LaTeX generated: ${outputLatex}`);

    // Optionally compile to PDF if requested
    if (args.pdf) {
      console.log('> Compiling LaTeX to PDF...');
      const outputPdf = resolve(cwd, 'dist', `${outFileBase}.pdf`);
      await run('pdflatex', ['-output-directory', resolve(cwd, 'dist'), outputLatex]);
      console.log(`✅ PDF generated: ${outputPdf}`);
    }

  } catch (error) {
    console.error('❌ Pandoc conversion failed:', error.message);
    process.exit(1);
  } finally {
    // Clean up temporary file
    try {
      await fs.unlink(tempMdFile);
    } catch { }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
