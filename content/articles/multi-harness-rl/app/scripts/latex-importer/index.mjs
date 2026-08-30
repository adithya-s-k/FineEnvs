#!/usr/bin/env node

import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync } from 'fs';
import { convertLatexToMarkdown } from './latex-converter.mjs';
import { convertToMdx } from './mdx-converter.mjs';
import { cleanBibliography } from './bib-cleaner.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default configuration
const DEFAULT_INPUT = join(__dirname, 'input', 'main.tex');
const DEFAULT_OUTPUT = join(__dirname, 'output');
const ASTRO_CONTENT_PATH = join(__dirname, '..', '..', 'src', 'content', 'article.mdx');

function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: DEFAULT_INPUT,
        output: DEFAULT_OUTPUT,
        clean: false,
        bibOnly: false,
        convertOnly: false,
        mdx: false,
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.split('=')[1];
        } else if (arg.startsWith('--output=')) {
            config.output = arg.split('=')[1];
        } else if (arg === '--clean') {
            config.clean = true;
        } else if (arg === '--bib-only') {
            config.bibOnly = true;
        } else if (arg === '--convert-only') {
            config.convertOnly = true;
        }
    }

    return config;
}

function showHelp() {
    console.log(`
🚀 LaTeX to Markdown Toolkit

Usage:
  node index.mjs [options]

Options:
  --input=PATH      Input LaTeX file (default: input/main.tex)
  --output=PATH     Output directory (default: output/)
  --clean           Clean output directory before processing
  --bib-only        Only clean bibliography file
  --convert-only    Only convert LaTeX to Markdown (skip bib cleaning)
  --help, -h        Show this help

Examples:
  # Full conversion with bibliography cleaning
  node index.mjs --clean

  # Only clean bibliography
  node index.mjs --bib-only --input=paper.tex --output=clean/

  # Only convert LaTeX (use existing clean bibliography)
  node index.mjs --convert-only

  # Custom paths
  node index.mjs --input=../paper/main.tex --output=../results/ --clean
`);
}

function main() {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        process.exit(0);
    }

    const config = parseArgs();

    console.log('🚀 LaTeX to Markdown Toolkit');
    console.log('==============================');

    try {
        if (config.bibOnly) {
            // Only clean bibliography
            console.log('📚 Bibliography cleaning mode');
            const bibInput = config.input.replace('.tex', '.bib');
            const bibOutput = join(config.output, 'main.bib');

            cleanBibliography(bibInput, bibOutput);
            console.log('🎉 Bibliography cleaning completed!');

        } else if (config.convertOnly) {
            // Only convert LaTeX
            console.log('📄 Conversion only mode');
            convertLatexToMarkdown(config.input, config.output);

        } else {
            // Full workflow
            console.log('🔄 Full conversion workflow');
            convertLatexToMarkdown(config.input, config.output);

            // Convert to MDX if requested
            const markdownFile = join(config.output, 'main.md');
            const mdxFile = join(config.output, 'main.mdx');

            console.log('📝 Converting Markdown to MDX...');
            convertToMdx(markdownFile, mdxFile);

            // Copy MDX to Astro content directory
            console.log('📋 Copying MDX to Astro content directory...');
            try {
                copyFileSync(mdxFile, ASTRO_CONTENT_PATH);
                console.log(`    ✅ Copied to ${ASTRO_CONTENT_PATH}`);
            } catch (error) {
                console.warn(`    ⚠️  Failed to copy MDX to Astro: ${error.message}`);
            }
        }

    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

// Export functions for use as module
export { convertLatexToMarkdown, cleanBibliography };

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
