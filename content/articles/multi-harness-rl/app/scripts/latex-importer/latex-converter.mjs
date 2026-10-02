#!/usr/bin/env node

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { cleanBibliography } from './bib-cleaner.mjs';
import { postProcessMarkdown } from './post-processor.mjs';
import { preprocessLatexReferences } from './reference-preprocessor.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const DEFAULT_INPUT = join(__dirname, 'input', 'main.tex');
const DEFAULT_OUTPUT = join(__dirname, 'output');

function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: DEFAULT_INPUT,
        output: DEFAULT_OUTPUT,
        clean: false
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.split('=')[1];
        } else if (arg.startsWith('--output=')) {
            config.output = arg.split('=')[1];
        } else if (arg === '--clean') {
            config.clean = true;
        }
    }

    return config;
}

function ensureDirectory(dir) {
    if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
    }
}

function cleanDirectory(dir) {
    if (existsSync(dir)) {
        execSync(`rm -rf "${dir}"/*`, { stdio: 'inherit' });
    }
}

function preprocessLatexFile(inputFile, outputDir) {
    const inputDir = dirname(inputFile);
    const tempFile = join(outputDir, 'temp_main.tex');

    console.log('🔄 Preprocessing LaTeX file to resolve \\input commands...');

    let content = readFileSync(inputFile, 'utf8');

    // Remove problematic commands that break pandoc
    console.log('🧹 Cleaning problematic LaTeX constructs...');

    // Fix citation issues - but not in citation keys
    content = content.replace(/\$p_0\$(?![A-Za-z])/g, 'p0');

    // Convert complex math environments to simple delimiters
    content = content.replace(/\$\$\\begin\{equation\*\}/g, '$$');
    content = content.replace(/\\end\{equation\*\}\$\$/g, '$$');
    content = content.replace(/\\begin\{equation\*\}/g, '$$');
    content = content.replace(/\\end\{equation\*\}/g, '$$');
    // Keep align environments intact for KaTeX support
    // Protect align environments by temporarily replacing them before cleaning & operators
    const alignBlocks = [];
    content = content.replace(/\\begin\{align\}([\s\S]*?)\\end\{align\}/g, (match, alignContent) => {
        alignBlocks.push(match);
        return `__ALIGN_BLOCK_${alignBlocks.length - 1}__`;
    });

    // Now remove & operators from non-align content (outside align environments)
    content = content.replace(/&=/g, '=');
    content = content.replace(/&/g, '');

    // Restore align blocks with their & operators intact
    alignBlocks.forEach((block, index) => {
        content = content.replace(`__ALIGN_BLOCK_${index}__`, block);
    });

    // Convert LaTeX citations to Pandoc format
    content = content.replace(/\\cite[tp]?\{([^}]+)\}/g, (match, citations) => {
        // Handle multiple citations separated by commas - all become simple @citations
        return citations.split(',').map(cite => `@${cite.trim()}`).join(', ');
    });

    // Handle complex \textsc with nested math - extract and simplify (but not in command definitions)
    content = content.replace(/\\textsc\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, (match, content_inside, offset) => {
        // Skip if this is inside a \newcommand or similar definition
        const before = content.substring(Math.max(0, offset - 50), offset);
        if (before.includes('\\newcommand') || before.includes('\\renewcommand') || before.includes('\\def')) {
            return match; // Keep original
        }

        // Remove math delimiters inside textsc for simplification
        const simplified = content_inside.replace(/\\\([^)]+\\\)/g, 'MATHEXPR');
        return `\\text{${simplified}}`;
    });

    // Remove complex custom commands that pandoc can't handle
    content = content.replace(/\\input\{snippets\/[^}]+\}/g, '% Code snippet removed');

    // Find all \input{} commands (but skip commented ones)
    const inputRegex = /^([^%]*?)\\input\{([^}]+)\}/gm;
    let match;

    while ((match = inputRegex.exec(content)) !== null) {
        const beforeInput = match[1];
        const inputPath = match[2];

        // Skip if the \input is commented (% appears before \input on the line)
        if (beforeInput.includes('%')) {
            continue;
        }
        let fullPath;

        // Skip only problematic files, let Pandoc handle macros
        if (inputPath.includes('snippets/')) {
            console.log(`   Skipping: ${inputPath}`);
            content = content.replace(`\\input{${inputPath}}`, `% Skipped: ${inputPath}`);
            continue;
        }

        // Handle paths with or without .tex extension
        if (inputPath.endsWith('.tex')) {
            fullPath = join(inputDir, inputPath);
        } else {
            fullPath = join(inputDir, inputPath + '.tex');
        }

        if (existsSync(fullPath)) {
            console.log(`   Including: ${inputPath}`);
            let includedContent = readFileSync(fullPath, 'utf8');

            // Clean included content too
            includedContent = includedContent.replace(/\$p_0\$/g, 'p0');
            includedContent = includedContent.replace(/\\input\{snippets\/[^}]+\}/g, '% Code snippet removed');

            // Handle complex \textsc in included content
            includedContent = includedContent.replace(/\\textsc\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, (match, content_inside, offset) => {
                // Skip if this is inside a \newcommand or similar definition
                const before = includedContent.substring(Math.max(0, offset - 50), offset);
                if (before.includes('\\newcommand') || before.includes('\\renewcommand') || before.includes('\\def')) {
                    return match; // Keep original
                }

                const simplified = content_inside.replace(/\\\([^)]+\\\)/g, 'MATHEXPR');
                return `\\text{${simplified}}`;
            });

            // Apply same align-preserving logic to included content
            const alignBlocksIncluded = [];
            includedContent = includedContent.replace(/\\begin\{align\}([\s\S]*?)\\end\{align\}/g, (match, alignContent) => {
                alignBlocksIncluded.push(match);
                return `__ALIGN_BLOCK_${alignBlocksIncluded.length - 1}__`;
            });

            // Remove alignment operators from non-align content in included files
            includedContent = includedContent.replace(/&=/g, '=');
            includedContent = includedContent.replace(/&/g, '');

            // Restore align blocks with their & operators intact
            alignBlocksIncluded.forEach((block, index) => {
                includedContent = includedContent.replace(`__ALIGN_BLOCK_${index}__`, block);
            });

            // Convert math environments in included content
            includedContent = includedContent.replace(/\$\$\\begin\{equation\*\}/g, '$$');
            includedContent = includedContent.replace(/\\end\{equation\*\}\$\$/g, '$$');
            includedContent = includedContent.replace(/\\begin\{equation\*\}/g, '$$');
            includedContent = includedContent.replace(/\\end\{equation\*\}/g, '$$');

            // Convert citations in included content
            includedContent = includedContent.replace(/\\cite[tp]?\{([^}]+)\}/g, (match, citations) => {
                return citations.split(',').map(cite => `@${cite.trim()}`).join(', ');
            });

            content = content.replace(`\\input{${inputPath}}`, includedContent);
        } else {
            console.log(`   ⚠️  File not found: ${fullPath} (skipping)`);
            content = content.replace(`\\input{${inputPath}}`, `% File not found: ${inputPath}`);
        }
    }

    // Apply reference preprocessing AFTER input inclusion to ensure all references are captured
    console.log('🔧 Preprocessing LaTeX references for MDX compatibility...');
    const referenceResult = preprocessLatexReferences(content);
    content = referenceResult.content;

    // Write the preprocessed file
    writeFileSync(tempFile, content);
    return tempFile;
}

function processBibliography(inputFile, outputDir) {
    const bibFile = join(dirname(inputFile), 'main.bib');
    const outputBibFile = join(outputDir, 'main.bib');

    if (!existsSync(bibFile)) {
        console.log('   ⚠️  No bibliography file found');
        return null;
    }

    const success = cleanBibliography(bibFile, outputBibFile);
    return success ? outputBibFile : null;
}

export function convertLatexToMarkdown(inputFile, outputDir) {
    console.log('🚀 Simple LaTeX to Markdown Converter');
    console.log(`📁 Input:  ${inputFile}`);
    console.log(`📁 Output: ${outputDir}`);

    // Check if input file exists
    if (!existsSync(inputFile)) {
        console.error(`❌ Input file not found: ${inputFile}`);
        process.exit(1);
    }

    // Ensure output directory exists
    ensureDirectory(outputDir);

    try {
        // Check if pandoc is available
        execSync('pandoc --version', { stdio: 'pipe' });
    } catch (error) {
        console.error('❌ Pandoc not found. Please install it: brew install pandoc');
        process.exit(1);
    }

    // Clean and copy bibliography
    const cleanBibFile = processBibliography(inputFile, outputDir);

    // Preprocess the LaTeX file to resolve \input commands
    const preprocessedFile = preprocessLatexFile(inputFile, outputDir);

    const inputFileName = basename(inputFile, '.tex');
    const outputFile = join(outputDir, `${inputFileName}.md`);

    try {
        console.log('📄 Converting with Pandoc...');

        // Enhanced pandoc conversion - use tex_math_dollars for KaTeX compatibility
        const bibOption = cleanBibFile ? `--bibliography="${cleanBibFile}"` : '';

        // Use gfm+tex_math_dollars for simple $ delimiters compatible with KaTeX
        const mediaDir = join(outputDir, 'assets', 'image');
        ensureDirectory(mediaDir);
        const inputDir = dirname(inputFile);
        const equationFilterPath = join(__dirname, 'filters', 'equation-ids.lua');
        const pandocCommand = `pandoc "${preprocessedFile}" -f latex+latex_macros -t gfm+tex_math_dollars+raw_html --shift-heading-level-by=1 --wrap=none ${bibOption} --extract-media="${mediaDir}" --resource-path="${inputDir}" --lua-filter="${equationFilterPath}" -o "${outputFile}"`;

        console.log(`   Running: ${pandocCommand}`);
        execSync(pandocCommand, { stdio: 'pipe' });

        // Clean up temp file
        execSync(`rm "${preprocessedFile}"`, { stdio: 'pipe' });

        // Post-processing to fix KaTeX incompatible constructions
        let markdownContent = readFileSync(outputFile, 'utf8');

        // Use modular post-processor with code injection
        markdownContent = postProcessMarkdown(markdownContent, inputDir);

        writeFileSync(outputFile, markdownContent);

        console.log(`✅ Conversion completed: ${outputFile}`);

        // Show file size
        const stats = execSync(`wc -l "${outputFile}"`, { encoding: 'utf8' });
        const lines = stats.trim().split(' ')[0];
        console.log(`📊 Result: ${lines} lines written`);

    } catch (error) {
        console.error('❌ Pandoc conversion failed:');
        console.error(error.message);
        // Clean up temp file on error
        try {
            execSync(`rm "${preprocessedFile}"`, { stdio: 'pipe' });
        } catch { }
        process.exit(1);
    }
}

function main() {
    const config = parseArgs();

    if (config.clean) {
        console.log('🧹 Cleaning output directory...');
        cleanDirectory(config.output);
    }

    convertLatexToMarkdown(config.input, config.output);

    console.log('🎉 Simple conversion completed!');
}

// Show help if requested
if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
🚀 Simple LaTeX to Markdown Converter

Usage:
  node scripts/simple-latex-to-markdown.mjs [options]

Options:
  --input=PATH     Input LaTeX file (default: latex-converter/input-example/main.tex)
  --output=PATH    Output directory (default: output/)
  --clean          Clean output directory before conversion
  --help, -h       Show this help

Examples:
  # Basic conversion
  node scripts/simple-latex-to-markdown.mjs

  # Custom paths
  node scripts/simple-latex-to-markdown.mjs --input=my-paper.tex --output=converted/

  # Clean output first
  node scripts/simple-latex-to-markdown.mjs --clean
`);
    process.exit(0);
}

main();
