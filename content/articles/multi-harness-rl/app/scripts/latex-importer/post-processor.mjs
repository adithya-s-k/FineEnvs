#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Post-processor for cleaning Markdown content from LaTeX conversion
 * Each function handles a specific type of cleanup for maintainability
 */

/**
 * Remove TeX low-level grouping commands that break KaTeX
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function removeTexGroupingCommands(content) {
    console.log('  🧹 Removing TeX grouping commands...');

    return content
        .replace(/\\mathopen\{\}\\mathclose\\bgroup/g, '')
        .replace(/\\aftergroup\\egroup/g, '')
        .replace(/\\bgroup/g, '')
        .replace(/\\egroup/g, '');
}

/**
 * Simplify LaTeX delimiter constructions
 * @param {string} content - Markdown content  
 * @returns {string} - Cleaned content
 */
function simplifyLatexDelimiters(content) {
    console.log('  🔧 Simplifying LaTeX delimiters...');

    return content
        .replace(/\\left\[\s*/g, '[')
        .replace(/\s*\\right\]/g, ']');
}

/**
 * Remove orphaned LaTeX labels
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function removeOrphanedLabels(content) {
    console.log('  🏷️  Removing orphaned labels...');

    return content
        .replace(/^\s*\\label\{[^}]+\}\s*$/gm, '')
        .replace(/\\label\{[^}]+\}/g, '');
}

/**
 * Fix KaTeX-incompatible math commands
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function fixMathCommands(content) {
    console.log('  📐 Fixing KaTeX-incompatible math commands...');

    return content
        // Replace \hdots with \ldots (KaTeX compatible)
        .replace(/\\hdots/g, '\\ldots')
        // Add more math command fixes here as needed
        .replace(/\\vdots/g, '\\vdots'); // This one should be fine, but kept for consistency
}

/**
 * Convert LaTeX matrix commands to KaTeX-compatible environments
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed matrix commands
 */
function fixMatrixCommands(content) {
    console.log('  🔢 Converting matrix commands to KaTeX format...');

    let fixedCount = 0;

    // Convert \pmatrix{...} to \begin{pmatrix}...\end{pmatrix}
    content = content.replace(/\\pmatrix\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, (match, matrixContent) => {
        fixedCount++;
        // Split by \\ for rows, handle nested braces
        const rows = matrixContent.split('\\\\').map(row => row.trim()).filter(row => row);
        return `\\begin{pmatrix}\n${rows.join(' \\\\\n')}\n\\end{pmatrix}`;
    });

    // Convert \bmatrix{...} to \begin{bmatrix}...\end{bmatrix}
    content = content.replace(/\\bmatrix\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, (match, matrixContent) => {
        fixedCount++;
        const rows = matrixContent.split('\\\\').map(row => row.trim()).filter(row => row);
        return `\\begin{bmatrix}\n${rows.join(' \\\\\n')}\n\\end{bmatrix}`;
    });

    // Convert \vmatrix{...} to \begin{vmatrix}...\end{vmatrix}
    content = content.replace(/\\vmatrix\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g, (match, matrixContent) => {
        fixedCount++;
        const rows = matrixContent.split('\\\\').map(row => row.trim()).filter(row => row);
        return `\\begin{vmatrix}\n${rows.join(' \\\\\n')}\n\\end{vmatrix}`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} matrix command(s)`);
    }

    return content;
}

/**
 * Fix Unicode characters that break MDX/JSX parsing
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function fixUnicodeIssues(content) {
    console.log('  🌐 Fixing Unicode characters for MDX compatibility...');

    return content
        // Replace Unicode middle dot (·) with \cdot in math expressions
        .replace(/\$([^$]*?)·([^$]*?)\$/g, (match, before, after) => {
            return `$${before}\\cdot${after}$`;
        })
        // Replace Unicode middle dot in display math
        .replace(/\$\$([^$]*?)·([^$]*?)\$\$/g, (match, before, after) => {
            return `$$${before}\\cdot${after}$$`;
        })
        // Replace other problematic Unicode characters
        .replace(/[""]/g, '"')  // Smart quotes to regular quotes
        .replace(/['']/g, "'")  // Smart apostrophes to regular apostrophes
        .replace(/…/g, '...')   // Ellipsis to three dots
        .replace(/–/g, '-')     // En dash to hyphen
        .replace(/—/g, '--');   // Em dash to double hyphen
}

/**
 * Fix multiline math expressions for MDX compatibility
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function fixMultilineMath(content) {
    console.log('  📏 Fixing multiline math expressions for MDX...');

    return content
        // Convert multiline inline math to display math blocks (more precise regex)
        // Only match if the content is a self-contained math expression within a single line
        .replace(/\$([^$\n]*\\\\[^$\n]*)\$/g, (match, mathContent) => {
            // Only convert if it contains actual math operators and line breaks
            if (mathContent.includes('\\\\') && /[=+\-*/^_{}]/.test(mathContent)) {
                // Remove leading/trailing whitespace and normalize newlines
                const cleanedMath = mathContent
                    .replace(/^\s+|\s+$/g, '')
                    .replace(/\s*\\\\\s*/g, '\\\\\n    ');
                return `$$\n${cleanedMath}\n$$`;
            }
            return match; // Keep original if it doesn't look like multiline math
        })
        // Ensure display math blocks are properly separated
        .replace(/\$\$\s*\n\s*([^$]+?)\s*\n\s*\$\$/g, (match, mathContent) => {
            return `\n$$\n${mathContent.trim()}\n$$\n`;
        });
}

/**
 * Inject code snippets into empty code blocks
 * @param {string} content - Markdown content
 * @param {string} inputDir - Directory containing the LaTeX source and snippets
 * @returns {string} - Content with injected code snippets
 */
function injectCodeSnippets(content, inputDir = null) {
    console.log('  💻 Injecting code snippets...');

    if (!inputDir) {
        console.log('    ⚠️  No input directory provided, skipping code injection');
        return content;
    }

    const snippetsDir = join(inputDir, 'snippets');

    if (!existsSync(snippetsDir)) {
        console.log('    ⚠️  Snippets directory not found, skipping code injection');
        return content;
    }

    // Get all available snippet files
    let availableSnippets = [];
    try {
        availableSnippets = readdirSync(snippetsDir);
        console.log(`    📁 Found ${availableSnippets.length} snippet file(s): ${availableSnippets.join(', ')}`);
    } catch (error) {
        console.log(`    ❌ Error reading snippets directory: ${error.message}`);
        return content;
    }

    // Find all empty code blocks
    const emptyCodeBlockPattern = /```\s*(\w+)\s*\n\s*```/g;

    let processedContent = content;
    let injectionCount = 0;

    processedContent = processedContent.replace(emptyCodeBlockPattern, (match, language) => {
        // Map language names to file extensions
        const extensionMap = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'bash': 'sh',
            'shell': 'sh'
        };

        const fileExtension = extensionMap[language] || language;

        // Try to find a matching snippet file for this language
        const matchingFiles = availableSnippets.filter(file =>
            file.endsWith(`.${fileExtension}`)
        );

        if (matchingFiles.length === 0) {
            console.log(`    ⚠️  No ${language} snippet found (looking for .${fileExtension})`);
            return match;
        }

        // Use the first matching file (could be made smarter with context analysis)
        const selectedFile = matchingFiles[0];
        const snippetPath = join(snippetsDir, selectedFile);

        try {
            const snippetContent = readFileSync(snippetPath, 'utf8');
            injectionCount++;
            console.log(`    ✅ Injected: ${selectedFile}`);
            return `\`\`\`${language}\n${snippetContent.trim()}\n\`\`\``;
        } catch (error) {
            console.log(`    ❌ Error reading ${selectedFile}: ${error.message}`);
            return match;
        }
    });

    if (injectionCount > 0) {
        console.log(`    📊 Injected ${injectionCount} code snippet(s)`);
    }

    return processedContent;
}

/**
 * Fix all attributes that still contain colons (href, data-reference, id)
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function fixAllAttributes(content) {
    console.log('  🔗 Fixing all attributes with colons...');

    let fixedCount = 0;

    // Fix href attributes containing colons
    content = content.replace(/href="([^"]*):([^"]*)"/g, (match, before, after) => {
        fixedCount++;
        return `href="${before}-${after}"`;
    });

    // Fix data-reference attributes containing colons
    content = content.replace(/data-reference="([^"]*):([^"]*)"/g, (match, before, after) => {
        fixedCount++;
        return `data-reference="${before}-${after}"`;
    });

    // Fix id attributes containing colons (like in Figure components)
    content = content.replace(/id="([^"]*):([^"]*)"/g, (match, before, after) => {
        fixedCount++;
        return `id="${before}-${after}"`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} attribute(s) with colons`);
    }

    return content;
}

/**
 * Fix link text content that still contains colons
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function fixLinkTextContent(content) {
    console.log('  📝 Fixing link text content with colons...');

    let fixedCount = 0;

    // Fix text content within links that contain references with colons
    // Pattern: <a ...>[text:content]</a>
    const cleanedContent = content.replace(/<a([^>]*)>\[([^:]*):([^\]]*)\]<\/a>/g, (match, attributes, before, after) => {
        fixedCount++;
        return `<a${attributes}>[${before}-${after}]</a>`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} link text(s) with colons`);
    }

    return cleanedContent;
}

/**
 * Convert align anchor markers to proper HTML spans outside math blocks
 * @param {string} content - Markdown content
 * @returns {string} - Content with converted anchor spans
 */
function convertAlignAnchors(content) {
    console.log('  🏷️  Converting align anchor markers to HTML spans...');

    let convertedCount = 0;

    // Find and replace align anchor markers with proper spans outside math blocks
    content = content.replace(/``` math\n%%ALIGN_ANCHOR_ID\{([^}]+)\}%%\n([\s\S]*?)\n```/g, (match, anchorId, mathContent) => {
        convertedCount++;
        return `<span id="${anchorId}" style="position: absolute;"></span>\n\n\`\`\` math\n${mathContent}\n\`\`\``;
    });

    if (convertedCount > 0) {
        console.log(`    ✅ Converted ${convertedCount} align anchor marker(s) to spans`);
    }

    return content;
}

/**
 * Main post-processing function that applies all cleanup steps
 * @param {string} content - Raw Markdown content from Pandoc
 * @param {string} inputDir - Optional: Directory containing LaTeX source for code injection
 * @returns {string} - Cleaned Markdown content
 */
export function postProcessMarkdown(content, inputDir = null) {
    console.log('🔧 Post-processing for KaTeX compatibility...');

    let processedContent = content;

    // Apply each cleanup step sequentially
    processedContent = removeTexGroupingCommands(processedContent);
    processedContent = simplifyLatexDelimiters(processedContent);
    processedContent = removeOrphanedLabels(processedContent);
    processedContent = convertAlignAnchors(processedContent);
    processedContent = fixMathCommands(processedContent);
    processedContent = fixMatrixCommands(processedContent);
    processedContent = fixUnicodeIssues(processedContent);
    processedContent = fixMultilineMath(processedContent);
    processedContent = fixAllAttributes(processedContent);
    processedContent = fixLinkTextContent(processedContent);

    // Inject code snippets if input directory is provided
    if (inputDir) {
        processedContent = injectCodeSnippets(processedContent, inputDir);
    }

    return processedContent;
}

/**
 * CLI interface for standalone usage
 */
function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: join(__dirname, 'output', 'main.md'),
        output: null, // Will default to input if not specified
        verbose: false,
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.substring('--input='.length);
        } else if (arg.startsWith('--output=')) {
            config.output = arg.substring('--output='.length);
        } else if (arg === '--verbose') {
            config.verbose = true;
        } else if (arg === '--help' || arg === '-h') {
            console.log(`
🔧 Markdown Post-Processor

Usage:
  node post-processor.mjs [options]

Options:
  --input=PATH     Input Markdown file (default: output/main.md)
  --output=PATH    Output file (default: overwrites input)
  --verbose        Verbose output
  --help, -h       Show this help

Examples:
  # Process main.md in-place
  node post-processor.mjs

  # Process with custom paths
  node post-processor.mjs --input=raw.md --output=clean.md
            `);
            process.exit(0);
        }
    }

    // Default output to input if not specified
    if (!config.output) {
        config.output = config.input;
    }

    return config;
}

function main() {
    const config = parseArgs();

    console.log('🔧 Markdown Post-Processor');
    console.log(`📁 Input:  ${config.input}`);
    console.log(`📁 Output: ${config.output}`);

    try {
        const content = readFileSync(config.input, 'utf8');
        const processedContent = postProcessMarkdown(content);

        writeFileSync(config.output, processedContent);

        console.log(`✅ Post-processing completed: ${config.output}`);

        // Show stats if verbose
        if (config.verbose) {
            const originalLines = content.split('\n').length;
            const processedLines = processedContent.split('\n').length;
            console.log(`📊 Lines: ${originalLines} → ${processedLines}`);
        }

    } catch (error) {
        console.error('❌ Post-processing failed:');
        console.error(error.message);
        process.exit(1);
    }
}

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
