#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname, basename, extname } from 'path';
import { fileURLToPath } from 'url';
import { extractAndGenerateFrontmatter } from './metadata-extractor.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const DEFAULT_INPUT = join(__dirname, 'output', 'main.md');
const DEFAULT_OUTPUT = join(__dirname, 'output', 'main.mdx');

function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: DEFAULT_INPUT,
        output: DEFAULT_OUTPUT,
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.substring('--input='.length);
        } else if (arg.startsWith('--output=')) {
            config.output = arg.substring('--output='.length);
        } else if (arg === '--help' || arg === '-h') {
            console.log(`
📝 Markdown to MDX Converter

Usage:
  node mdx-converter.mjs [options]

Options:
  --input=PATH     Input Markdown file (default: ${DEFAULT_INPUT})
  --output=PATH    Output MDX file (default: ${DEFAULT_OUTPUT})
  --help, -h       Show this help

Examples:
  # Basic conversion
  node mdx-converter.mjs

  # Custom paths
  node mdx-converter.mjs --input=article.md --output=article.mdx
            `);
            process.exit(0);
        } else if (!config.input) {
            config.input = arg;
        } else if (!config.output) {
            config.output = arg;
        }
    }
    return config;
}

/**
 * Modular MDX post-processing functions for Astro compatibility
 * Each function handles a specific type of transformation
 */

/**
 * Track which Astro components are used during transformations
 */
const usedComponents = new Set();

/**
 * Track individual image imports needed
 */
const imageImports = new Map(); // src -> varName

/**
 * Add required component imports to the frontmatter
 * @param {string} content - MDX content
 * @returns {string} - Content with component imports
 */
/**
 * Generate a variable name from image path
 * @param {string} src - Image source path
 * @returns {string} - Valid variable name
 */
function generateImageVarName(src) {
    // Extract filename without extension and make it a valid JS variable
    const filename = src.split('/').pop().replace(/\.[^.]+$/, '');
    return filename.replace(/[^a-zA-Z0-9]/g, '_').replace(/^[0-9]/, 'img_$&');
}

function addComponentImports(content) {
    console.log('  📦 Adding component and image imports...');

    let imports = [];

    // Add component imports
    if (usedComponents.size > 0) {
        const componentImports = Array.from(usedComponents)
            .map(component => `import ${component} from '../components/${component}.astro';`);
        imports.push(...componentImports);
        console.log(`    ✅ Importing components: ${Array.from(usedComponents).join(', ')}`);
    }

    // Add image imports
    if (imageImports.size > 0) {
        const imageImportStatements = Array.from(imageImports.entries())
            .map(([src, varName]) => `import ${varName} from '${src}';`);
        imports.push(...imageImportStatements);
        console.log(`    ✅ Importing ${imageImports.size} image(s)`);
    }

    if (imports.length === 0) {
        console.log('    ℹ️  No imports needed');
        return content;
    }

    const importBlock = imports.join('\n');

    // Insert imports after frontmatter
    const frontmatterEnd = content.indexOf('---', 3) + 3;
    if (frontmatterEnd > 2) {
        return content.slice(0, frontmatterEnd) + '\n\n' + importBlock + '\n' + content.slice(frontmatterEnd);
    } else {
        // No frontmatter, add at beginning
        return importBlock + '\n\n' + content;
    }
}


/**
 * Convert grouped figures (subfigures) to MultiFigure components
 * @param {string} content - MDX content
 * @returns {string} - Content with MultiFigure components for grouped figures
 */
function convertSubfiguresToMultiFigure(content) {
    console.log('  🖼️✨ Converting subfigures to MultiFigure components...');

    let convertedCount = 0;

    // Pattern to match: <figure> containing multiple <figure> elements with a global caption
    // This matches the LaTeX subfigure pattern that gets converted by Pandoc
    const subfigureGroupPattern = /<figure>\s*((?:<figure>[\s\S]*?<\/figure>\s*){2,})<figcaption>([\s\S]*?)<\/figcaption>\s*<\/figure>/g;

    const convertedContent = content.replace(subfigureGroupPattern, (match, figuresMatch, globalCaption) => {
        convertedCount++;

        // Extract individual figures within the group
        // This pattern is more flexible to handle variations in HTML structure
        const individualFigurePattern = /<figure>\s*<img src="([^"]*)"[^>]*\/>\s*<p>&lt;span id="([^"]*)"[^&]*&gt;&lt;\/span&gt;<\/p>\s*<figcaption>([\s\S]*?)<\/figcaption>\s*<\/figure>/g;

        const images = [];
        let figureMatch;

        while ((figureMatch = individualFigurePattern.exec(figuresMatch)) !== null) {
            const [, src, id, caption] = figureMatch;

            // Clean the source path (similar to existing transformImages function)
            const cleanSrc = src.replace(/.*\/output\/assets\//, './assets/')
                .replace(/\/Users\/[^\/]+\/[^\/]+\/[^\/]+\/[^\/]+\/[^\/]+\/app\/scripts\/latex-to-markdown\/output\/assets\//, './assets/');

            // Clean caption text (remove HTML, normalize whitespace)
            const cleanCaption = caption
                .replace(/<[^>]*>/g, '')
                .replace(/\n/g, ' ')
                .replace(/\s+/g, ' ')
                .replace(/'/g, "\\'")
                .trim();

            // Generate alt text from caption
            const altText = cleanCaption.length > 100
                ? cleanCaption.substring(0, 100) + '...'
                : cleanCaption;

            // Generate variable name for import
            const varName = generateImageVarName(cleanSrc);
            imageImports.set(cleanSrc, varName);

            images.push({
                src: varName,
                alt: altText,
                caption: cleanCaption,
                id: id
            });
        }

        // Clean global caption
        const cleanGlobalCaption = globalCaption
            .replace(/<[^>]*>/g, '')
            .replace(/\n/g, ' ')
            .replace(/\s+/g, ' ')
            .replace(/'/g, "\\'")
            .trim();

        // Mark MultiFigure component as used
        usedComponents.add('MultiFigure');

        // Determine layout based on number of images
        let layout = 'auto';
        if (images.length === 2) layout = '2-column';
        else if (images.length === 3) layout = '3-column';
        else if (images.length === 4) layout = '4-column';

        // Generate MultiFigure component
        const imagesJson = images.map(img =>
            `    {\n      src: ${img.src},\n      alt: "${img.alt}",\n      caption: "${img.caption}",\n      id: "${img.id}"\n    }`
        ).join(',\n');

        return `<MultiFigure
  images={[
${imagesJson}
  ]}
  layout="${layout}"
  zoomable
  downloadable
  caption="${cleanGlobalCaption}"
/>`;
    });

    if (convertedCount > 0) {
        console.log(`    ✅ Converted ${convertedCount} subfigure group(s) to MultiFigure component(s)`);
    } else {
        console.log('    ℹ️  No subfigure groups found');
    }

    return convertedContent;
}

/**
 * Transform images to Figure components
 * @param {string} content - MDX content
 * @returns {string} - Content with Figure components
 */
/**
 * Create Figure component with import
 * @param {string} src - Clean image source
 * @param {string} alt - Alt text  
 * @param {string} id - Element ID
 * @param {string} caption - Figure caption
 * @param {string} width - Optional width
 * @returns {string} - Figure component markup
 */
function createFigureComponent(src, alt = '', id = '', caption = '', width = '') {
    const varName = generateImageVarName(src);
    imageImports.set(src, varName);
    usedComponents.add('Figure');

    const props = [];
    props.push(`src={${varName}}`);
    props.push('zoomable');
    props.push('downloadable');
    if (id) props.push(`id="${id}"`);
    props.push('layout="fixed"');
    if (alt) props.push(`alt="${alt}"`);
    if (caption) props.push(`caption={'${caption}'}`);

    return `<Figure\n  ${props.join('\n  ')}\n/>`;
}

function transformImages(content) {
    console.log('  🖼️  Transforming images to Figure components with imports...');

    let hasImages = false;

    // Helper function to clean source paths
    const cleanSrcPath = (src) => {
        return src.replace(/.*\/output\/assets\//, './assets/')
            .replace(/\/Users\/[^\/]+\/[^\/]+\/[^\/]+\/[^\/]+\/[^\/]+\/app\/scripts\/latex-to-markdown\/output\/assets\//, './assets/');
    };

    // Helper to clean caption text
    const cleanCaption = (caption) => {
        return caption
            .replace(/<[^>]*>/g, '')          // Remove HTML tags
            .replace(/\n/g, ' ')              // Replace newlines with spaces
            .replace(/\r/g, ' ')              // Replace carriage returns with spaces
            .replace(/\s+/g, ' ')             // Replace multiple spaces with single space
            .replace(/'/g, "\\'")             // Escape quotes
            .trim();                          // Trim whitespace
    };

    // Helper to clean alt text
    const cleanAltText = (alt, maxLength = 100) => {
        const cleaned = alt
            .replace(/<[^>]*>/g, '')          // Remove HTML tags
            .replace(/\n/g, ' ')              // Replace newlines with spaces
            .replace(/\r/g, ' ')              // Replace carriage returns with spaces
            .replace(/\s+/g, ' ')             // Replace multiple spaces with single space
            .trim();                          // Trim whitespace

        return cleaned.length > maxLength
            ? cleaned.substring(0, maxLength) + '...'
            : cleaned;
    };

    // 1. Transform complex HTML figures with style attributes
    content = content.replace(
        /<figure id="([^"]*)">\s*<img src="([^"]*)"(?:\s+style="([^"]*)")?\s*\/>\s*<figcaption>\s*(.*?)\s*<\/figcaption>\s*<\/figure>/gs,
        (match, id, src, style, caption) => {
            const cleanSrc = cleanSrcPath(src);
            const cleanCap = cleanCaption(caption);
            const altText = cleanAltText(cleanCap);
            hasImages = true;

            return createFigureComponent(cleanSrc, altText, id, cleanCap);
        }
    );

    // 2. Transform standalone img tags with style
    content = content.replace(
        /<img src="([^"]*)"(?:\s+style="([^"]*)")?\s*(?:alt="([^"]*)")?\s*\/>/g,
        (match, src, style, alt) => {
            const cleanSrc = cleanSrcPath(src);
            const cleanAlt = cleanAltText(alt || 'Figure');
            hasImages = true;

            return createFigureComponent(cleanSrc, cleanAlt);
        }
    );

    // 3. Transform images within wrapfigure divs
    content = content.replace(
        /<div class="wrapfigure">\s*r[\d.]+\s*<img src="([^"]*)"[^>]*\/>\s*<\/div>/gs,
        (match, src) => {
            const cleanSrc = cleanSrcPath(src);
            hasImages = true;

            return createFigureComponent(cleanSrc, 'Figure');
        }
    );

    // 4. Transform simple HTML figure/img without style
    content = content.replace(
        /<figure id="([^"]*)">\s*<img src="([^"]*)" \/>\s*<figcaption>\s*(.*?)\s*<\/figcaption>\s*<\/figure>/gs,
        (match, id, src, caption) => {
            const cleanSrc = cleanSrcPath(src);
            const cleanCap = cleanCaption(caption);
            const altText = cleanAltText(cleanCap);
            hasImages = true;

            return createFigureComponent(cleanSrc, altText, id, cleanCap);
        }
    );

    // 5. Clean up figures with minipage divs
    content = content.replace(
        /<figure id="([^"]*)">\s*<div class="minipage">\s*<img src="([^"]*)"[^>]*\/>\s*<\/div>\s*<figcaption[^>]*>(.*?)<\/figcaption>\s*<\/figure>/gs,
        (match, id, src, caption) => {
            const cleanSrc = cleanSrcPath(src);
            const cleanCap = cleanCaption(caption);
            const altText = cleanAltText(cleanCap);
            hasImages = true;

            return createFigureComponent(cleanSrc, altText, id, cleanCap);
        }
    );

    // 6. Transform Pandoc-style images: ![alt](src){#id attr="value"}
    content = content.replace(
        /!\[([^\]]*)\]\(([^)]+)\)(?:\{([^}]+)\})?/g,
        (match, alt, src, attributes) => {
            const cleanSrc = cleanSrcPath(src);
            const cleanAlt = cleanAltText(alt || 'Figure');
            hasImages = true;

            let id = '';
            if (attributes) {
                const idMatch = attributes.match(/#([\w-]+)/);
                if (idMatch) id = idMatch[1];
            }

            return createFigureComponent(cleanSrc, cleanAlt, id);
        }
    );

    if (hasImages) {
        console.log('    ✅ Figure components with imports will be created');
    }

    return content;
}

/**
 * Transform HTML spans with style attributes to appropriate components
 * @param {string} content - MDX content  
 * @returns {string} - Content with transformed spans
 */
function transformStyledSpans(content) {
    console.log('  🎨 Transforming styled spans...');

    // Transform HTML spans with style attributes
    content = content.replace(
        /<span style="color: ([^"]+)">(.*?)<\/span>/g,
        (match, color, text) => {
            // Map colors to semantic classes or components
            const colorMap = {
                'hf2': 'text-hf-secondary',
                'hf1': 'text-hf-primary'
            };

            const className = colorMap[color] || `text-${color}`;
            return `<span class="${className}">${text}</span>`;
        }
    );

    // Transform markdown spans with style attributes: [text]{style="color: color"}
    content = content.replace(
        /\[([^\]]+)\]\{style="color: ([^"]+)"\}/g,
        (match, text, color) => {
            // Map colors to semantic classes or components
            const colorMap = {
                'hf2': 'text-hf-secondary',
                'hf1': 'text-hf-primary'
            };

            const className = colorMap[color] || `text-${color}`;
            return `<span class="${className}">${text}</span>`;
        }
    );

    return content;
}

/**
 * Transform reference links to proper Astro internal links
 * @param {string} content - MDX content
 * @returns {string} - Content with transformed links
 */
function fixHtmlEscaping(content) {
    console.log('  🔧 Fixing HTML escaping in spans...');

    let fixedCount = 0;

    // Pattern 1: \<span id="..." style="..."\>\</span\>
    content = content.replace(/\\<span id="([^"]*)" style="([^"]*)"\\>\\<\/span\\>/g, (match, id, style) => {
        fixedCount++;
        // Fix common style issues like "position- absolute;" -> "position: absolute;"
        const cleanStyle = style.replace('position- absolute;', 'position: absolute;');
        return `<span id="${id}" style="${cleanStyle}"></span>`;
    });

    // Pattern 2: \<span class="..."\>...\</span\>
    content = content.replace(/\\<span class="([^"]*)"\\>([^\\]+)\\<\/span\\>/g, (match, className, text) => {
        fixedCount++;
        // Remove numbering like (1), (2), (3) from highlight spans
        let cleanText = text;
        if (className === 'highlight') {
            cleanText = text.replace(/^\(\d+\)\s*/, '');
        }
        return `<span class="${className}">${cleanText}</span>`;
    });

    // Pattern 3: HTML-encoded spans in paragraph tags
    // <p>&lt;span id="..." style="..."&gt;&lt;/span&gt;</p>
    content = content.replace(/<p>&lt;span id="([^"]*)" style="([^"]*)"&gt;&lt;\/span&gt;<\/p>/g, (match, id, style) => {
        fixedCount++;
        // Fix common style issues like "position- absolute;" -> "position: absolute;"
        const cleanStyle = style.replace('position- absolute;', 'position: absolute;');
        return `<span id="${id}" style="${cleanStyle}"></span>`;
    });

    // Pattern 4: HTML-encoded spans with class in paragraph tags
    // <p>&lt;span class="..."&gt;...&lt;/span&gt;</p>
    content = content.replace(/<p>&lt;span class="([^"]*)"&gt;([^&]*)&lt;\/span&gt;<\/p>/g, (match, className, text) => {
        fixedCount++;
        // Remove numbering like (1), (2), (3) from highlight spans
        let cleanText = text;
        if (className === 'highlight') {
            cleanText = text.replace(/^\(\d+\)\s*/, '');
        }
        return `<span class="${className}">${cleanText}</span>`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} escaped span(s)`);
    }

    return content;
}

function cleanHighlightNumbering(content) {
    console.log('  🔢 Removing numbering from highlight spans...');

    let cleanedCount = 0;
    // Clean numbering from non-escaped highlight spans too
    content = content.replace(/<span class="highlight">(\(\d+\)\s*)([^<]+)<\/span>/g, (match, numbering, text) => {
        cleanedCount++;
        return `<span class="highlight">${text}</span>`;
    });

    if (cleanedCount > 0) {
        console.log(`    ✅ Removed numbering from ${cleanedCount} highlight span(s)`);
    }

    return content;
}

function transformReferenceLinks(content) {
    console.log('  🔗 Transforming reference links...');

    // Transform Pandoc reference links: [text](#ref){reference-type="ref" reference="ref"}
    return content.replace(
        /\[([^\]]+)\]\((#[^)]+)\)\{[^}]*reference[^}]*\}/g,
        (match, text, href) => {
            return `[${text}](${href})`;
        }
    );
}


/**
 * Fix frontmatter and ensure proper MDX format
 * @param {string} content - MDX content
 * @param {string} latexContent - Original LaTeX content for metadata extraction
 * @returns {string} - Content with proper frontmatter
 */
function ensureFrontmatter(content, latexContent = '') {
    console.log('  📄 Ensuring proper frontmatter...');

    if (!content.startsWith('---')) {
        let frontmatter;

        if (latexContent) {
            // Extract metadata from LaTeX using dedicated module
            frontmatter = extractAndGenerateFrontmatter(latexContent);
            console.log('    ✅ Generated frontmatter from LaTeX metadata');
        } else {
            // Fallback frontmatter
            const currentDate = new Date().toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: '2-digit'
            });
            frontmatter = `---
title: "Research Article"
published: "${currentDate}"
tableOfContentsAutoCollapse: true
---

`;
            console.log('    ✅ Generated basic frontmatter');
        }

        return frontmatter + content;
    }

    return content;
}

/**
 * Fix mixed math delimiters like $`...`$ or `...`$
 * @param {string} content - MDX content
 * @returns {string} - Content with fixed math delimiters
 */
function fixMixedMathDelimiters(content) {
    console.log('  🔧 Fixing mixed math delimiters...');

    let fixedCount = 0;

    // Fix patterns like $`...`$ (mixed delimiters)
    content = content.replace(/\$`([^`]*)`\$/g, (match, mathContent) => {
        fixedCount++;
        return `$${mathContent}$`;
    });

    // Fix patterns like `...`$ (backtick start, dollar end)
    content = content.replace(/`([^`]*)`\$/g, (match, mathContent) => {
        fixedCount++;
        return `$${mathContent}$`;
    });

    // Fix patterns like $`...` (dollar start, backtick end - less common)
    content = content.replace(/\$`([^`]*)`(?!\$)/g, (match, mathContent) => {
        fixedCount++;
        return `$${mathContent}$`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} mixed math delimiter(s)`);
    }

    return content;
}

/**
 * Clean up orphaned math delimiters and fix mixed content
 * @param {string} content - MDX content
 * @returns {string} - Content with cleaned math blocks
 */
function cleanOrphanedMathDelimiters(content) {
    console.log('  🧹 Cleaning orphaned math delimiters...');
    console.log('    🔍 Content length:', content.length, 'chars');

    let fixedCount = 0;

    // Fix orphaned $$ that are alone on lines (but not part of display math blocks)
    // Only remove $$ that appear alone without corresponding closing $$
    content = content.replace(/^\$\$\s*$(?!\s*[\s\S]*?\$\$)/gm, () => {
        fixedCount++;
        return '';
    });

    // Fix backticks inside $$....$$ blocks (Pandoc artifact)
    const mathMatches = content.match(/\$\$([\s\S]*?)\$\$/g);
    console.log(`    🔍 Found ${mathMatches ? mathMatches.length : 0} math blocks`);

    content = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
        // More aggressive: remove ALL single backticks in math blocks (they shouldn't be there)
        let cleanedMath = mathContent;

        // Count backticks before
        const backticksBefore = (mathContent.match(/`/g) || []).length;

        if (backticksBefore > 0) {
            console.log(`    🔧 Found math block with ${backticksBefore} backtick(s)`);
        }

        // Remove all isolated backticks (not in pairs)
        cleanedMath = cleanedMath.replace(/`/g, '');

        const backticksAfter = (cleanedMath.match(/`/g) || []).length;

        if (backticksBefore > 0) {
            fixedCount++;
            console.log(`    🔧 Removed ${backticksBefore} backtick(s) from math block`);
            return `$$${cleanedMath}$$`;
        }
        return match;
    });

    // Fix escaped align in math blocks: \begin{align} -> \begin{align}
    content = content.replace(/\\begin\{align\}/g, (match) => {
        fixedCount++;
        return '\\begin{align}';
    });

    content = content.replace(/\\end\{align\}/g, (match) => {
        fixedCount++;
        return '\\end{align}';
    });

    // Fix cases where text gets mixed with math blocks
    // Pattern: ``` math ... ``` text ``` math 
    content = content.replace(/``` math\s*\n([\s\S]*?)\n```\s*([^`\n]*?)\s*``` math/g, (match, math1, text, math2) => {
        if (text.trim().length > 0 && !text.includes('```')) {
            fixedCount++;
            return '```' + ' math\n' + math1 + '\n```\n\n' + text.trim() + '\n\n```' + ' math';
        }
        return match;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} orphaned math delimiter(s)`);
    }

    return content;
}

/**
 * Clean newlines from single-dollar math blocks ($...$) ONLY
 * @param {string} content - MDX content
 * @returns {string} - Content with cleaned math blocks
 */
function cleanSingleLineMathNewlines(content) {
    console.log('  🔢 Cleaning newlines in single-dollar math blocks ($...$)...');

    let cleanedCount = 0;

    // ULTRA STRICT: Only target single dollar blocks ($...$) that contain newlines
    // Use dotall flag (s) to match newlines with .*, and ensure we don't match $$
    const cleanedContent = content.replace(/\$(?!\$)([\s\S]*?)\$(?!\$)/g, (match, mathContent) => {
        // Only process if the content contains newlines
        if (mathContent.includes('\n')) {
            cleanedCount++;

            // Remove ALL newlines and carriage returns, normalize whitespace
            const cleanedMath = mathContent
                .replace(/\n+/g, ' ')           // Replace all newlines with spaces
                .replace(/\r+/g, ' ')           // Replace carriage returns with spaces  
                .replace(/\s+/g, ' ')           // Normalize multiple spaces to single
                .trim();                        // Remove leading/trailing spaces

            return `$${cleanedMath}$`;
        }
        return match; // Keep original if no newlines
    });

    if (cleanedCount > 0) {
        console.log(`    ✅ Cleaned ${cleanedCount} single-dollar math block(s) with newlines`);
    }

    return cleanedContent;
}

/**
 * Add proper line breaks around display math blocks ($$...$$)
 * @param {string} content - MDX content
 * @returns {string} - Content with properly spaced display math
 */
function formatDisplayMathBlocks(content) {
    console.log('  📐 Formatting display math blocks with proper spacing...');

    let formattedCount = 0;

    // Find all $$...$$$ blocks (display math) and ensure proper line breaks
    // Very strict: only matches exactly $$ followed by content followed by $$
    const formattedContent = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, mathContent) => {
        formattedCount++;

        // Clean up the math content - trim whitespace but preserve structure
        const cleanedMath = mathContent.trim();

        // Return with proper line breaks before and after
        return `\n$$\n${cleanedMath}\n$$\n`;
    });

    if (formattedCount > 0) {
        console.log(`    ✅ Formatted ${formattedCount} display math block(s) with proper spacing`);
    }

    return formattedContent;
}

/**
 * Clean newlines from figcaption content
 * @param {string} content - MDX content
 * @returns {string} - Content with cleaned figcaptions
 */
function cleanFigcaptionNewlines(content) {
    console.log('  📝 Cleaning newlines in figcaption elements...');

    let cleanedCount = 0;

    // Find all <figcaption>...</figcaption> blocks and remove internal newlines
    const cleanedContent = content.replace(/<figcaption([^>]*)>([\s\S]*?)<\/figcaption>/g, (match, attributes, captionContent) => {
        // Only process if the content contains newlines
        if (captionContent.includes('\n')) {
            cleanedCount++;

            // Remove newlines and normalize whitespace
            const cleanedCaption = captionContent
                .replace(/\n+/g, ' ')           // Replace newlines with spaces
                .replace(/\s+/g, ' ')           // Normalize multiple spaces
                .trim();                        // Trim whitespace

            return `<figcaption${attributes}>${cleanedCaption}</figcaption>`;
        }

        return match; // Return unchanged if no newlines
    });

    if (cleanedCount > 0) {
        console.log(`    ✅ Cleaned ${cleanedCount} figcaption element(s)`);
    } else {
        console.log(`    ℹ️  No figcaption elements with newlines found`);
    }

    return cleanedContent;
}

/**
 * Remove HTML comments from MDX content
 * @param {string} content - MDX content
 * @returns {string} - Content without HTML comments
 */
function removeHtmlComments(content) {
    console.log('  🗑️  Removing HTML comments...');

    let removedCount = 0;

    // Remove all HTML comments <!-- ... -->
    const cleanedContent = content.replace(/<!--[\s\S]*?-->/g, () => {
        removedCount++;
        return '';
    });

    if (removedCount > 0) {
        console.log(`    ✅ Removed ${removedCount} HTML comment(s)`);
    }

    return cleanedContent;
}

/**
 * Clean up MDX-incompatible syntax  
 * @param {string} content - MDX content
 * @returns {string} - Cleaned content
 */
function cleanMdxSyntax(content) {
    console.log('  🧹 Cleaning MDX syntax...');

    return content
        // NOTE: Math delimiter fixing is now handled by fixMixedMathDelimiters()
        // Ensure proper spacing around JSX-like constructs
        .replace(/>\s*</g, '>\n<')
        // Remove problematic heading attributes - be more specific to avoid matching \begin{align}
        .replace(/^(#{1,6}\s+[^{#\n]+)\{[^}]+\}$/gm, '$1')
        // Fix escaped quotes in text
        .replace(/\\("|')/g, '$1');
}

/**
 * Main MDX processing function that applies all transformations
 * @param {string} content - Raw Markdown content
 * @param {string} latexContent - Original LaTeX content for metadata extraction
 * @returns {string} - Processed MDX content compatible with Astro
 */
function processMdxContent(content, latexContent = '') {
    console.log('🔧 Processing for Astro MDX compatibility...');

    // Clear previous tracking
    usedComponents.clear();
    imageImports.clear();

    let processedContent = content;

    // Apply each transformation step sequentially
    processedContent = ensureFrontmatter(processedContent, latexContent);
    processedContent = fixMixedMathDelimiters(processedContent);

    // Debug: check for $$ blocks after fixMixedMathDelimiters
    const mathBlocksAfterMixed = (processedContent.match(/\$\$([\s\S]*?)\$\$/g) || []).length;
    console.log(`    📊 Math blocks after mixed delimiters fix: ${mathBlocksAfterMixed}`);

    processedContent = cleanOrphanedMathDelimiters(processedContent);
    processedContent = cleanSingleLineMathNewlines(processedContent);
    processedContent = formatDisplayMathBlocks(processedContent);
    processedContent = removeHtmlComments(processedContent);
    processedContent = cleanMdxSyntax(processedContent);
    processedContent = convertSubfiguresToMultiFigure(processedContent);
    processedContent = transformImages(processedContent);
    processedContent = transformStyledSpans(processedContent);
    processedContent = transformReferenceLinks(processedContent);
    processedContent = fixHtmlEscaping(processedContent);
    processedContent = cleanHighlightNumbering(processedContent);
    processedContent = cleanFigcaptionNewlines(processedContent);

    // Add component imports at the end
    processedContent = addComponentImports(processedContent);

    return processedContent;
}

function convertToMdx(inputFile, outputFile) {
    console.log('📝 Modular Markdown to Astro MDX Converter');
    console.log(`📁 Input:  ${inputFile}`);
    console.log(`📁 Output: ${outputFile}`);

    // Check if input file exists
    if (!existsSync(inputFile)) {
        console.error(`❌ Input file not found: ${inputFile}`);
        process.exit(1);
    }

    try {
        console.log('🔄 Reading Markdown file...');
        const markdownContent = readFileSync(inputFile, 'utf8');

        // Try to read original LaTeX file for metadata extraction
        let latexContent = '';
        try {
            const inputDir = dirname(inputFile);
            const latexFile = join(inputDir, '..', 'input', 'main.tex');
            if (existsSync(latexFile)) {
                latexContent = readFileSync(latexFile, 'utf8');
            }
        } catch (error) {
            // Ignore LaTeX reading errors - we'll use fallback frontmatter
        }

        // Apply modular MDX processing
        const mdxContent = processMdxContent(markdownContent, latexContent);

        console.log('💾 Writing MDX file...');
        writeFileSync(outputFile, mdxContent);

        console.log(`✅ Conversion completed: ${outputFile}`);

        // Show file size
        const inputSize = Math.round(markdownContent.length / 1024);
        const outputSize = Math.round(mdxContent.length / 1024);
        console.log(`📊 Input: ${inputSize}KB → Output: ${outputSize}KB`);

    } catch (error) {
        console.error('❌ Conversion failed:');
        console.error(error.message);
        process.exit(1);
    }
}

export { convertToMdx };

function main() {
    const config = parseArgs();
    convertToMdx(config.input, config.output);
    console.log('🎉 MDX conversion completed!');
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
