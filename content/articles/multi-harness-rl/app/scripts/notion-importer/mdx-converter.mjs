#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'fs';
import { join, dirname, basename, extname } from 'path';
import { fileURLToPath } from 'url';
import matter from 'gray-matter';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const DEFAULT_INPUT = join(__dirname, 'output');
const DEFAULT_OUTPUT = join(__dirname, 'output');
const STATIC_FRONTMATTER_PATH = join(__dirname, 'static', 'frontmatter.mdx');

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
📝 Notion Markdown to MDX Converter

Usage:
  node mdx-converter.mjs [options]

Options:
  --input=PATH     Input directory or file (default: ${DEFAULT_INPUT})
  --output=PATH    Output directory (default: ${DEFAULT_OUTPUT})
  --help, -h       Show this help

Examples:
  # Convert all markdown files in output directory
  node mdx-converter.mjs

  # Convert specific file
  node mdx-converter.mjs --input=article.md --output=converted/

  # Convert directory
  node mdx-converter.mjs --input=markdown-files/ --output=mdx-files/
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
 * Track which Astro components are used during transformations
 */
const usedComponents = new Set();

/**
 * Track individual image imports needed
 */
const imageImports = new Map(); // src -> varName

/**
 * Track external images that need to be downloaded
 */
const externalImagesToDownload = new Map(); // url -> localPath

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

/**
 * Check if a URL is an external URL (HTTP/HTTPS)
 * @param {string} url - URL to check
 * @returns {boolean} - True if it's an external URL
 */
function isExternalImageUrl(url) {
    try {
        const urlObj = new URL(url);
        // Just check if it's HTTP/HTTPS - we'll try to download everything
        return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
    } catch {
        return false;
    }
}

/**
 * Extract image URL from Twitter/X page
 * @param {string} tweetUrl - URL of the tweet
 * @returns {Promise<string|null>} - URL of the image or null if not found
 */
async function extractTwitterImageUrl(tweetUrl) {
    try {
        const response = await fetch(tweetUrl, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });

        if (!response.ok) {
            return null;
        }

        const html = await response.text();

        // Try to find image URLs in meta tags (Twitter Card)
        const metaImageMatch = html.match(/<meta property="og:image" content="([^"]+)"/);
        if (metaImageMatch) {
            let imageUrl = metaImageMatch[1];
            // Try to get the large version
            if (imageUrl.includes('?')) {
                imageUrl = imageUrl.split('?')[0] + '?format=jpg&name=large';
            }
            return imageUrl;
        }

        // Fallback: try to find pbs.twimg.com URLs in the HTML
        const pbsMatch = html.match(/https:\/\/pbs\.twimg\.com\/media\/([^"?]+)/);
        if (pbsMatch) {
            return `https://pbs.twimg.com/media/${pbsMatch[1]}?format=jpg&name=large`;
        }

        return null;
    } catch (error) {
        console.log(`    ⚠️  Failed to extract Twitter image: ${error.message}`);
        return null;
    }
}

/**
 * Download an external URL and save it locally
 * @param {string} imageUrl - External URL
 * @param {string} outputDir - Directory to save the file
 * @returns {Promise<string>} - Local path to the downloaded file
 */
async function downloadExternalImage(imageUrl, outputDir) {
    try {
        console.log(`    🌐 Downloading external URL: ${imageUrl}`);

        // Create output directory if it doesn't exist
        if (!existsSync(outputDir)) {
            mkdirSync(outputDir, { recursive: true });
        }

        let actualImageUrl = imageUrl;

        // Check if it's a Twitter/X URL
        if (imageUrl.includes('twitter.com/') || imageUrl.includes('x.com/')) {
            console.log(`    🐦 Detected Twitter/X URL, attempting to extract image...`);
            const extractedUrl = await extractTwitterImageUrl(imageUrl);
            if (extractedUrl) {
                actualImageUrl = extractedUrl;
                console.log(`    ✅ Extracted image URL: ${extractedUrl}`);
            } else {
                console.log(`    ⚠️  Could not automatically extract image from Twitter/X`);
                console.log(`    💡 Manual download required:`);
                console.log(`       1. Open ${imageUrl} in your browser`);
                console.log(`       2. Right-click on the image and "Save image as..."`);
                console.log(`       3. Save it to: app/src/content/assets/image/`);
                throw new Error('Twitter/X images require manual download');
            }
        }

        // Generate filename from URL
        const urlObj = new URL(actualImageUrl);
        const pathname = urlObj.pathname;

        // Determine file extension - try to get it from URL, default to jpg
        let extension = 'jpg';
        if (pathname.includes('.')) {
            const urlExtension = pathname.split('.').pop().toLowerCase();
            if (['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp', 'bmp', 'tiff'].includes(urlExtension)) {
                extension = urlExtension;
            }
        }

        // Generate unique filename
        const filename = `external_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.${extension}`;
        const localPath = join(outputDir, filename);

        // Try to download the URL
        const response = await fetch(actualImageUrl, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const buffer = await response.buffer();

        // Validate that we actually got data
        if (buffer.length === 0) {
            throw new Error('Empty response');
        }

        // Validate that it's actually an image, not HTML
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('text/html')) {
            throw new Error('Downloaded content is HTML, not an image');
        }

        // Save to local file
        writeFileSync(localPath, buffer);

        console.log(`    ✅ Downloaded: ${filename} (${buffer.length} bytes)`);
        return localPath;

    } catch (error) {
        console.log(`    ❌ Failed to download ${imageUrl}: ${error.message}`);
        throw error;
    }
}

/**
 * Process external images in content and download them
 * @param {string} content - Markdown content
 * @param {string} outputDir - Directory to save downloaded images
 * @returns {Promise<string>} - Content with external images replaced by local paths
 */
async function processExternalImages(content, outputDir) {
    console.log('  🌐 Processing external images...');

    let processedCount = 0;
    let downloadedCount = 0;

    // Find all external image URLs in markdown format: ![alt](url)
    const externalImageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    let match;
    const externalImages = new Map(); // url -> alt text

    // First pass: collect all external image URLs
    while ((match = externalImageRegex.exec(content)) !== null) {
        const alt = match[1];
        const url = match[2];

        if (isExternalImageUrl(url)) {
            externalImages.set(url, alt);
            console.log(`    🔍 Found external image: ${url}`);
        }
    }

    if (externalImages.size === 0) {
        console.log('    ℹ️  No external images found');
        return content;
    }

    // Second pass: download images and replace URLs
    let processedContent = content;

    for (const [url, alt] of externalImages) {
        try {
            // Download the image
            const localPath = await downloadExternalImage(url, outputDir);
            const relativePath = `./assets/image/${basename(localPath)}`;

            // Replace the URL in content
            processedContent = processedContent.replace(
                new RegExp(`!\\[${alt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\]\\(${url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`, 'g'),
                `![${alt}](${relativePath})`
            );

            downloadedCount++;
            processedCount++;

        } catch (error) {
            console.log(`    ⚠️  Skipping external image due to download failure: ${url}`);
        }
    }

    if (downloadedCount > 0) {
        console.log(`    ✅ Downloaded ${downloadedCount} external image(s)`);
    }

    return processedContent;
}

/**
 * Detect and track Astro components used in the content
 * @param {string} content - MDX content
 */
function detectAstroComponents(content) {
    console.log('  🔍 Detecting Astro components in content...');

    let detectedCount = 0;

    // Known Astro components that should be auto-imported
    const knownComponents = [
        'HtmlEmbed', 'Image', 'Note', 'Sidenote', 'Wide', 'FullWidth',
        'Accordion', 'Quote', 'Reference', 'Glossary', 'Stack', 'ThemeToggle',
        'RawHtml', 'HfUser'
    ];

    // Find all JSX elements that look like Astro components
    // Pattern: <ComponentName ... />
    const componentMatches = content.match(/<([A-Z][a-zA-Z0-9]*)\s*[^>]*\/?>/g);

    if (componentMatches) {
        for (const match of componentMatches) {
            // Extract component name from the JSX element
            const componentMatch = match.match(/<([A-Z][a-zA-Z0-9]*)/);
            if (componentMatch) {
                const componentName = componentMatch[1];

                // Only track known Astro components (skip HTML elements)
                if (knownComponents.includes(componentName) && !usedComponents.has(componentName)) {
                    usedComponents.add(componentName);
                    detectedCount++;
                    console.log(`    📦 Found component: ${componentName}`);
                }
            }
        }
    }

    if (detectedCount > 0) {
        console.log(`    ✅ Detected ${detectedCount} new Astro component(s)`);
    } else {
        console.log(`    ℹ️  No new Astro components detected`);
    }
}

/**
 * Add required component imports to the frontmatter
 * @param {string} content - MDX content
 * @returns {string} - Content with component imports
 */
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
        return content.slice(0, frontmatterEnd) + '\n\n' + importBlock + '\n\n' + content.slice(frontmatterEnd);
    } else {
        // No frontmatter, add at beginning
        return importBlock + '\n\n' + content;
    }
}


/**
 * Load static frontmatter from file
 * @returns {object} - Static frontmatter data
 */
function loadStaticFrontmatter() {
    try {
        if (existsSync(STATIC_FRONTMATTER_PATH)) {
            const staticContent = readFileSync(STATIC_FRONTMATTER_PATH, 'utf8');
            const { data } = matter(staticContent);
            console.log('    ✅ Loaded static frontmatter from file');
            return data;
        }
        console.log('    ℹ️  No static frontmatter file found');
        return {};
    } catch (error) {
        console.log(`    ⚠️  Failed to load static frontmatter: ${error.message}`);
        return {};
    }
}

/**
 * Ensure proper frontmatter for MDX using static file first, then existing data
 * @param {string} content - MDX content
 * @param {string} pageId - Notion page ID (optional, kept for compatibility but ignored)
 * @param {string} notionToken - Notion API token (optional, kept for compatibility but ignored)
 * @returns {string} - Content with proper frontmatter
 */
async function ensureFrontmatter(content, pageId = null, notionToken = null) {
    console.log('  📄 Ensuring proper frontmatter...');

    // Load static frontmatter first (highest priority)
    const staticData = loadStaticFrontmatter();

    if (!content.startsWith('---')) {
        // No frontmatter in content, use static + basic defaults
        let baseData = { ...staticData };

        // Add basic defaults for required fields if not in static
        if (!baseData.title) baseData.title = 'Article';
        if (!baseData.published) {
            baseData.published = new Date().toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: '2-digit'
            });
        }
        if (baseData.tableOfContentsAutoCollapse === undefined) {
            baseData.tableOfContentsAutoCollapse = true;
        }

        const frontmatter = matter.stringify('', baseData);
        console.log('    ✅ Applied static frontmatter to content without frontmatter');
        return frontmatter + content;
    }

    // Parse existing frontmatter and merge with static (static takes priority)
    try {
        const { data: existingData, content: body } = matter(content);

        // Merge: existing data first, then static data overrides
        const mergedData = { ...existingData, ...staticData };

        // Ensure required fields if still missing after merge
        if (!mergedData.title) mergedData.title = 'Article';
        if (!mergedData.published) {
            mergedData.published = new Date().toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: '2-digit'
            });
        }
        if (mergedData.tableOfContentsAutoCollapse === undefined) {
            mergedData.tableOfContentsAutoCollapse = true;
        }

        const enhancedContent = matter.stringify(body, mergedData);
        console.log('    ✅ Merged static and existing frontmatter');
        return enhancedContent;
    } catch (error) {
        console.log('    ⚠️  Could not parse frontmatter, keeping as is');
        return content;
    }
}

/**
 * Generate basic frontmatter
 * @returns {string} - Basic frontmatter
 */
function generateBasicFrontmatter() {
    const currentDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: '2-digit'
    });
    return `---
title: "Notion Article"
published: "${currentDate}"
tableOfContentsAutoCollapse: true
---

`;
}


/**
 * Check if a line is a table line
 * @param {string} line - Line to check
 * @returns {boolean} - True if it's a table line
 */
function isTableLine(line) {
    const trimmed = line.trim();
    return trimmed.startsWith('|') && trimmed.endsWith('|');
}

/**
 * Check if a line is a list item
 * @param {string} line - Line to check
 * @returns {boolean} - True if it's a list item
 */
function isListItem(line) {
    const trimmed = line.trim();
    // Match: * -, + (bullet points) or 1. 2. 3. (numbered lists)
    return /^\s*[\*\-\+]\s/.test(trimmed) || /^\s*\d+\.\s/.test(trimmed);
}

/**
 * Add a blank line after each markdown table and list
 * @param {string} content - MDX content
 * @returns {string} - Content with blank lines after tables and lists
 */
function addBlankLineAfterTablesAndLists(content) {
    console.log('  📋 Adding blank lines after tables and lists...');

    let addedTableCount = 0;
    let addedListCount = 0;
    const lines = content.split('\n');
    const result = [];

    for (let i = 0; i < lines.length; i++) {
        result.push(lines[i]);

        // Check if current line is the end of a table
        if (isTableLine(lines[i])) {
            // Look ahead to see if this is the last line of a table
            let isLastTableLine = false;

            // Check if next line is empty or doesn't start with |
            if (i + 1 >= lines.length ||
                lines[i + 1].trim() === '' ||
                !isTableLine(lines[i + 1])) {

                // Look back to find if we're actually inside a table
                let tableLineCount = 0;
                for (let j = i; j >= 0 && isTableLine(lines[j]); j--) {
                    tableLineCount++;
                }

                // Only add blank line if we found at least 2 table lines (making it a real table)
                if (tableLineCount >= 2) {
                    isLastTableLine = true;
                }
            }

            if (isLastTableLine) {
                addedTableCount++;
                result.push(''); // Add blank line
            }
        }
        // Check if current line is the end of a list
        else if (isListItem(lines[i])) {
            // Look ahead to see if this is the last line of a list
            let isLastListItem = false;

            // Check if next line is empty or doesn't start with list marker
            if (i + 1 >= lines.length ||
                lines[i + 1].trim() === '' ||
                !isListItem(lines[i + 1])) {
                isLastListItem = true;
            }

            if (isLastListItem) {
                addedListCount++;
                result.push(''); // Add blank line
            }
        }
    }

    if (addedTableCount > 0 || addedListCount > 0) {
        console.log(`    ✅ Added blank line after ${addedTableCount} table(s) and ${addedListCount} list(s)`);
    } else {
        console.log('    ℹ️  No tables or lists found to process');
    }

    return result.join('\n');
}

/**
 * Transform markdown images to Image components
 * @param {string} content - Markdown content
 * @returns {string} - Content with Image components
 */
function transformMarkdownImages(content) {
    console.log('  🖼️  Transforming markdown images to Image components...');

    let transformedCount = 0;

    // Transform markdown images: ![alt](src) -> <Image src={varName} alt="alt" />
    content = content.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
        transformedCount++;

        // Clean up the src path - remove /media/ prefix and use relative path
        let cleanSrc = src;
        if (src.startsWith('/media/')) {
            cleanSrc = src.replace('/media/', './assets/image/');
        }

        // Generate variable name for the image import
        const varName = generateImageVarName(cleanSrc);

        // Add to imageImports if not already present
        if (!imageImports.has(cleanSrc)) {
            imageImports.set(cleanSrc, varName);
        }

        // Extract filename for alt text if none provided
        const finalAlt = alt || src.split('/').pop().split('.')[0];

        return `<Image src={${varName}} alt="${finalAlt}" />`;
    });

    if (transformedCount > 0) {
        console.log(`    ✅ Transformed ${transformedCount} markdown image(s) to Image components with imports`);
    } else {
        console.log('    ℹ️  No markdown images found to transform');
    }

    return content;
}

/**
 * Add proper spacing around Astro components
 * @param {string} content - MDX content
 * @returns {string} - Content with proper spacing around components
 */
function addSpacingAroundComponents(content) {
    console.log('  📏 Adding spacing around Astro components...');

    let processedContent = content;
    let spacingCount = 0;

    // Known Astro components that should have spacing
    const knownComponents = [
        'HtmlEmbed', 'Image', 'Note', 'Sidenote', 'Wide', 'FullWidth',
        'Accordion', 'Quote', 'Reference', 'Glossary', 'Stack', 'ThemeToggle',
        'RawHtml', 'HfUser', 'Figure'
    ];

    // Process each component type
    for (const component of knownComponents) {
        // Pattern for components with content: <Component>...</Component>
        // Process this first to handle the complete component structure
        const withContentPattern = new RegExp(`(<${component}[^>]*>)([\\s\\S]*?)(<\\/${component}>)`, 'g');
        processedContent = processedContent.replace(withContentPattern, (match, openTag, content, closeTag) => {
            spacingCount++;
            // Ensure blank line before opening tag and after closing tag
            // Also ensure closing tag is on its own line
            const trimmedContent = content.trim();
            return `\n\n${openTag}\n${trimmedContent}\n${closeTag}\n\n`;
        });

        // Pattern for self-closing components: <Component ... />
        const selfClosingPattern = new RegExp(`(<${component}[^>]*\\/?>)`, 'g');
        processedContent = processedContent.replace(selfClosingPattern, (match) => {
            spacingCount++;
            return `\n\n${match}\n\n`;
        });
    }

    // Clean up excessive newlines (more than 2 consecutive)
    processedContent = processedContent.replace(/\n{3,}/g, '\n\n');

    if (spacingCount > 0) {
        console.log(`    ✅ Added spacing around ${spacingCount} component(s)`);
    } else {
        console.log('    ℹ️  No components found to add spacing around');
    }

    return processedContent;
}

/**
 * Fix smart quotes (curly quotes) and replace them with straight quotes
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed quotes
 */
function fixSmartQuotes(content) {
    console.log('  ✏️  Fixing smart quotes (curly quotes)...');

    let fixedCount = 0;
    const originalContent = content;

    // Replace opening smart double quotes (\u201C) with straight quotes (")
    content = content.replace(/\u201C/g, '"');

    // Replace closing smart double quotes (\u201D) with straight quotes (")
    content = content.replace(/\u201D/g, '"');

    // Replace opening smart single quotes (\u2018) with straight quotes (')
    content = content.replace(/\u2018/g, "'");

    // Replace closing smart single quotes (\u2019) with straight quotes (')
    content = content.replace(/\u2019/g, "'");

    // Count the number of replacements made
    fixedCount = 0;
    for (let i = 0; i < originalContent.length; i++) {
        const char = originalContent[i];
        if (char === '\u201C' || char === '\u201D' || char === '\u2018' || char === '\u2019') {
            fixedCount++;
        }
    }

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} smart quote(s)`);
    } else {
        console.log('    ℹ️  No smart quotes found');
    }

    return content;
}

/**
 * Main MDX processing function that applies all transformations
 * @param {string} content - Raw Markdown content
 * @param {string} pageId - Notion page ID (optional)
 * @param {string} notionToken - Notion API token (optional)
 * @param {string} outputDir - Output directory for downloaded images (optional)
 * @returns {string} - Processed MDX content compatible with Astro
 */
async function processMdxContent(content, pageId = null, notionToken = null, outputDir = null) {
    console.log('🔧 Processing for Astro MDX compatibility...');

    // Clear previous tracking
    usedComponents.clear();
    imageImports.clear();
    externalImagesToDownload.clear();

    let processedContent = content;

    // Fix smart quotes first
    processedContent = fixSmartQuotes(processedContent);

    // Process external images first (before other transformations)
    if (outputDir) {
        // Create a temporary external images directory in the output folder
        const externalImagesDir = join(outputDir, 'external-images');
        processedContent = await processExternalImages(processedContent, externalImagesDir);
    }

    // Apply essential steps only
    processedContent = await ensureFrontmatter(processedContent, pageId, notionToken);

    // Add blank lines after tables and lists
    processedContent = addBlankLineAfterTablesAndLists(processedContent);

    // Transform markdown images to Image components
    processedContent = transformMarkdownImages(processedContent);

    // Add spacing around Astro components
    processedContent = addSpacingAroundComponents(processedContent);

    // Detect Astro components used in the content before adding imports
    detectAstroComponents(processedContent);

    // Add component imports at the end
    processedContent = addComponentImports(processedContent);

    return processedContent;
}

/**
 * Convert a single markdown file to MDX
 * @param {string} inputFile - Input markdown file
 * @param {string} outputDir - Output directory
 * @param {string} pageId - Notion page ID (optional)
 * @param {string} notionToken - Notion API token (optional)
 */
async function convertFileToMdx(inputFile, outputDir, pageId = null, notionToken = null) {
    const filename = basename(inputFile, '.md');
    const outputFile = join(outputDir, `${filename}.mdx`);

    console.log(`📝 Converting: ${basename(inputFile)} → ${basename(outputFile)}`);

    try {
        const markdownContent = readFileSync(inputFile, 'utf8');
        const mdxContent = await processMdxContent(markdownContent, pageId, notionToken, outputDir);
        writeFileSync(outputFile, mdxContent);

        console.log(`    ✅ Converted: ${outputFile}`);

        // Show file size
        const inputSize = Math.round(markdownContent.length / 1024);
        const outputSize = Math.round(mdxContent.length / 1024);
        console.log(`    📊 Input: ${inputSize}KB → Output: ${outputSize}KB`);

    } catch (error) {
        console.error(`    ❌ Failed to convert ${inputFile}: ${error.message}`);
    }
}

/**
 * Convert all markdown files in a directory to MDX
 * @param {string} inputPath - Input path (file or directory)
 * @param {string} outputDir - Output directory
 * @param {string} pageId - Notion page ID (optional)
 * @param {string} notionToken - Notion API token (optional)
 */
async function convertToMdx(inputPath, outputDir, pageId = null, notionToken = null) {
    console.log('📝 Notion Markdown to Astro MDX Converter');
    console.log(`📁 Input:  ${inputPath}`);
    console.log(`📁 Output: ${outputDir}`);

    // Check if input exists
    if (!existsSync(inputPath)) {
        console.error(`❌ Input not found: ${inputPath}`);
        process.exit(1);
    }

    try {
        // Ensure output directory exists
        if (!existsSync(outputDir)) {
            mkdirSync(outputDir, { recursive: true });
        }

        let filesToConvert = [];

        if (statSync(inputPath).isDirectory()) {
            // Convert all .md files in directory
            const files = readdirSync(inputPath);
            filesToConvert = files
                .filter(file => file.endsWith('.md') && !file.includes('.raw.md'))
                .map(file => join(inputPath, file));
        } else if (inputPath.endsWith('.md')) {
            // Convert single file
            filesToConvert = [inputPath];
        } else {
            console.error('❌ Input must be a .md file or directory containing .md files');
            process.exit(1);
        }

        if (filesToConvert.length === 0) {
            console.log('ℹ️  No .md files found to convert');
            return;
        }

        console.log(`🔄 Found ${filesToConvert.length} file(s) to convert`);

        // Convert each file
        for (const file of filesToConvert) {
            await convertFileToMdx(file, outputDir, pageId, notionToken);
        }

        console.log(`✅ Conversion completed! ${filesToConvert.length} file(s) processed`);

    } catch (error) {
        console.error('❌ Conversion failed:', error.message);
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
