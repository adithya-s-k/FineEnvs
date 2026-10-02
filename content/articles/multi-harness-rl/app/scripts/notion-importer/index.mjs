#!/usr/bin/env node

import { config } from 'dotenv';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, statSync, unlinkSync } from 'fs';
import { convertNotionToMarkdown } from './notion-converter.mjs';
import { convertToMdx } from './mdx-converter.mjs';
import { Client } from '@notionhq/client';

// Load environment variables from .env file (but don't override existing ones)
config({ override: false });

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Default configuration
const DEFAULT_INPUT = join(__dirname, 'input', 'pages.json');
const DEFAULT_OUTPUT = join(__dirname, 'output');
const ASTRO_CONTENT_PATH = join(__dirname, '..', '..', 'src', 'content', 'article.mdx');
const ASTRO_ASSETS_PATH = join(__dirname, '..', '..', 'src', 'content', 'assets', 'image');
const ASTRO_BIB_PATH = join(__dirname, '..', '..', 'src', 'content', 'bibliography.bib');
const STATIC_BIB_PATH = join(__dirname, 'static', 'bibliography.bib');

function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: DEFAULT_INPUT,
        output: DEFAULT_OUTPUT,
        clean: false,
        notionOnly: false,
        mdxOnly: false,
        token: process.env.NOTION_TOKEN,
        pageId: process.env.NOTION_PAGE_ID
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.split('=')[1];
        } else if (arg.startsWith('--output=')) {
            config.output = arg.split('=')[1];
        } else if (arg.startsWith('--token=')) {
            config.token = arg.split('=')[1];
        } else if (arg.startsWith('--page-id=')) {
            config.pageId = arg.split('=')[1];
        } else if (arg === '--clean') {
            config.clean = true;
        } else if (arg === '--notion-only') {
            config.notionOnly = true;
        } else if (arg === '--mdx-only') {
            config.mdxOnly = true;
        }
    }

    return config;
}

function showHelp() {
    console.log(`
🚀 Notion to MDX Toolkit

Usage:
  node index.mjs [options]

Options:
  --input=PATH      Input pages configuration file (default: input/pages.json)
  --output=PATH     Output directory (default: output/)
  --token=TOKEN     Notion API token (or set NOTION_TOKEN env var)
  --clean           Clean output directory before processing
  --notion-only     Only convert Notion to Markdown (skip MDX conversion)
  --mdx-only        Only convert existing Markdown to MDX
  --help, -h        Show this help

Environment Variables:
  NOTION_TOKEN      Your Notion integration token

Examples:
  # Full conversion workflow
  NOTION_TOKEN=your_token node index.mjs --clean

  # Only convert Notion pages to Markdown
  node index.mjs --notion-only --token=your_token

  # Only convert existing Markdown to MDX
  node index.mjs --mdx-only

  # Custom paths
  node index.mjs --input=my-pages.json --output=converted/ --token=your_token

Configuration File Format (pages.json):
{
  "pages": [
    {
      "id": "your-notion-page-id",
      "title": "Page Title",
      "slug": "page-slug"
    }
  ]
}

Workflow:
  1. Notion → Markdown (with media download)
  2. Markdown → MDX (with Astro components)
  3. Copy to Astro content directory
`);
}

function ensureDirectory(dir) {
    if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
    }
}

async function cleanDirectory(dir) {
    if (existsSync(dir)) {
        const { execSync } = await import('child_process');
        execSync(`rm -rf "${dir}"/*`, { stdio: 'inherit' });
    }
}

function readPagesConfig(inputFile) {
    try {
        const content = readFileSync(inputFile, 'utf8');
        return JSON.parse(content);
    } catch (error) {
        console.error(`❌ Error reading pages config: ${error.message}`);
        return { pages: [] };
    }
}

/**
 * Create a temporary pages.json from NOTION_PAGE_ID environment variable
 * Extracts title and generates slug from the Notion page
 */
async function createPagesConfigFromEnv(pageId, token, outputPath) {
    try {
        console.log('🔍 Fetching page info from Notion API...');
        const notion = new Client({ auth: token });
        const page = await notion.pages.retrieve({ page_id: pageId });

        // Extract title
        let title = 'Article';
        if (page.properties.title && page.properties.title.title && page.properties.title.title.length > 0) {
            title = page.properties.title.title[0].plain_text;
        } else if (page.properties.Name && page.properties.Name.title && page.properties.Name.title.length > 0) {
            title = page.properties.Name.title[0].plain_text;
        }

        // Generate slug from title
        const slug = title
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .trim();

        console.log(`    ✅ Found page: "${title}" (slug: ${slug})`);

        // Create pages config
        const pagesConfig = {
            pages: [{
                id: pageId,
                title: title,
                slug: slug
            }]
        };

        // Write to temporary file
        writeFileSync(outputPath, JSON.stringify(pagesConfig, null, 4));
        console.log(`    ✅ Created temporary pages config`);

        return pagesConfig;
    } catch (error) {
        console.error(`❌ Error fetching page from Notion: ${error.message}`);
        throw error;
    }
}

/**
 * Final cleanup function to remove exclude tags and unused imports
 * @param {string} content - MDX content
 * @returns {string} - Cleaned content
 */
function cleanupExcludeTagsAndImports(content) {
    let cleanedContent = content;
    let removedCount = 0;
    const removedImageVariables = new Set();

    // First, extract image variable names from exclude blocks before removing them
    const excludeBlocks = cleanedContent.match(/<exclude>[\s\S]*?<\/exclude>/g) || [];
    excludeBlocks.forEach(match => {
        const imageMatches = match.match(/src=\{([^}]+)\}/g);
        if (imageMatches) {
            imageMatches.forEach(imgMatch => {
                const varName = imgMatch.match(/src=\{([^}]+)\}/)?.[1];
                if (varName) {
                    removedImageVariables.add(varName);
                }
            });
        }
    });

    // Remove <exclude> tags and everything between them (including multiline)
    cleanedContent = cleanedContent.replace(/<exclude>[\s\S]*?<\/exclude>/g, (match) => {
        removedCount++;
        return '';
    });

    // Remove unused image imports that were only used in exclude blocks
    if (removedImageVariables.size > 0) {
        removedImageVariables.forEach(varName => {
            // Check if the variable is still used elsewhere in the content after removing exclude blocks
            const remainingUsage = cleanedContent.includes(`{${varName}}`) || cleanedContent.includes(`src={${varName}}`);

            if (!remainingUsage) {
                // Remove import lines for unused image variables
                // Pattern: import VarName from './assets/image/filename';
                const importPattern = new RegExp(`import\\s+${varName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s+from\\s+['"][^'"]+['"];?\\s*`, 'g');
                cleanedContent = cleanedContent.replace(importPattern, '');
                console.log(`    🗑️  Removed unused import: ${varName}`);
            }
        });
    }

    if (removedCount > 0) {
        console.log(`    🧹 Final cleanup: removed ${removedCount} exclude block(s) and ${removedImageVariables.size} unused import(s)`);
    }

    // Ensure there's always a blank line after imports before content starts
    // Find the last import line and ensure there's a blank line before the next non-empty line
    const lines = cleanedContent.split('\n');
    let lastImportIndex = -1;

    // Find the last import line
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim().startsWith('import ') && lines[i].trim().endsWith(';')) {
            lastImportIndex = i;
        }
    }

    // If we found imports, ensure there's a blank line after the last one
    if (lastImportIndex >= 0) {
        // Find the next non-empty line after the last import
        let nextNonEmptyIndex = lastImportIndex + 1;
        while (nextNonEmptyIndex < lines.length && lines[nextNonEmptyIndex].trim() === '') {
            nextNonEmptyIndex++;
        }

        // If there's no blank line between the last import and next content, add one
        if (nextNonEmptyIndex > lastImportIndex + 1) {
            // There are already blank lines, this is fine
        } else {
            // No blank line, add one
            lines.splice(nextNonEmptyIndex, 0, '');
        }

        cleanedContent = lines.join('\n');
    }

    return cleanedContent;
}

function copyToAstroContent(outputDir) {
    console.log('📋 Copying MDX files to Astro content directory...');

    try {
        // Ensure Astro directories exist
        mkdirSync(dirname(ASTRO_CONTENT_PATH), { recursive: true });
        mkdirSync(ASTRO_ASSETS_PATH, { recursive: true });

        // Copy MDX file
        const files = readdirSync(outputDir);
        const mdxFiles = files.filter(file => file.endsWith('.mdx'));
        if (mdxFiles.length > 0) {
            const mdxFile = join(outputDir, mdxFiles[0]); // Take the first MDX file
            // Read and write instead of copy to avoid EPERM issues
            let mdxContent = readFileSync(mdxFile, 'utf8');

            // Apply final cleanup to ensure no exclude tags or unused imports remain
            mdxContent = cleanupExcludeTagsAndImports(mdxContent);

            writeFileSync(ASTRO_CONTENT_PATH, mdxContent);
            console.log(`    ✅ Copied and cleaned MDX to ${ASTRO_CONTENT_PATH}`);
        }

        // Copy images from both media and external-images directories
        const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.tiff', '.html'];
        let totalImageCount = 0;

        function copyImagesRecursively(dir, sourceName) {
            if (!existsSync(dir)) return;

            const files = readdirSync(dir);
            for (const file of files) {
                const filePath = join(dir, file);
                const stat = statSync(filePath);

                if (stat.isDirectory()) {
                    copyImagesRecursively(filePath, sourceName);
                } else if (imageExtensions.some(ext => file.toLowerCase().endsWith(ext))) {
                    const filename = basename(filePath);
                    const destPath = join(ASTRO_ASSETS_PATH, filename);

                    try {
                        // Validate image by checking file size and basic structure
                        const stats = statSync(filePath);
                        if (stats.size === 0) {
                            console.log(`    ⚠️  Skipping empty image: ${filename}`);
                            return;
                        }

                        // Try to copy and validate the result
                        copyFileSync(filePath, destPath);

                        // Additional validation - check if the copied file has reasonable size
                        const destStats = statSync(destPath);
                        if (destStats.size === 0) {
                            console.log(`    ❌ Failed to copy corrupted image: ${filename}`);
                            // Remove the empty file
                            try {
                                unlinkSync(destPath);
                            } catch (e) { }
                            return;
                        }

                        console.log(`    ✅ Copied ${sourceName}: ${filename} (${destStats.size} bytes)`);
                        totalImageCount++;
                    } catch (error) {
                        console.log(`    ❌ Failed to copy ${filename}: ${error.message}`);
                    }
                }
            }
        }

        // Copy images from media directory (Notion images)
        const mediaDir = join(outputDir, 'media');
        copyImagesRecursively(mediaDir, 'Notion image');

        // Copy images from external-images directory (downloaded external images)
        const externalImagesDir = join(outputDir, 'external-images');
        copyImagesRecursively(externalImagesDir, 'external image');

        if (totalImageCount > 0) {
            console.log(`    ✅ Copied ${totalImageCount} total image(s) to ${ASTRO_ASSETS_PATH}`);
        }

        // Always update image paths and filter problematic references in MDX file
        if (existsSync(ASTRO_CONTENT_PATH)) {
            const mdxContent = readFileSync(ASTRO_CONTENT_PATH, 'utf8');
            let updatedContent = mdxContent.replace(/\.\/media\//g, './assets/image/');
            // Remove the subdirectory from image paths since we copy images directly to assets/image/
            updatedContent = updatedContent.replace(/\.\/assets\/image\/[^\/]+\//g, './assets/image/');

            // Check which images actually exist and remove references to missing/corrupted ones
            const imageReferences = updatedContent.match(/\.\/assets\/image\/[^\s\)]+/g) || [];
            const existingImages = existsSync(ASTRO_ASSETS_PATH) ? readdirSync(ASTRO_ASSETS_PATH).filter(f =>
                ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.tiff'].some(ext => f.toLowerCase().endsWith(ext))
            ) : [];

            for (const imgRef of imageReferences) {
                const filename = basename(imgRef);
                if (!existingImages.includes(filename)) {
                    console.log(`    ⚠️  Removing reference to missing/corrupted image: ${filename}`);
                    // Remove the entire image reference (both Image component and markdown syntax)
                    updatedContent = updatedContent.replace(
                        new RegExp(`<Image[^>]*src=["']${imgRef.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}["'][^>]*\/?>`, 'g'),
                        ''
                    );
                    updatedContent = updatedContent.replace(
                        new RegExp(`!\\[.*?\\]\\(${imgRef.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)`, 'g'),
                        ''
                    );
                }
            }

            writeFileSync(ASTRO_CONTENT_PATH, updatedContent);
            console.log(`    ✅ Updated image paths and filtered problematic references in MDX file`);
        }

        // Copy static bibliography.bib if it exists, otherwise create empty
        if (existsSync(STATIC_BIB_PATH)) {
            const bibContent = readFileSync(STATIC_BIB_PATH, 'utf8');
            writeFileSync(ASTRO_BIB_PATH, bibContent);
            console.log(`    ✅ Copied static bibliography from ${STATIC_BIB_PATH}`);
        } else {
            writeFileSync(ASTRO_BIB_PATH, '');
            console.log(`    ✅ Created empty bibliography (no static file found)`);
        }

    } catch (error) {
        console.warn(`    ⚠️  Failed to copy to Astro: ${error.message}`);
    }
}


async function main() {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        process.exit(0);
    }

    const config = parseArgs();

    console.log('🚀 Notion to MDX Toolkit');
    console.log('========================');

    try {
        // Prepare input config file
        let inputConfigFile = config.input;
        let pageIdFromEnv = null;

        // If NOTION_PAGE_ID is provided via env var, create temporary pages.json
        if (config.pageId && config.token) {
            console.log('✨ Using NOTION_PAGE_ID from environment variable');
            const tempConfigPath = join(config.output, '.temp-pages.json');
            ensureDirectory(config.output);
            await createPagesConfigFromEnv(config.pageId, config.token, tempConfigPath);
            inputConfigFile = tempConfigPath;
            pageIdFromEnv = config.pageId;
        } else if (!existsSync(config.input)) {
            console.error(`❌ No NOTION_PAGE_ID environment variable and no pages.json found at: ${config.input}`);
            console.log('💡 Either set NOTION_PAGE_ID env var or create input/pages.json');
            process.exit(1);
        }

        // Always clean output directory to avoid conflicts with previous imports
        console.log('🧹 Cleaning output directory to avoid conflicts...');
        await cleanDirectory(config.output);

        // Clean assets/image directory and ensure proper permissions
        console.log('🧹 Cleaning assets/image directory and setting permissions...');
        if (existsSync(ASTRO_ASSETS_PATH)) {
            await cleanDirectory(ASTRO_ASSETS_PATH);
        } else {
            ensureDirectory(ASTRO_ASSETS_PATH);
        }

        // Ensure proper permissions for assets directory
        const { execSync } = await import('child_process');
        try {
            execSync(`chmod -R 755 "${ASTRO_ASSETS_PATH}"`, { stdio: 'inherit' });
            console.log('    ✅ Set permissions for assets/image directory');
        } catch (error) {
            console.log('    ⚠️  Could not set permissions (non-critical):', error.message);
        }

        if (config.mdxOnly) {
            // Only convert existing Markdown to MDX
            console.log('📝 MDX conversion only mode');
            await convertToMdx(config.output, config.output);
            copyToAstroContent(config.output);

        } else if (config.notionOnly) {
            // Only convert Notion to Markdown
            console.log('📄 Notion conversion only mode');
            await convertNotionToMarkdown(inputConfigFile, config.output, config.token);

        } else {
            // Full workflow
            console.log('🔄 Full conversion workflow');

            // Step 1: Convert Notion to Markdown
            console.log('\n📄 Step 1: Converting Notion pages to Markdown...');
            await convertNotionToMarkdown(inputConfigFile, config.output, config.token);

            // Step 2: Convert Markdown to MDX with Notion metadata
            console.log('\n📝 Step 2: Converting Markdown to MDX...');
            const pagesConfig = readPagesConfig(inputConfigFile);
            const firstPage = pagesConfig.pages && pagesConfig.pages.length > 0 ? pagesConfig.pages[0] : null;
            const pageId = pageIdFromEnv || (firstPage ? firstPage.id : null);
            await convertToMdx(config.output, config.output, pageId, config.token);

            // Step 3: Copy to Astro content directory
            console.log('\n📋 Step 3: Copying to Astro content directory...');
            copyToAstroContent(config.output);
        }

        console.log('\n🎉 Conversion completed successfully!');

    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

// Export functions for use as module
export { convertNotionToMarkdown, convertToMdx };

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
