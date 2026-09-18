#!/usr/bin/env node

import { config } from 'dotenv';
import { Client } from '@notionhq/client';
import { NotionConverter } from 'notion-to-md';
import { DefaultExporter } from 'notion-to-md/plugins/exporter';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { postProcessMarkdown } from './post-processor.mjs';

// Load environment variables from .env file (but don't override existing ones)
config({ override: false });

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configuration
const DEFAULT_INPUT = join(__dirname, 'input', 'pages.json');
const DEFAULT_OUTPUT = join(__dirname, 'output');

function parseArgs() {
    const args = process.argv.slice(2);
    const config = {
        input: DEFAULT_INPUT,
        output: DEFAULT_OUTPUT,
        clean: false,
        token: process.env.NOTION_TOKEN
    };

    for (const arg of args) {
        if (arg.startsWith('--input=')) {
            config.input = arg.split('=')[1];
        } else if (arg.startsWith('--output=')) {
            config.output = arg.split('=')[1];
        } else if (arg.startsWith('--token=')) {
            config.token = arg.split('=')[1];
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

function loadPagesConfig(configFile) {
    if (!existsSync(configFile)) {
        console.error(`❌ Configuration file not found: ${configFile}`);
        console.log('📝 Create a pages.json file with your Notion page IDs:');
        console.log(`
{
  "pages": [
    {
      "id": "your-notion-page-id-1",
      "title": "Page Title 1",
      "slug": "page-1"
    },
    {
      "id": "your-notion-page-id-2", 
      "title": "Page Title 2",
      "slug": "page-2"
    }
  ]
}
        `);
        process.exit(1);
    }

    try {
        const config = JSON.parse(readFileSync(configFile, 'utf8'));
        return config.pages || [];
    } catch (error) {
        console.error(`❌ Error reading configuration: ${error.message}`);
        process.exit(1);
    }
}

/**
 * Convert a single Notion page to Markdown with advanced media handling
 * @param {Object} notion - Notion client
 * @param {string} pageId - Notion page ID
 * @param {string} outputDir - Output directory
 * @param {string} pageTitle - Page title for file naming
 * @returns {Promise<string>} - Path to generated markdown file
 */
async function convertNotionPage(notion, pageId, outputDir, pageTitle) {
    console.log(`📄 Converting Notion page: ${pageTitle} (${pageId})`);

    try {
        // Create media directory for this page
        const mediaDir = join(outputDir, 'media', pageId);
        ensureDirectory(mediaDir);

        // Configure the DefaultExporter to save to a file
        const outputFile = join(outputDir, `${pageTitle}.md`);
        const exporter = new DefaultExporter({
            outputType: 'file',
            outputPath: outputFile,
        });

        // Create the converter with media downloading strategy
        const n2m = new NotionConverter(notion)
            .withExporter(exporter)
            // Download media to local directory with path transformation
            .downloadMediaTo({
                outputDir: mediaDir,
                // Transform paths to be web-accessible
                transformPath: (localPath) => `/media/${pageId}/${basename(localPath)}`,
            });

        // Convert the page
        const result = await n2m.convert(pageId);

        console.log(`    ✅ Converted to: ${outputFile}`);
        console.log(`    📊 Content length: ${result.content.length} characters`);
        console.log(`    🖼️  Media saved to: ${mediaDir}`);

        return outputFile;

    } catch (error) {
        console.error(`    ❌ Failed to convert page ${pageId}: ${error.message}`);
        throw error;
    }
}

/**
 * Process Notion pages with advanced configuration
 * @param {string} inputFile - Path to pages configuration
 * @param {string} outputDir - Output directory
 * @param {string} notionToken - Notion API token
 */
export async function convertNotionToMarkdown(inputFile, outputDir, notionToken) {
    console.log('🚀 Notion to Markdown Converter');
    console.log(`📁 Input:  ${inputFile}`);
    console.log(`📁 Output: ${outputDir}`);

    // Validate Notion token
    if (!notionToken) {
        console.error('❌ NOTION_TOKEN not found. Please set it as environment variable or use --token=YOUR_TOKEN');
        process.exit(1);
    }

    // Ensure output directory exists
    ensureDirectory(outputDir);

    try {
        // Initialize Notion client
        const notion = new Client({
            auth: notionToken,
        });

        // Load pages configuration
        const pages = loadPagesConfig(inputFile);
        console.log(`📋 Found ${pages.length} page(s) to convert`);

        const convertedFiles = [];

        // Convert each page
        for (const page of pages) {
            try {
                const outputFile = await convertNotionPage(
                    notion,
                    page.id,
                    outputDir,
                    page.slug || page.title?.toLowerCase().replace(/\s+/g, '-') || page.id
                );
                convertedFiles.push(outputFile);
            } catch (error) {
                console.error(`❌ Failed to convert page ${page.id}: ${error.message}`);
                // Continue with other pages
            }
        }

        // Post-process all converted files and create one intermediate file
        console.log('🔧 Post-processing converted files...');
        for (const file of convertedFiles) {
            try {
                // Read the raw markdown from notion-to-md
                let rawContent = readFileSync(file, 'utf8');

                // Create intermediate file: raw markdown (from notion-to-md)
                const rawFile = file.replace('.md', '.raw.md');
                writeFileSync(rawFile, rawContent);
                console.log(`    📄 Created raw markdown: ${basename(rawFile)}`);

                // Apply post-processing with Notion client for page inclusion
                let processedContent = await postProcessMarkdown(rawContent, notion, notionToken);
                writeFileSync(file, processedContent);
                console.log(`    ✅ Post-processed: ${basename(file)}`);
            } catch (error) {
                console.error(`    ❌ Failed to post-process ${file}: ${error.message}`);
            }
        }

        console.log(`✅ Conversion completed! ${convertedFiles.length} file(s) generated`);

    } catch (error) {
        console.error('❌ Conversion failed:', error.message);
        process.exit(1);
    }
}

function main() {
    const config = parseArgs();

    if (config.clean) {
        console.log('🧹 Cleaning output directory...');
        // Clean output directory logic would go here
    }

    convertNotionToMarkdown(config.input, config.output, config.token);
    console.log('🎉 Notion conversion completed!');
}

// Show help if requested
if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(`
🚀 Notion to Markdown Converter

Usage:
  node notion-converter.mjs [options]

Options:
  --input=PATH     Input pages configuration file (default: input/pages.json)
  --output=PATH    Output directory (default: output/)
  --token=TOKEN    Notion API token (or set NOTION_TOKEN env var)
  --clean          Clean output directory before conversion
  --help, -h       Show this help

Environment Variables:
  NOTION_TOKEN     Your Notion integration token

Examples:
  # Basic conversion with environment token
  NOTION_TOKEN=your_token node notion-converter.mjs

  # Custom paths and token
  node notion-converter.mjs --input=my-pages.json --output=converted/ --token=your_token

  # Clean output first
  node notion-converter.mjs --clean

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
`);
    process.exit(0);
}

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
