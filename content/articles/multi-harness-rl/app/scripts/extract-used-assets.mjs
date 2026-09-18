#!/usr/bin/env node

/**
 * Extract used assets from MDX files
 * 
 * This script analyzes all MDX files in the content directory and extracts:
 * - HtmlEmbed sources (HTML visualization files)
 * - Image imports (from assets/image/)
 * 
 * It also detects:
 * - Missing assets (referenced but don't exist)
 * - Unused assets (exist but not referenced)
 * 
 * Usage: node scripts/extract-used-assets.mjs [--verbose] [--json]
 * 
 * Output is written to src/generated/used-assets.json
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(__dirname, '..');
const CONTENT_DIR = path.join(APP_ROOT, 'src', 'content');
const EMBEDS_DIR = path.join(CONTENT_DIR, 'embeds');
const IMAGES_DIR = path.join(CONTENT_DIR, 'assets', 'image');
const DATA_DIR = path.join(CONTENT_DIR, 'assets', 'data');
const OUTPUT_DIR = path.join(APP_ROOT, 'src', 'generated');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'used-assets.json');

const args = process.argv.slice(2);
const isVerbose = args.includes('--verbose') || args.includes('-v');
const isJson = args.includes('--json');

/**
 * Check if a directory exists
 */
async function dirExists(dir) {
    try {
        const stat = await fs.stat(dir);
        return stat.isDirectory();
    } catch {
        return false;
    }
}

/**
 * Recursively find all files with given extensions in a directory
 */
async function findFiles(dir, extensions, files = [], baseDir = null) {
    if (!await dirExists(dir)) return files;
    
    baseDir = baseDir || dir;
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
            // Skip node_modules and hidden directories
            if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
                await findFiles(fullPath, extensions, files, baseDir);
            }
        } else {
            const ext = path.extname(entry.name).toLowerCase();
            if (extensions.includes(ext)) {
                // Store relative path from base directory
                files.push(path.relative(baseDir, fullPath));
            }
        }
    }
    
    return files;
}

/**
 * Recursively find all MDX files in a directory
 */
async function findMdxFiles(dir, files = []) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
            // Skip node_modules and hidden directories
            if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
                await findMdxFiles(fullPath, files);
            }
        } else if (entry.name.endsWith('.mdx')) {
            files.push(fullPath);
        }
    }
    
    return files;
}

/**
 * Extract HtmlEmbed src attributes from MDX content
 */
function extractEmbedSources(content) {
    const sources = new Set();
    
    // Match <HtmlEmbed ... src="..." ...> or <HtmlEmbed ... src='...' ...>
    // Handle multiline attributes
    const regex = /<HtmlEmbed[^>]*\bsrc=["']([^"']+)["']/g;
    let match;
    
    while ((match = regex.exec(content)) !== null) {
        let src = match[1];
        // Normalize: remove leading slash and "embeds/" prefix variations
        src = src.replace(/^\//, '').replace(/^embeds\//, '');
        sources.add(src);
    }
    
    return [...sources];
}

/**
 * Extract image imports from MDX content
 */
function extractImageImports(content) {
    const images = new Set();
    
    // Match: import varName from './assets/image/filename.png';
    // Also match: import varName from "./assets/image/filename.png";
    const regex = /import\s+\w+\s+from\s+['"]\.\/assets\/image\/([^'"]+)['"]/g;
    let match;
    
    while ((match = regex.exec(content)) !== null) {
        images.add(match[1]);
    }
    
    // Also match imports from '../assets/image/' (for chapters)
    const regexRelative = /import\s+\w+\s+from\s+['"]\.\.\/assets\/image\/([^'"]+)['"]/g;
    while ((match = regexRelative.exec(content)) !== null) {
        images.add(match[1]);
    }
    
    // Match deeper relative paths like '../../assets/image/'
    const regexDeeper = /import\s+\w+\s+from\s+['"](?:\.\.\/)+assets\/image\/([^'"]+)['"]/g;
    while ((match = regexDeeper.exec(content)) !== null) {
        images.add(match[1]);
    }
    
    return [...images];
}

/**
 * Extract data file references from HtmlEmbed
 */
function extractDataFiles(content) {
    const dataFiles = new Set();
    
    // Match data="..." or data={["...", "..."]} or data='...'
    const regexSingle = /<HtmlEmbed[^>]*\bdata=["']([^"']+)["']/g;
    let match;
    
    while ((match = regexSingle.exec(content)) !== null) {
        dataFiles.add(match[1]);
    }
    
    // Match data={["file1", "file2"]} array syntax
    const regexArray = /<HtmlEmbed[^>]*\bdata=\{\[([^\]]+)\]\}/g;
    while ((match = regexArray.exec(content)) !== null) {
        const arrayContent = match[1];
        // Extract strings from array
        const stringRegex = /["']([^"']+)["']/g;
        let strMatch;
        while ((strMatch = stringRegex.exec(arrayContent)) !== null) {
            dataFiles.add(strMatch[1]);
        }
    }
    
    return [...dataFiles];
}

/**
 * Main extraction function
 */
async function extractAssets() {
    console.log('🔍 Extracting used assets from MDX files...\n');
    
    // Find all MDX files
    const mdxFiles = await findMdxFiles(CONTENT_DIR);
    
    if (isVerbose) {
        console.log(`Found ${mdxFiles.length} MDX files:\n`);
        mdxFiles.forEach(f => console.log(`  - ${path.relative(CONTENT_DIR, f)}`));
        console.log('');
    }
    
    // Scan existing assets on disk
    const existingEmbeds = new Set(await findFiles(EMBEDS_DIR, ['.html']));
    const existingImages = new Set(await findFiles(IMAGES_DIR, ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']));
    const existingData = new Set(await findFiles(DATA_DIR, ['.csv', '.json', '.tsv']));
    
    const result = {
        generatedAt: new Date().toISOString(),
        embeds: new Set(),
        images: new Set(),
        dataFiles: new Set(),
        byFile: {}
    };
    
    // Process each MDX file
    for (const file of mdxFiles) {
        const content = await fs.readFile(file, 'utf-8');
        const relativePath = path.relative(CONTENT_DIR, file);
        
        const embeds = extractEmbedSources(content);
        const images = extractImageImports(content);
        const dataFiles = extractDataFiles(content);
        
        // Add to global sets
        embeds.forEach(e => result.embeds.add(e));
        images.forEach(i => result.images.add(i));
        dataFiles.forEach(d => result.dataFiles.add(d));
        
        // Store per-file info
        if (embeds.length > 0 || images.length > 0 || dataFiles.length > 0) {
            result.byFile[relativePath] = {
                embeds: embeds.length > 0 ? embeds : undefined,
                images: images.length > 0 ? images : undefined,
                dataFiles: dataFiles.length > 0 ? dataFiles : undefined
            };
        }
    }
    
    // Calculate missing and unused assets
    const usedEmbeds = [...result.embeds];
    const usedImages = [...result.images];
    const usedData = [...result.dataFiles];
    
    const missingEmbeds = usedEmbeds.filter(e => !existingEmbeds.has(e));
    const unusedEmbeds = [...existingEmbeds].filter(e => !usedEmbeds.includes(e));
    
    const missingImages = usedImages.filter(i => !existingImages.has(i));
    const unusedImages = [...existingImages].filter(i => !usedImages.includes(i));
    
    // For data files, normalize paths (remove 'data/' prefix if present)
    const normalizeDataPath = (p) => p.replace(/^data\//, '');
    const usedDataNormalized = usedData.map(normalizeDataPath);
    const missingData = usedDataNormalized.filter(d => !existingData.has(d));
    const unusedData = [...existingData].filter(d => !usedDataNormalized.includes(d));
    
    // Convert sets to sorted arrays for output
    const output = {
        generatedAt: result.generatedAt,
        summary: {
            totalMdxFiles: mdxFiles.length,
            embeds: {
                used: result.embeds.size,
                existing: existingEmbeds.size,
                missing: missingEmbeds.length,
                unused: unusedEmbeds.length
            },
            images: {
                used: result.images.size,
                existing: existingImages.size,
                missing: missingImages.length,
                unused: unusedImages.length
            },
            dataFiles: {
                used: result.dataFiles.size,
                existing: existingData.size,
                missing: missingData.length,
                unused: unusedData.length
            }
        },
        embeds: {
            used: [...result.embeds].sort(),
            missing: missingEmbeds.sort(),
            unused: unusedEmbeds.sort()
        },
        images: {
            used: [...result.images].sort(),
            missing: missingImages.sort(),
            unused: unusedImages.sort()
        },
        dataFiles: {
            used: [...result.dataFiles].sort(),
            missing: missingData.sort(),
            unused: unusedData.sort()
        },
        byFile: result.byFile
    };
    
    // Ensure output directory exists
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    
    // Write output
    await fs.writeFile(OUTPUT_FILE, JSON.stringify(output, null, 2));
    
    // Print summary
    if (isJson) {
        console.log(JSON.stringify(output, null, 2));
    } else {
        console.log('📊 ASSET EXTRACTION SUMMARY');
        console.log('===========================\n');
        
        console.log(`📁 MDX files scanned: ${mdxFiles.length}\n`);
        
        // Embeds summary
        console.log(`🎨 HTML EMBEDS`);
        console.log(`   Used: ${result.embeds.size} | Existing: ${existingEmbeds.size}`);
        if (missingEmbeds.length > 0) {
            console.log(`   ❌ Missing: ${missingEmbeds.length}`);
        }
        if (unusedEmbeds.length > 0) {
            console.log(`   ⚠️  Unused: ${unusedEmbeds.length}`);
        }
        
        // Images summary
        console.log(`\n🖼️  IMAGES`);
        console.log(`   Used: ${result.images.size} | Existing: ${existingImages.size}`);
        if (missingImages.length > 0) {
            console.log(`   ❌ Missing: ${missingImages.length}`);
        }
        if (unusedImages.length > 0) {
            console.log(`   ⚠️  Unused: ${unusedImages.length}`);
        }
        
        // Data files summary
        console.log(`\n📄 DATA FILES`);
        console.log(`   Used: ${result.dataFiles.size} | Existing: ${existingData.size}`);
        if (missingData.length > 0) {
            console.log(`   ❌ Missing: ${missingData.length}`);
        }
        if (unusedData.length > 0) {
            console.log(`   ⚠️  Unused: ${unusedData.length}`);
        }
        
        // Verbose output
        if (isVerbose) {
            if (missingEmbeds.length > 0) {
                console.log('\n❌ MISSING EMBEDS (referenced but don\'t exist):');
                missingEmbeds.sort().forEach(e => console.log(`   - ${e}`));
            }
            
            if (unusedEmbeds.length > 0) {
                console.log('\n⚠️  UNUSED EMBEDS (exist but not referenced):');
                unusedEmbeds.sort().forEach(e => console.log(`   - ${e}`));
            }
            
            if (missingImages.length > 0) {
                console.log('\n❌ MISSING IMAGES (referenced but don\'t exist):');
                missingImages.sort().forEach(i => console.log(`   - ${i}`));
            }
            
            if (unusedImages.length > 0) {
                console.log('\n⚠️  UNUSED IMAGES (exist but not referenced):');
                unusedImages.sort().forEach(i => console.log(`   - ${i}`));
            }
            
            if (missingData.length > 0) {
                console.log('\n❌ MISSING DATA FILES (referenced but don\'t exist):');
                missingData.sort().forEach(d => console.log(`   - ${d}`));
            }
            
            if (unusedData.length > 0) {
                console.log('\n⚠️  UNUSED DATA FILES (exist but not referenced):');
                unusedData.sort().forEach(d => console.log(`   - ${d}`));
            }
        }
        
        // Health check
        const hasIssues = missingEmbeds.length > 0 || missingImages.length > 0 || missingData.length > 0;
        const hasWarnings = unusedEmbeds.length > 0 || unusedImages.length > 0 || unusedData.length > 0;
        
        console.log('\n' + '─'.repeat(40));
        if (hasIssues) {
            console.log('❌ ISSUES FOUND - Some assets are missing!');
            if (!isVerbose) {
                console.log('   Run with --verbose to see details.');
            }
        } else if (hasWarnings) {
            console.log('⚠️  WARNINGS - Some assets are unused.');
            if (!isVerbose) {
                console.log('   Run with --verbose to see details.');
            }
        } else {
            console.log('✅ All assets are healthy!');
        }
        
        console.log(`\n📄 Output written to: ${path.relative(APP_ROOT, OUTPUT_FILE)}`);
    }
    
    return output;
}

// Run
extractAssets().catch(err => {
    console.error('❌ Error:', err.message);
    process.exit(1);
});
