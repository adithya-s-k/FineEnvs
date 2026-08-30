#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { Client } from '@notionhq/client';
import { NotionConverter } from 'notion-to-md';
import { DefaultExporter } from 'notion-to-md/plugins/exporter';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Ensure directory exists
 */
function ensureDirectory(dir) {
    if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
    }
}

/**
 * Post-process Notion-generated Markdown for better MDX compatibility
 * @param {string} content - Raw markdown content from Notion
 * @param {Client} notionClient - Notion API client (optional)
 * @param {string} notionToken - Notion API token (optional)
 * @returns {Promise<string>} - Processed markdown content
 */
export async function postProcessMarkdown(content, notionClient = null, notionToken = null) {
    console.log('🔧 Post-processing Notion Markdown for MDX compatibility...');

    let processedContent = content;

    // Apply each transformation step
    processedContent = removeExcludeTags(processedContent);
    processedContent = await includeNotionPages(processedContent, notionClient, notionToken);
    processedContent = cleanNotionArtifacts(processedContent);
    processedContent = fixImageAltTextWithLinks(processedContent);
    processedContent = fixNotionLinks(processedContent);
    processedContent = fixJsxAttributes(processedContent);
    processedContent = optimizeImages(processedContent);
    processedContent = shiftHeadingLevels(processedContent);
    processedContent = cleanEmptyLines(processedContent);
    processedContent = fixCodeBlocks(processedContent);
    processedContent = fixCodeBlockEndings(processedContent);
    processedContent = unwrapHtmlCodeBlocks(processedContent);
    processedContent = fixPlainTextCodeBlocks(processedContent);
    processedContent = optimizeTables(processedContent);

    return processedContent;
}

/**
 * Remove <exclude> tags and their content, plus associated media files
 * @param {string} content - Markdown content
 * @returns {string} - Content with exclude tags removed and unused imports cleaned
 */
function removeExcludeTags(content) {
    console.log('  🗑️  Removing <exclude> tags and associated media...');

    let removedCount = 0;
    const removedImageVariables = new Set();
    const mediaFilesToDelete = new Set();

    // First, extract image variable names and media files from exclude blocks before removing them
    const excludeBlocks = content.match(/<exclude>[\s\S]*?<\/exclude>/g) || [];
    excludeBlocks.forEach(match => {
        // Extract image variables from JSX components
        const imageMatches = match.match(/src=\{([^}]+)\}/g);
        if (imageMatches) {
            imageMatches.forEach(imgMatch => {
                const varName = imgMatch.match(/src=\{([^}]+)\}/)?.[1];
                if (varName) {
                    removedImageVariables.add(varName);
                }
            });
        }

        // Extract media file paths from markdown images
        const markdownImages = match.match(/!\[[^\]]*\]\(([^)]+)\)/g);
        if (markdownImages) {
            markdownImages.forEach(imgMatch => {
                const src = imgMatch.match(/!\[[^\]]*\]\(([^)]+)\)/)?.[1];
                if (src) {
                    // Extract filename from path like /media/pageId/filename.png
                    const filename = basename(src);
                    if (filename) {
                        mediaFilesToDelete.add(filename);
                    }
                }
            });
        }
    });

    // Remove <exclude> tags and everything between them (including multiline)
    content = content.replace(/<exclude>[\s\S]*?<\/exclude>/g, (match) => {
        removedCount++;
        return '';
    });

    // Delete associated media files
    if (mediaFilesToDelete.size > 0) {
        console.log(`    🗑️  Found ${mediaFilesToDelete.size} media file(s) to delete from exclude blocks`);

        // Try to find and delete media files in common locations
        const possibleMediaDirs = [
            join(__dirname, 'output', 'media'),
            join(__dirname, '..', '..', 'src', 'content', 'assets', 'image')
        ];

        mediaFilesToDelete.forEach(filename => {
            let deleted = false;
            for (const mediaDir of possibleMediaDirs) {
                if (existsSync(mediaDir)) {
                    const filePath = join(mediaDir, filename);
                    if (existsSync(filePath)) {
                        try {
                            unlinkSync(filePath);
                            console.log(`    🗑️  Deleted media file: ${filename}`);
                            deleted = true;
                            break;
                        } catch (error) {
                            console.log(`    ⚠️  Failed to delete ${filename}: ${error.message}`);
                        }
                    }
                }
            }
            if (!deleted) {
                console.log(`    ℹ️  Media file not found: ${filename}`);
            }
        });
    }

    // Remove unused image imports that were only used in exclude blocks
    if (removedImageVariables.size > 0) {
        console.log(`    🖼️  Found ${removedImageVariables.size} unused image import(s) in exclude blocks`);

        removedImageVariables.forEach(varName => {
            // Check if the variable is still used elsewhere in the content after removing exclude blocks
            const remainingUsage = content.includes(`{${varName}}`) || content.includes(`src={${varName}}`);

            if (!remainingUsage) {
                // Remove import lines for unused image variables
                // Pattern: import VarName from './assets/image/filename';
                const importPattern = new RegExp(`import\\s+${varName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s+from\\s+['"][^'"]+['"];?\\s*`, 'g');
                content = content.replace(importPattern, '');
                console.log(`    🗑️  Removed unused import: ${varName}`);
            }
        });

        console.log(`    🧹 Cleaned up unused image imports`);
    }

    if (removedCount > 0) {
        console.log(`    ✅ Removed ${removedCount} <exclude> tag(s) and their content`);
    } else {
        console.log('    ℹ️  No <exclude> tags found');
    }

    return content;
}

/**
 * Replace Notion page links with their actual content
 * @param {string} content - Markdown content
 * @param {Client} notionClient - Notion API client
 * @param {string} notionToken - Notion API token
 * @returns {Promise<string>} - Content with page links replaced
 */
async function includeNotionPages(content, notionClient, notionToken) {
    console.log('  📄 Including linked Notion pages...');

    if (!notionClient || !notionToken) {
        console.log('    ℹ️  Skipping page inclusion (no Notion client/token provided)');
        return content;
    }

    let includedCount = 0;
    let skippedCount = 0;

    // First, identify all exclude blocks to avoid processing links within them
    const excludeBlocks = [];
    const excludeRegex = /<exclude>[\s\S]*?<\/exclude>/g;
    let excludeMatch;

    while ((excludeMatch = excludeRegex.exec(content)) !== null) {
        excludeBlocks.push({
            start: excludeMatch.index,
            end: excludeMatch.index + excludeMatch[0].length
        });
    }

    // Helper function to check if a position is within an exclude block
    const isWithinExcludeBlock = (position) => {
        return excludeBlocks.some(block => position >= block.start && position <= block.end);
    };

    // Regex to match links to Notion pages with UUID format
    // Pattern: [text](uuid-with-dashes)
    const notionPageLinkRegex = /\[([^\]]+)\]\(([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\)/g;

    let processedContent = content;
    let match;

    // Find all matches
    const matches = [];
    while ((match = notionPageLinkRegex.exec(content)) !== null) {
        const linkStartPos = match.index;

        // Skip if this link is within an exclude block
        if (isWithinExcludeBlock(linkStartPos)) {
            console.log(`    ⏭️  Skipping page link in exclude block: ${match[1]} (${match[2]})`);
            skippedCount++;
            continue;
        }

        matches.push({
            fullMatch: match[0],
            linkText: match[1],
            pageId: match[2],
            startPos: match.index,
            endPos: match.index + match[0].length
        });
    }

    // Process matches in reverse order to maintain correct indices
    for (let i = matches.length - 1; i >= 0; i--) {
        const link = matches[i];

        try {
            console.log(`    🔗 Fetching content for page: ${link.pageId}`);

            // Create media directory for this sub-page
            const outputDir = join(__dirname, 'output');
            const mediaDir = join(outputDir, 'media', link.pageId);
            ensureDirectory(mediaDir);

            // Configure the DefaultExporter to get content as string
            const exporter = new DefaultExporter({
                outputType: 'string',
            });

            // Create the converter with media downloading strategy (same as convertNotionPage)
            const converter = new NotionConverter(notionClient)
                .withExporter(exporter)
                // Download media to local directory with path transformation
                .downloadMediaTo({
                    outputDir: mediaDir,
                    // Transform paths to be web-accessible
                    transformPath: (localPath) => `/media/${link.pageId}/${basename(localPath)}`,
                });

            // Convert the page
            const result = await converter.convert(link.pageId);

            console.log(`    🖼️  Media saved to: ${mediaDir}`);

            if (result && result.content) {
                // Save raw content as .raw.md file
                const rawFileName = `${link.linkText.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${link.pageId}`;
                const rawFilePath = join(outputDir, `${rawFileName}.raw.md`);

                try {
                    writeFileSync(rawFilePath, result.content);
                    console.log(`    📄 Saved raw markdown: ${rawFileName}.raw.md`);
                } catch (error) {
                    console.log(`    ⚠️  Failed to save raw file: ${error.message}`);
                }

                // Clean the content (remove frontmatter, etc.)
                let pageContent = result.content;

                // Remove YAML frontmatter if present
                pageContent = pageContent.replace(/^---[\s\S]*?---\s*\n/, '');

                // Remove the first markdown heading (H1, H2, H3, etc.) from the included page
                pageContent = pageContent.replace(/^#+ .+\n\n?/, '');

                // Keep the page content without title
                const finalContent = '\n\n' + pageContent.trim() + '\n\n';

                // Replace the link with the content
                processedContent = processedContent.substring(0, link.startPos) +
                    finalContent +
                    processedContent.substring(link.endPos);

                includedCount++;
                console.log(`    ✅ Included page content: ${link.linkText}`);
            } else {
                console.log(`    ⚠️  No content found for page: ${link.pageId}`);
            }
        } catch (error) {
            console.log(`    ❌ Failed to fetch page ${link.pageId}: ${error.message}`);
            // Keep the original link if we can't fetch the content
        }
    }

    if (includedCount > 0) {
        console.log(`    ✅ Included ${includedCount} Notion page(s)`);
    } else {
        console.log('    ℹ️  No Notion page links found to include');
    }

    if (skippedCount > 0) {
        console.log(`    ⏭️  Skipped ${skippedCount} page link(s) in exclude blocks`);
    }

    return processedContent;
}

/**
 * Clean Notion-specific artifacts and formatting
 * @param {string} content - Markdown content
 * @returns {string} - Cleaned content
 */
function cleanNotionArtifacts(content) {
    console.log('  🧹 Cleaning Notion artifacts...');

    let cleanedCount = 0;

    // Remove Notion's internal page references that don't convert well
    content = content.replace(/\[([^\]]+)\]\(https:\/\/www\.notion\.so\/[^)]+\)/g, (match, text) => {
        cleanedCount++;
        return text; // Keep just the text, remove the broken link
    });

    // Clean up Notion's callout blocks that might not render properly
    content = content.replace(/^> \*\*([^*]+)\*\*\s*\n/gm, '> **$1**\n\n');

    // Remove Notion's page dividers that don't have markdown equivalents
    content = content.replace(/^---+\s*$/gm, '');

    // Clean up empty blockquotes
    content = content.replace(/^>\s*$/gm, '');

    // Fix corrupted bold/italic formatting from notion-to-md conversion
    // Pattern: ***text***  **** -> ***text***
    content = content.replace(/\*\*\*([^*]+)\*\*\*\s+\*\*\*\*/g, (match, text) => {
        cleanedCount++;
        return `***${text.trim()}***`;
    });

    // Fix other corrupted asterisk patterns
    // Pattern: **text**  ** -> **text**
    content = content.replace(/\*\*([^*]+)\*\*\s+\*\*/g, (match, text) => {
        cleanedCount++;
        return `**${text.trim()}**`;
    });

    if (cleanedCount > 0) {
        console.log(`    ✅ Cleaned ${cleanedCount} Notion artifact(s)`);
    }

    return content;
}

/**
 * Fix image alt text that contains markdown links
 * notion-to-md v4 sometimes generates: ![alt with [link](url)](image_path)
 * This breaks MDX parsing. Clean it to: ![alt with @mention](image_path)
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed image alt text
 */
function fixImageAltTextWithLinks(content) {
    console.log('  🖼️  Fixing image alt text with embedded links...');

    let fixedCount = 0;

    // Pattern: ![text [link](url) more_text](image_path)
    // This regex finds images where the alt text contains markdown links
    const imageWithLinksPattern = /!\[([^\]]*\[[^\]]+\]\([^)]+\)[^\]]*)\]\(([^)]+)\)/g;

    content = content.replace(imageWithLinksPattern, (match, altText, imagePath) => {
        fixedCount++;

        // Remove all markdown links from alt text: [text](url) -> text
        const cleanedAlt = altText.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

        // Also clean up any remaining brackets
        const finalAlt = cleanedAlt.replace(/[\[\]]/g, '');

        console.log(`    🔧 Fixed: "${altText.substring(0, 50)}..." -> "${finalAlt.substring(0, 50)}..."`);

        return `![${finalAlt}](${imagePath})`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} image(s) with embedded links in alt text`);
    } else {
        console.log('    ℹ️  No images with embedded links found');
    }

    return content;
}

/**
 * Fix Notion internal links to be more MDX-friendly
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed links
 */
function fixNotionLinks(content) {
    console.log('  🔗 Fixing Notion internal links...');

    let fixedCount = 0;

    // Convert Notion page links to relative links (assuming they'll be converted to MDX)
    content = content.replace(/\[([^\]]+)\]\(https:\/\/www\.notion\.so\/[^/]+\/([^?#)]+)\)/g, (match, text, pageId) => {
        fixedCount++;
        // Convert to relative link - this will need to be updated based on your routing
        return `[${text}](#${pageId})`;
    });

    // Fix broken notion.so links that might be malformed
    content = content.replace(/\[([^\]]+)\]\(https:\/\/www\.notion\.so\/[^)]*\)/g, (match, text) => {
        fixedCount++;
        return text; // Remove broken links, keep text
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} Notion link(s)`);
    }

    return content;
}

/**
 * Fix JSX attributes that were corrupted during Notion conversion
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed JSX attributes
 */
function fixJsxAttributes(content) {
    console.log('  🔧 Fixing JSX attributes corrupted by Notion conversion...');

    let fixedCount = 0;

    // Fix the specific issue: <HtmlEmbed  *src* ="/path" /> → <HtmlEmbed src="/path" />
    // Pattern: <TagName  *attribute* ="value" />
    content = content.replace(/<(\w+)\s+\*\s*([^*\s]+)\s*\*\s*=\s*"([^"]*)"\s*\/?>/g, (match, tagName, attribute, value) => {
        fixedCount++;
        return `<${tagName} ${attribute}="${value}" />`;
    });

    // Pattern: <TagName  *attribute* =value />
    content = content.replace(/<(\w+)\s+\*\s*([^*\s]+)\s*\*\s*=\s*([^>\s\/]+)\s*\/?>/g, (match, tagName, attribute, value) => {
        fixedCount++;
        return `<${tagName} ${attribute}=${value} />`;
    });

    // Handle cases with **double asterisks** around attribute names
    content = content.replace(/<(\w+)\s+\*\*\s*([^*\s]+)\s*\*\*\s*=\s*"([^"]*)"\s*\/?>/g, (match, tagName, attribute, value) => {
        fixedCount++;
        return `<${tagName} ${attribute}="${value}" />`;
    });

    content = content.replace(/<(\w+)\s+\*\*\s*([^*\s]+)\s*\*\*\s*=\s*([^>\s\/]+)\s*\/?>/g, (match, tagName, attribute, value) => {
        fixedCount++;
        return `<${tagName} ${attribute}=${value} />`;
    });

    // Fix HTML tags (like iframe, video, etc.) where URLs were corrupted by markdown conversion
    // Pattern: src="[url](url)" -> src="url"
    // Handle both regular quotes and various smart quote characters (", ", ', ', """, etc.)
    // Handle attributes before and after src

    // Handle iframe tags with separate opening and closing tags FIRST: <iframe ... src="[url](url)" ...>...</iframe>
    content = content.replace(/<iframe([^>]*?)\ssrc=[""''""\u201C\u201D\u2018\u2019]\[([^\]]+)\]\([^)]+\)[""''""\u201C\u201D\u2018\u2019]([^>]*?)>\s*<\/iframe>/gi, (match, before, urlText, after) => {
        fixedCount++;
        return `<iframe${before} src="${urlText}"${after}></iframe>`;
    });

    // Handle self-closing iframe tags SECOND: <iframe ... src="[url](url)" ... />
    content = content.replace(/<iframe([^>]*?)\ssrc=[""''""\u201C\u201D\u2018\u2019]\[([^\]]+)\]\([^)]+\)[""''""\u201C\u201D\u2018\u2019]([^>]*?)\s*\/?>/gi, (match, before, urlText, after) => {
        fixedCount++;
        return `<iframe${before} src="${urlText}"${after} />`;
    });

    // Handle other HTML tags with separate opening and closing tags FIRST: <video ... src="[url](url)" ...>...</video>
    content = content.replace(/<(video|audio|embed|object)([^>]*?)\ssrc=[""''""\u201C\u201D\u2018\u2019]\[([^\]]+)\]\([^)]+\)[""''""\u201C\u201D\u2018\u2019]([^>]*?)>\s*<\/\1>/gi, (match, tagName, before, urlText, after) => {
        fixedCount++;
        return `<${tagName}${before} src="${urlText}"${after}></${tagName}>`;
    });

    // Handle other HTML tags with the same pattern (self-closing) SECOND: <video ... src="[url](url)" ... />
    content = content.replace(/<(video|audio|embed|object)([^>]*?)\ssrc=[""''""\u201C\u201D\u2018\u2019]\[([^\]]+)\]\([^)]+\)[""''""\u201C\u201D\u2018\u2019]([^>]*?)\s*\/?>/gi, (match, tagName, before, urlText, after) => {
        fixedCount++;
        return `<${tagName}${before} src="${urlText}"${after} />`;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} corrupted JSX attribute(s)`);
    }

    return content;
}

/**
 * Optimize images for better MDX compatibility
 * @param {string} content - Markdown content
 * @returns {string} - Content with optimized images
 */
function optimizeImages(content) {
    console.log('  🖼️  Optimizing images...');

    let optimizedCount = 0;

    // Ensure images have proper alt text
    content = content.replace(/!\[\]\(([^)]+)\)/g, (match, src) => {
        optimizedCount++;
        const filename = basename(src);
        return `![${filename}](${src})`;
    });

    // Clean up image paths that might have query parameters
    content = content.replace(/!\[([^\]]*)\]\(([^)]+)\?[^)]*\)/g, (match, alt, src) => {
        optimizedCount++;
        return `![${alt}](${src})`;
    });

    if (optimizedCount > 0) {
        console.log(`    ✅ Optimized ${optimizedCount} image(s)`);
    }

    return content;
}

/**
 * Shift all heading levels down by one (H1 → H2, H2 → H3, etc.)
 * @param {string} content - Markdown content
 * @returns {string} - Content with shifted heading levels
 */
function shiftHeadingLevels(content) {
    console.log('  📝 Shifting heading levels down by one...');

    let shiftedCount = 0;

    // Shift heading levels: H1 → H2, H2 → H3, H3 → H4, H4 → H5, H5 → H6
    // Process from highest to lowest to avoid conflicts
    content = content.replace(/^##### (.*$)/gim, '###### $1');
    content = content.replace(/^#### (.*$)/gim, '##### $1');
    content = content.replace(/^### (.*$)/gim, '#### $1');
    content = content.replace(/^## (.*$)/gim, '### $1');
    content = content.replace(/^# (.*$)/gim, '## $1');

    // Count the number of headings shifted
    const headingMatches = content.match(/^#{1,6} /gm);
    if (headingMatches) {
        shiftedCount = headingMatches.length;
    }

    console.log(`    ✅ Shifted ${shiftedCount} heading level(s)`);
    return content;
}

/**
 * Fix code block endings that end with "text" instead of proper closing
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed code block endings
 */
function fixCodeBlockEndings(content) {
    console.log('  💻 Fixing code block endings...');

    let fixedCount = 0;

    // Fix code blocks that end with ```text instead of ```
    content = content.replace(/```text\n/g, '```\n');

    // Count the number of fixes
    const textEndingMatches = content.match(/```text\n/g);
    if (textEndingMatches) {
        fixedCount = textEndingMatches.length;
    }

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} code block ending(s)`);
    }

    return content;
}

/**
 * Clean up excessive empty lines
 * @param {string} content - Markdown content
 * @returns {string} - Content with cleaned spacing
 */
function cleanEmptyLines(content) {
    console.log('  📝 Cleaning excessive empty lines...');

    // Only replace 4+ consecutive newlines with 2 newlines (be more conservative)
    // This preserves single empty lines between paragraphs which are important for readability
    const cleanedContent = content.replace(/\n{4,}/g, '\n\n');

    const originalLines = content.split('\n').length;
    const cleanedLines = cleanedContent.split('\n').length;
    const removedLines = originalLines - cleanedLines;

    if (removedLines > 0) {
        console.log(`    ✅ Removed ${removedLines} excessive empty line(s)`);
    }

    return cleanedContent;
}

/**
 * Fix code blocks for better MDX compatibility
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed code blocks
 */
function fixCodeBlocks(content) {
    console.log('  💻 Fixing code blocks...');

    let fixedCount = 0;

    // Ensure code blocks have proper language identifiers
    content = content.replace(/^```\s*$/gm, '```text');

    // Fix code blocks that might have Notion-specific formatting
    content = content.replace(/^```(\w+)\s*\n([\s\S]*?)\n```$/gm, (match, lang, code) => {
        // Clean up any Notion artifacts in code
        const cleanCode = code.replace(/\u00A0/g, ' '); // Replace non-breaking spaces
        return `\`\`\`${lang}\n${cleanCode}\n\`\`\``;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} code block(s)`);
    }

    return content;
}

/**
 * Optimize tables for better MDX rendering
 * @param {string} content - Markdown content
 * @returns {string} - Content with optimized tables
 */
function optimizeTables(content) {
    console.log('  📊 Optimizing tables...');

    let optimizedCount = 0;

    // Fix tables that might have inconsistent column counts
    content = content.replace(/^\|(.+)\|\s*$/gm, (match, row) => {
        const cells = row.split('|').map(cell => cell.trim());
        const cleanCells = cells.filter(cell => cell.length > 0);

        if (cleanCells.length > 0) {
            optimizedCount++;
            return `| ${cleanCells.join(' | ')} |`;
        }
        return match;
    });

    // Ensure table headers are properly formatted
    content = content.replace(/^\|(.+)\|\s*\n\|([-:\s|]+)\|\s*$/gm, (match, header, separator) => {
        const headerCells = header.split('|').map(cell => cell.trim()).filter(cell => cell.length > 0);
        const separatorCells = separator.split('|').map(cell => cell.trim()).filter(cell => cell.length > 0);

        if (headerCells.length !== separatorCells.length) {
            optimizedCount++;
            const newSeparator = headerCells.map(() => '---').join(' | ');
            return `| ${headerCells.join(' | ')} |\n| ${newSeparator} |`;
        }
        return match;
    });

    if (optimizedCount > 0) {
        console.log(`    ✅ Optimized ${optimizedCount} table(s)`);
    }

    return content;
}

/**
 * Unwrap HTML code blocks to allow direct HTML integration in MDX
 * @param {string} content - Markdown content
 * @returns {string} - Content with unwrapped HTML code blocks
 */
function unwrapHtmlCodeBlocks(content) {
    console.log('  🔧 Unwrapping HTML code blocks for MDX integration...');

    let unwrappedCount = 0;

    // Pattern to match ```html ... ``` blocks
    // This regex captures the entire code block including the ```html and ``` markers
    const htmlCodeBlockRegex = /```html\s*\n([\s\S]*?)\n```/g;

    content = content.replace(htmlCodeBlockRegex, (match, htmlContent) => {
        unwrappedCount++;

        // Clean up the HTML content - remove leading/trailing whitespace
        const cleanHtmlContent = htmlContent.trim();

        console.log(`    🔧 Unwrapped HTML code block (${cleanHtmlContent.length} chars)`);

        // Return the HTML content without the code block wrapper
        return cleanHtmlContent;
    });

    if (unwrappedCount > 0) {
        console.log(`    ✅ Unwrapped ${unwrappedCount} HTML code block(s) for MDX integration`);
    } else {
        console.log('    ℹ️  No HTML code blocks found to unwrap');
    }

    return content;
}

/**
 * Fix plain text code blocks by removing the "plain text" language identifier
 * @param {string} content - Markdown content
 * @returns {string} - Content with fixed plain text code blocks
 */
function fixPlainTextCodeBlocks(content) {
    console.log('  🔧 Fixing plain text code blocks...');

    let fixedCount = 0;

    // Pattern to match ```plain text ... ``` blocks and convert them to ``` ... ```
    const plainTextCodeBlockRegex = /```plain text\s*\n([\s\S]*?)\n```/g;

    content = content.replace(plainTextCodeBlockRegex, (match, codeContent) => {
        fixedCount++;

        console.log(`    🔧 Fixed plain text code block (${codeContent.length} chars)`);

        // Return the code block without the "plain text" language identifier
        return `\`\`\`\n${codeContent}\n\`\`\``;
    });

    if (fixedCount > 0) {
        console.log(`    ✅ Fixed ${fixedCount} plain text code block(s)`);
    } else {
        console.log('    ℹ️  No plain text code blocks found to fix');
    }

    return content;
}

/**
 * Extract frontmatter from Notion page properties
 * @param {Object} pageProperties - Notion page properties
 * @returns {string} - YAML frontmatter
 */
export function generateFrontmatter(pageProperties) {
    console.log('  📄 Generating frontmatter from Notion properties...');

    const frontmatter = {
        title: pageProperties.title || 'Untitled',
        published: new Date().toISOString().split('T')[0],
        tableOfContentsAutoCollapse: true
    };

    // Add other properties if they exist
    if (pageProperties.description) {
        frontmatter.description = pageProperties.description;
    }
    if (pageProperties.tags) {
        frontmatter.tags = pageProperties.tags;
    }
    if (pageProperties.author) {
        frontmatter.author = pageProperties.author;
    }

    // Convert to YAML string
    const yamlLines = Object.entries(frontmatter)
        .map(([key, value]) => {
            if (Array.isArray(value)) {
                return `${key}:\n${value.map(v => `  - ${v}`).join('\n')}`;
            }
            return `${key}: "${value}"`;
        });

    return `---\n${yamlLines.join('\n')}\n---\n\n`;
}

function main() {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
🔧 Notion Markdown Post-Processor

Usage:
  node post-processor.mjs [options] [input-file] [output-file]

Options:
  --verbose        Show detailed processing information
  --help, -h       Show this help

Examples:
  # Process a single file
  node post-processor.mjs input.md output.md

  # Process with verbose output
  node post-processor.mjs --verbose input.md output.md
`);
        process.exit(0);
    }

    const verbose = args.includes('--verbose');
    const inputFile = args.find(arg => !arg.startsWith('--') && arg.endsWith('.md'));
    const outputFile = args.find(arg => !arg.startsWith('--') && arg !== inputFile && arg.endsWith('.md'));

    if (!inputFile) {
        console.error('❌ Please provide an input markdown file');
        process.exit(1);
    }

    if (!existsSync(inputFile)) {
        console.error(`❌ Input file not found: ${inputFile}`);
        process.exit(1);
    }

    try {
        console.log(`📖 Reading: ${inputFile}`);
        const content = readFileSync(inputFile, 'utf8');

        const processedContent = postProcessMarkdown(content);

        const finalOutputFile = outputFile || inputFile.replace('.md', '.processed.md');
        writeFileSync(finalOutputFile, processedContent);

        console.log(`✅ Processed: ${finalOutputFile}`);

        if (verbose) {
            console.log(`📊 Input: ${content.length} chars → Output: ${processedContent.length} chars`);
        }

    } catch (error) {
        console.error('❌ Processing failed:', error.message);
        process.exit(1);
    }
}

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
