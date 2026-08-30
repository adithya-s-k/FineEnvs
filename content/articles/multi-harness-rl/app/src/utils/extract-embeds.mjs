import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

/**
 * Extract HtmlEmbed, Image components and tables from MDX/Markdown content
 * Simple utility to find visual elements and their props
 */

/**
 * Parse image import statements from MDX content.
 * Returns a Map of variable name → filename (just the basename).
 * e.g. import placeholder from '../../assets/image/placeholder.png'
 *      → Map { 'placeholder' => 'placeholder.png' }
 */
function parseImageImports(content) {
    const importMap = new Map();
    const importPattern = /import\s+(\w+)\s+from\s+["']([^"']+)["']/g;
    let match;
    while ((match = importPattern.exec(content)) !== null) {
        const varName = match[1];
        const importPath = match[2];
        // Extract just the filename from the path
        const filename = importPath.split('/').pop();
        if (filename && /\.(png|jpe?g|gif|webp|svg)$/i.test(filename)) {
            importMap.set(varName, filename);
        }
    }
    return importMap;
}

/**
 * Strip fenced code blocks (``` ... ```) from content.
 * Replaces code block content with whitespace of equal length
 * to preserve character positions for downstream extraction.
 */
function stripCodeBlocks(content) {
    return content.replace(/```[\s\S]*?```/g, (match) => ' '.repeat(match.length));
}

/**
 * Simple Markdown to HTML converter for table cells
 * Handles: links, bold, italic, code, strikethrough
 */
function markdownToHtml(md) {
    if (!md) return '';
    
    let html = md;
    
    // Escape HTML entities first (but not for already-converted content)
    // Skip if it already looks like HTML
    if (!html.includes('<a ') && !html.includes('<strong>')) {
        html = html
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
    
    // Links: [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    // Bold: **text** or __text__
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    
    // Italic: *text* or _text_ (but not inside words)
    html = html.replace(/(?<![*_])\*([^*]+)\*(?![*_])/g, '<em>$1</em>');
    html = html.replace(/(?<![*_])_([^_]+)_(?![*_])/g, '<em>$1</em>');
    
    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Strikethrough: ~~text~~
    html = html.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    
    // Checkboxes (common in tables)
    html = html.replace(/\[x\]/gi, '✅');
    html = html.replace(/\[ \]/g, '❌');
    
    return html;
}

/**
 * Extract Image components from MDX content
 */
export function extractImages(content) {
    const images = [];
    
    // Match <Image ... /> components
    const imagePattern = /<Image[^>]*\/>/gi;
    let match;
    
    while ((match = imagePattern.exec(content)) !== null) {
        const tag = match[0];
        
        // Extract src attribute (variable reference like {myImage})
        const srcMatch = tag.match(/src\s*=\s*\{([^}]+)\}/i);
        const src = srcMatch ? srcMatch[1].trim() : null;
        
        // Extract alt
        const altMatch = tag.match(/alt\s*=\s*["']([^"']+)["']/i);
        const alt = altMatch ? altMatch[1] : 'Image';
        
        // Extract caption
        const captionMatch = tag.match(/caption\s*=\s*["']([^"']+)["']/i) || 
                            tag.match(/caption\s*=\s*\{`([^`]+)`\}/i);
        const caption = captionMatch ? captionMatch[1] : null;
        
        // Extract id
        const idMatch = tag.match(/id\s*=\s*["']([^"']+)["']/i);
        const id = idMatch ? idMatch[1] : null;

        // Extract skipGallery
        const skipGallery = /\bskipGallery\b/i.test(tag);
        
        if (src) {
            images.push({
                type: 'image',
                src,
                alt,
                caption,
                id,
                skipGallery
            });
        }
    }
    
    return images;
}

/**
 * Split a markdown table row on pipe characters, respecting backtick spans.
 * Pipes inside `inline code` are treated as literal text, not separators.
 * Strips the leading/trailing empty strings from outer pipes but keeps
 * internal empty cells (e.g. | val | | → ['val', '']).
 */
function splitTableRow(row) {
    const raw = [];
    let current = '';
    let inBacktick = false;
    
    for (let i = 0; i < row.length; i++) {
        const ch = row[i];
        if (ch === '`') {
            inBacktick = !inBacktick;
            current += ch;
        } else if (ch === '|' && !inBacktick) {
            raw.push(current);
            current = '';
        } else {
            current += ch;
        }
    }
    raw.push(current);
    
    // Remove leading/trailing empty strings from outer | delimiters
    // but preserve internal empty cells
    let start = 0;
    let end = raw.length - 1;
    while (start <= end && !raw[start].trim()) start++;
    while (end >= start && !raw[end].trim()) end--;
    
    return raw.slice(start, end + 1).map(c => c.trim());
}

/**
 * Parse a markdown table row into exactly `expectedCols` cells.
 * First splits respecting backticks (splitTableRow), then if the row
 * has too many cells (e.g. unescaped | in values), merges overflow
 * cells back together to match the expected column count.
 */
function parseTableRow(row, expectedCols) {
    let cells = splitTableRow(row);
    
    if (cells.length <= expectedCols) return cells;
    
    // Too many cells: merge overflow into the last "overflowing" column.
    // Strategy: keep first (expectedCols - 1) cells, merge the rest into one,
    // then take the last cell separately (it's usually the description).
    // This handles: | val | No | 'a' | 'b' | 'c' | Description |
    //  → [val, No, 'a' | 'b' | 'c', Description]
    const head = cells.slice(0, expectedCols - 1);
    const tail = cells.slice(expectedCols - 1);
    // The last element is the final column; everything in between is the overflowing column
    if (tail.length > 1) {
        const lastCell = tail.pop();
        const merged = tail.join(' | ');
        return [...head, merged, lastCell].slice(0, expectedCols);
    }
    
    return cells.slice(0, expectedCols);
}

/**
 * Extract markdown tables from content
 */
export function extractTables(content) {
    const tables = [];
    
    // Match markdown tables (lines starting with |)
    // A table has at least a header row, separator row, and one data row
    const tablePattern = /(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)/g;
    let match;
    let tableIndex = 0;
    
    while ((match = tablePattern.exec(content)) !== null) {
        const tableContent = match[1].trim();
        const rows = tableContent.split('\n').filter(row => row.trim());
        
        if (rows.length >= 3) {
            // Parse header (filter empty = no empty header columns)
            const headerRow = rows[0];
            const headers = splitTableRow(headerRow)
                .filter(cell => cell !== '')
                .map(cell => markdownToHtml(cell));
            
            const expectedCols = headers.length;
            
            // Parse data rows (skip separator at index 1)
            // Use parseTableRow to handle overflow pipes (e.g. union types)
            // Keep empty cells to preserve column alignment
            const dataRows = rows.slice(2).map(row => {
                return parseTableRow(row, expectedCols)
                    .map(cell => markdownToHtml(cell));
            });
            
            tables.push({
                type: 'table',
                id: `table-${tableIndex++}`,
                headers,
                rows: dataRows,
                raw: tableContent
            });
        }
    }
    
    return tables;
}

export function extractHtmlEmbeds(rawContent) {
    const embeds = [];

    // Strip code blocks to avoid extracting components from code examples
    const content = stripCodeBlocks(rawContent);

    // First, find all Wide components and mark their content
    // Pattern to match <Wide>...</Wide> blocks
    const widePattern = /<Wide[\s\S]*?>([\s\S]*?)<\/Wide>/gi;
    const wideBlocks = [];
    let wideMatch;
    while ((wideMatch = widePattern.exec(content)) !== null) {
        wideBlocks.push({
            start: wideMatch.index,
            end: wideMatch.index + wideMatch[0].length,
            content: wideMatch[0]
        });
    }

    // Helper to check if an embed is inside a Wide block
    const isInsideWide = (embedStartIndex) => {
        return wideBlocks.some(block =>
            embedStartIndex >= block.start && embedStartIndex < block.end
        );
    };

    // Pattern to match HtmlEmbed opening tags
    const embedPattern = /<HtmlEmbed/gi;
    let embedMatch;

    while ((embedMatch = embedPattern.exec(content)) !== null) {
        const matchIndex = embedMatch.index;

        // Manually find the closing /> while respecting string boundaries
        let pos = matchIndex + 10; // After "<HtmlEmbed"
        let match = '<HtmlEmbed';
        let inString = false;
        let stringDelim = null;
        let inJSXBraces = 0; // Track depth in JSX expressions like config={{...}}

        while (pos < content.length) {
            const char = content[pos];
            const prevChar = pos > 0 ? content[pos - 1] : '';

            match += char;

            // Track string boundaries (template strings, single, double quotes)
            if (!inString) {
                if ((char === '`' || char === '"' || char === "'") && prevChar !== '\\') {
                    inString = true;
                    stringDelim = char;
                }
            } else {
                if (char === stringDelim && prevChar !== '\\') {
                    inString = false;
                    stringDelim = null;
                }
            }

            // Track JSX expression braces (for config={{...}}, data={{...}}, etc.)
            if (!inString) {
                if (char === '{') {
                    inJSXBraces++;
                } else if (char === '}') {
                    inJSXBraces--;
                }
            }

            // Check for closing /> - only valid if not in string AND all JSX braces are closed
            if (!inString && inJSXBraces === 0 && char === '/' && pos + 1 < content.length && content[pos + 1] === '>') {
                match += '>';
                break;
            }

            pos++;
        }

        // If config={{ is present, we need to find the real closing after }}
        // Check if config={{ is there but the match doesn't include the full config (doesn't end with }})
        if (match.includes('config={{') && !match.includes('}}')) {
            // The match was cut off at the first /> it found (probably in desc)
            // We need to find the real closing after }}

            // Find where config={{ starts
            const configStart = match.indexOf('config={{');
            if (configStart >= 0) {
                // Look for the matching }} after this, starting from the content
                let braceCount = 2; // We're inside {{
                let pos = matchIndex + configStart + 9; // After "config={{"
                let foundEnd = false;

                while (pos < content.length) {
                    const char = content[pos];
                    const prevChar = pos > 0 ? content[pos - 1] : '';

                    // Track strings to avoid counting braces inside strings
                    if ((char === '"' || char === "'" || char === '`') && prevChar !== '\\') {
                        // We hit a string delimiter - skip the entire string
                        const stringDelim = char;
                        pos++;
                        while (pos < content.length) {
                            if (content[pos] === stringDelim && content[pos - 1] !== '\\') {
                                break;
                            }
                            // Handle template string ${...}
                            if (stringDelim === '`' && content[pos] === '$' && pos + 1 < content.length && content[pos + 1] === '{') {
                                // Skip ${...} without counting
                                pos += 2;
                                let innerBraces = 1;
                                while (pos < content.length && innerBraces > 0) {
                                    if (content[pos] === '{') innerBraces++;
                                    if (content[pos] === '}') innerBraces--;
                                    pos++;
                                }
                                continue;
                            }
                            pos++;
                        }
                        pos++; // Skip the closing quote
                        continue;
                    }

                    if (char === '{') braceCount++;
                    if (char === '}') {
                        braceCount--;
                        if (braceCount === 0) {
                            // Found the closing }}
                            // Now look for /> after optional whitespace
                            pos++;
                            while (pos < content.length && /\s/.test(content[pos])) {
                                pos++;
                            }
                            if (pos < content.length && content[pos] === '/' && pos + 1 < content.length && content[pos + 1] === '>') {
                                // Found the real closing
                                match = content.substring(matchIndex, pos + 2);
                                foundEnd = true;
                                break;
                            }
                        }
                    }
                    pos++;
                }

                if (!foundEnd) {
                    // Fallback: try to find }} /> pattern after match
                    const after = content.substring(matchIndex + match.length);
                    const endPattern = after.match(/\}\}\s*\/>/);
                    if (endPattern) {
                        match = content.substring(matchIndex, matchIndex + match.length + endPattern.index + endPattern[0].length);
                    }
                }
            }
        }

        // Helper function to extract attribute value supporting multiline
        const extractAttr = (attrName, content) => {
            // Try JSX template strings first: desc={`...`}
            const templateMatch = content.match(new RegExp(`${attrName}\\s*=\\s*\\{\`([\\s\\S]*?)\`\\}`, 'i'));
            if (templateMatch) return templateMatch[1].trim();

            // Try single quotes: desc='...'
            const singleQuoteMatch = content.match(new RegExp(`${attrName}\\s*=\\s*'([\\s\\S]*?)'`, 'i'));
            if (singleQuoteMatch) return singleQuoteMatch[1].trim();

            // Try double quotes: desc="..."
            const doubleQuoteMatch = content.match(new RegExp(`${attrName}\\s*=\\s*"([\\s\\S]*?)"`, 'i'));
            if (doubleQuoteMatch) return doubleQuoteMatch[1].trim();

            return undefined;
        };

        // Extract src attribute (required)
        const src = extractAttr('src', match);
        if (!src) continue;

        // Extract optional attributes
        const title = extractAttr('title', match);
        const desc = extractAttr('desc', match);
        const id = extractAttr('id', match);
        const data = extractAttr('data', match);
        const frameless = /\bframeless\b/i.test(match);
        const wideAttr = /\bwide\b/i.test(match);
        const skipGallery = /\bskipGallery\b/i.test(match);

        // Extract config attribute - JSX object format: config={{ ... }}
        let config = null;

        // Pattern to match config={{ ... }} with balanced braces
        const jsxConfigRegex = /config\s*=\s*\{\{/i;
        const jsxConfigMatch = match.match(jsxConfigRegex);

        if (jsxConfigMatch) {
            try {
                // Find the start position after config={{ 
                const configStart = jsxConfigMatch.index;
                const startPos = match.indexOf('{{', configStart) + 2;

                // Find matching closing braces with better handling
                let braceCount = 1; // Start at 1 because we're inside {{ 
                let inString = false;
                let stringChar = null;
                let pos = startPos;

                for (; pos < match.length; pos++) {
                    const char = match[pos];
                    const prevChar = pos > 0 ? match[pos - 1] : '';
                    const nextChar = pos < match.length - 1 ? match[pos + 1] : '';

                    // Handle string literals - check for template strings too
                    if (!inString) {
                        if (char === '`') {
                            inString = true;
                            stringChar = '`';
                        } else if (char === '"' && prevChar !== '\\') {
                            inString = true;
                            stringChar = '"';
                        } else if (char === "'" && prevChar !== '\\') {
                            inString = true;
                            stringChar = "'";
                        }
                    } else {
                        // Check for end of string
                        if (char === stringChar && prevChar !== '\\') {
                            inString = false;
                            stringChar = null;
                        }
                        // Template strings can contain ${...} - handle that
                        if (stringChar === '`' && char === '$' && nextChar === '{') {
                            // Skip the ${ but don't count it as a brace yet
                            pos++; // Skip $
                            braceCount++; // Count the { we're about to see
                            continue;
                        }
                    }

                    if (!inString) {
                        if (char === '{') {
                            braceCount++;
                        } else if (char === '}') {
                            braceCount--;
                            if (braceCount === 0) {
                                // Found matching closing }}
                                break;
                            }
                        }
                    }
                }

                if (braceCount !== 0) {
                    throw new Error(`Unbalanced braces: braceCount=${braceCount}`);
                }

                // Extract the JSX object content
                let jsxContent = match.substring(startPos, pos).trim();

                // Instead of converting to JSON, evaluate the JavaScript object directly
                // This is safer in a build context (not in browser)
                try {
                    // Wrap in parentheses and braces to make it a valid expression
                    const jsCode = `({${jsxContent}})`;

                    // Use Function constructor to safely evaluate (no access to local scope)
                    // This is safe because we're in Node.js build time, not browser runtime
                    config = new Function('return ' + jsCode)();
                } catch (evalError) {
                    // If eval fails, try the JSON approach as fallback
                    // Fallback: try JSON parsing
                    let jsonStr = jsxContent;

                    // Add braces around the content
                    jsonStr = '{' + jsonStr + '}';

                    // Quote unquoted keys
                    for (let pass = 0; pass < 5; pass++) {
                        jsonStr = jsonStr.replace(/([{,\[\s])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:/g, '$1"$2":');
                        jsonStr = jsonStr.replace(/^([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:/gm, '"$1":');
                    }

                    // Replace single quotes with double quotes
                    jsonStr = jsonStr.replace(/'/g, '"');

                    // Remove trailing commas
                    jsonStr = jsonStr.replace(/,\s*([}\]])/g, '$1');

                    try {
                        config = JSON.parse(jsonStr);
                    } catch (jsonError) {
                        // Both methods failed - log warning but don't throw
                        console.warn('[extract-embeds] Config parsing failed:', jsonError.message);
                    }
                }
            } catch (e) {
                // If parsing fails, keep config as null
                // Component will handle missing config
            }
        }

        // Fallback: try standard attribute extraction (for string-based config)
        if (!config) {
            const configAttr = extractAttr('config', match);
            if (configAttr) {
                try {
                    config = JSON.parse(configAttr);
                } catch (e) {
                    // Keep as string if not valid JSON
                    config = configAttr;
                }
            }
        }

        // Check if this embed is inside a Wide component OR has wide prop
        const isWide = isInsideWide(matchIndex) || wideAttr;

        embeds.push({
            src,
            title,
            desc,
            id,
            frameless,
            data,
            config,
            wide: isWide,
            skipGallery
        });
    }

    return embeds;
}

/**
 * Recursively find all MDX files in a directory
 * Skips demo chapters by default to avoid missing embeds
 */
function findMdxFiles(dir, baseDir = dir, files = [], skipDemo = true) {
    const entries = readdirSync(dir);

    for (const entry of entries) {
        const fullPath = join(dir, entry);
        const stat = statSync(fullPath);

        if (stat.isDirectory()) {
            // Skip demo directory if skipDemo is true
            if (skipDemo && entry === 'demo') {
                continue;
            }
            findMdxFiles(fullPath, baseDir, files, skipDemo);
        } else if (entry.endsWith('.mdx')) {
            files.push(fullPath);
        }
    }

    return files;
}

/**
 * Parse imports and chapter usage order from article.mdx
 */
function parseArticleChapters(articleContent, contentDir) {
    const chapterMap = new Map(); // Component name -> file path
    const chapterOrder = []; // Ordered list of file paths

    // Extract import statements
    const importPattern = /import\s+(\w+)\s+from\s+["'](.\/chapters\/[^"']+)["']/g;
    let match;
    while ((match = importPattern.exec(articleContent)) !== null) {
        const [, componentName, importPath] = match;
        const fullPath = join(contentDir, importPath);
        chapterMap.set(componentName, fullPath);
    }

    // Extract chapter usage order (e.g., <Introduction />)
    const usagePattern = /<(\w+)\s*\/>/g;
    while ((match = usagePattern.exec(articleContent)) !== null) {
        const componentName = match[1];
        if (chapterMap.has(componentName)) {
            const chapterPath = chapterMap.get(componentName);
            if (!chapterOrder.includes(chapterPath)) {
                chapterOrder.push(chapterPath);
            }
        }
    }

    return chapterOrder;
}

/**
 * Build a unique identity key for an embed.
 *
 * Strategy (in priority order):
 *   1. `id` — if the author gave an explicit id, it's unique by convention.
 *   2. `src` + deterministic hash of (config, data) — same template with
 *      different parameters produces different keys.
 *   3. `src` alone — for embeds with no config/data (unique HTML file).
 *
 * This allows the same generic template (e.g. d3-line-chart.html) to appear
 * multiple times when each instance carries a different config, while still
 * deduplicating true duplicates (same src + same config that appear in both
 * article.mdx and a chapter).
 */
function embedKey(embed) {
    if (embed.id) return `id:${embed.id}`;

    const hasConfig = embed.config != null;
    const hasData = embed.data != null;

    if (!hasConfig && !hasData) return `src:${embed.src}`;

    // Deterministic hash of the variable parts
    const payload = JSON.stringify({ config: embed.config ?? null, data: embed.data ?? null });
    const hash = createHash('sha1').update(payload).digest('hex').slice(0, 10);
    return `src:${embed.src}#${hash}`;
}

/**
 * Load and extract embeds from MDX content files, following article structure
 */
export function loadEmbedsFromMDX() {
    // Get absolute path to content directory
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);

    // Try to resolve content directory - works in both dev and build
    let contentDir = join(__dirname, '../content');

    if (!statSync(contentDir, { throwIfNoEntry: false })) {
        contentDir = join(__dirname, '../../src/content');
    }

    if (!statSync(contentDir, { throwIfNoEntry: false })) {
        contentDir = join(__dirname, '../../../src/content');
    }

    const allEmbeds = [];
    const articleFile = join(contentDir, 'article.mdx');

    try {
        // Read main article file
        const articleContent = readFileSync(articleFile, 'utf-8');

        // Extract embeds from main article first
        const articleEmbeds = extractHtmlEmbeds(articleContent);
        articleEmbeds.forEach(embed => {
            embed.sourceFile = 'content/article.mdx';
        });
        allEmbeds.push(...articleEmbeds);

        // Parse chapter order from article
        const chapterOrder = parseArticleChapters(articleContent, contentDir);

        // Extract embeds from chapters in order
        for (const chapterPath of chapterOrder) {
            try {
                const chapterContent = readFileSync(chapterPath, 'utf-8');
                const embeds = extractHtmlEmbeds(chapterContent);

                // Add source file info (relative path)
                const relativePath = relative(contentDir, chapterPath);
                embeds.forEach(embed => {
                    embed.sourceFile = `content/${relativePath}`;
                });

                allEmbeds.push(...embeds);
            } catch (error) {
                console.error(`Error reading chapter ${chapterPath}:`, error);
            }
        }

        // Also include any other MDX files not in chapters (for completeness)
        const allMdxFiles = findMdxFiles(contentDir, contentDir, [], false);
        const processedFiles = new Set([articleFile, ...chapterOrder]);

        for (const filePath of allMdxFiles) {
            if (!processedFiles.has(filePath)) {
                try {
                    const rawContent = readFileSync(filePath, 'utf-8');
                    const embeds = extractHtmlEmbeds(rawContent);
                    const relativePath = relative(contentDir, filePath);
                    embeds.forEach(embed => {
                        embed.sourceFile = `content/${relativePath}`;
                    });
                    allEmbeds.push(...embeds);
                } catch (error) {
                    console.error(`Error reading ${filePath}:`, error);
                }
            }
        }
    } catch (error) {
        console.error('Error processing article:', error);
        // Fallback to old behavior if article.mdx can't be read
        const mdxFiles = findMdxFiles(contentDir, contentDir, [], false);
        for (const filePath of mdxFiles) {
            try {
                const rawContent = readFileSync(filePath, 'utf-8');
                const embeds = extractHtmlEmbeds(rawContent);
                const relativePath = relative(contentDir, filePath);
                embeds.forEach(embed => {
                    embed.sourceFile = `content/${relativePath}`;
                });
                allEmbeds.push(...embeds);
            } catch (err) {
                console.error(`Error reading ${filePath}:`, err);
            }
        }
    }

    // Remove true duplicates (same identity) keeping first occurrence (= order of appearance).
    // Identity = id (if set), or src + hash(config, data). This means the same generic
    // template with different configs produces distinct entries.
    const seen = new Map();
    const uniqueEmbeds = [];
    for (const embed of allEmbeds) {
        const key = embedKey(embed);
        if (!seen.has(key)) {
            seen.set(key, true);
            uniqueEmbeds.push(embed);
        }
    }

    return uniqueEmbeds;
}

/**
 * Helper to extract attribute from tag content
 */
function extractAttrFromTag(attrName, tagContent) {
    // Try JSX template strings first: attr={`...`}
    const templateMatch = tagContent.match(new RegExp(`${attrName}\\s*=\\s*\\{\`([\\s\\S]*?)\`\\}`, 'i'));
    if (templateMatch) return templateMatch[1].trim();

    // Try single quotes: attr='...'
    const singleQuoteMatch = tagContent.match(new RegExp(`${attrName}\\s*=\\s*'([\\s\\S]*?)'`, 'i'));
    if (singleQuoteMatch) return singleQuoteMatch[1].trim();

    // Try double quotes: attr="..."
    const doubleQuoteMatch = tagContent.match(new RegExp(`${attrName}\\s*=\\s*"([\\s\\S]*?)"`, 'i'));
    if (doubleQuoteMatch) return doubleQuoteMatch[1].trim();

    return undefined;
}

/**
 * Check if position is inside a Wide component
 */
function isPositionInsideWide(content, position) {
    const widePattern = /<Wide[\s\S]*?>([\s\S]*?)<\/Wide>/gi;
    let match;
    while ((match = widePattern.exec(content)) !== null) {
        if (position >= match.index && position < match.index + match[0].length) {
            return true;
        }
    }
    return false;
}

/**
 * Extract all visual elements from content with their position
 * Returns sorted by position (order of appearance)
 */
function extractAllVisualsWithPosition(rawContent) {
    const visuals = [];
    
    // Parse image imports before stripping code blocks (imports are never in code blocks)
    const imageImports = parseImageImports(rawContent);
    
    // Strip code blocks to avoid extracting components from code examples
    const content = stripCodeBlocks(rawContent);
    
    // Extract HtmlEmbeds with position and ALL props
    const embedPattern = /<HtmlEmbed/gi;
    let match;
    while ((match = embedPattern.exec(content)) !== null) {
        const position = match.index;
        // Find the end of this tag
        let pos = position + 10;
        let tagContent = '<HtmlEmbed';
        let inString = false;
        let stringDelim = null;
        let inJSXBraces = 0;
        
        while (pos < content.length) {
            const char = content[pos];
            const prevChar = pos > 0 ? content[pos - 1] : '';
            tagContent += char;
            
            if (!inString) {
                if ((char === '`' || char === '"' || char === "'") && prevChar !== '\\') {
                    inString = true;
                    stringDelim = char;
                }
            } else {
                if (char === stringDelim && prevChar !== '\\') {
                    inString = false;
                    stringDelim = null;
                }
            }
            
            if (!inString) {
                if (char === '{') inJSXBraces++;
                else if (char === '}') inJSXBraces--;
            }
            
            if (!inString && inJSXBraces === 0 && char === '/' && pos + 1 < content.length && content[pos + 1] === '>') {
                tagContent += '>';
                break;
            }
            pos++;
        }
        
        // Extract all props
        const src = extractAttrFromTag('src', tagContent);
        if (src) {
            const title = extractAttrFromTag('title', tagContent);
            const desc = extractAttrFromTag('desc', tagContent);
            const id = extractAttrFromTag('id', tagContent);
            const data = extractAttrFromTag('data', tagContent);
            const frameless = /\bframeless\b/i.test(tagContent);
            const wideAttr = /\bwide\b/i.test(tagContent);
            const skipGallery = /\bskipGallery\b/i.test(tagContent);
            
            // Parse config if present
            let config = null;
            const jsxConfigMatch = tagContent.match(/config\s*=\s*\{\{/i);
            if (jsxConfigMatch) {
                try {
                    const configStart = tagContent.indexOf('{{', jsxConfigMatch.index) + 2;
                    let braceCount = 1;
                    let configEnd = configStart;
                    for (let i = configStart; i < tagContent.length && braceCount > 0; i++) {
                        if (tagContent[i] === '{') braceCount++;
                        if (tagContent[i] === '}') braceCount--;
                        if (braceCount === 0) configEnd = i;
                    }
                    const jsxContent = tagContent.substring(configStart, configEnd).trim();
                    config = new Function('return ({' + jsxContent + '})')();
                } catch (e) {
                    // Config parsing failed, keep null
                }
            }
            
            const isWide = isPositionInsideWide(content, position) || wideAttr;
            
            visuals.push({
                type: 'embed',
                position,
                src,
                title,
                desc,
                id,
                data,
                frameless,
                config,
                wide: isWide,
                skipGallery
            });
        }
    }
    
    // Find all Stack blocks to detect grouped images
    const stackBlocks = [];
    const stackPattern = /<Stack([\s\S]*?)>([\s\S]*?)<\/Stack>/gi;
    while ((match = stackPattern.exec(content)) !== null) {
        const stackAttrs = match[1];
        const stackContent = match[2];
        const stackStart = match.index;
        const stackEnd = stackStart + match[0].length;
        
        // Check if this Stack contains <Image> components
        const innerImages = [];
        const innerImagePattern = /<Image([^>]*)\/?>/gi;
        let imgMatch;
        while ((imgMatch = innerImagePattern.exec(stackContent)) !== null) {
            const tag = imgMatch[0];
            const srcM = tag.match(/src\s*=\s*\{([^}]+)\}/i);
            if (srcM) {
                const varName = srcM[1].trim();
                const altM = tag.match(/alt\s*=\s*["']([^"']+)["']/i);
                const captionM = tag.match(/caption\s*=\s*["']([^"']+)["']/i);
                const imgSkipGallery = /\bskipGallery\b/i.test(tag);
                innerImages.push({
                    src: varName,
                    resolvedFilename: imageImports.get(varName) || null,
                    alt: altM ? altM[1] : 'Image',
                    caption: captionM ? captionM[1] : null,
                    skipGallery: imgSkipGallery,
                });
            }
        }
        
        if (innerImages.length > 0) {
            // Extract Stack layout/gap props
            const layoutM = stackAttrs.match(/layout\s*=\s*["']([^"']+)["']/i);
            const gapM = stackAttrs.match(/gap\s*=\s*["']([^"']+)["']/i);
            
            // If ALL images have skipGallery, the whole stack is skipped
            const allSkipped = innerImages.every(img => img.skipGallery);
            
            stackBlocks.push({ start: stackStart, end: stackEnd });
            visuals.push({
                type: 'stack',
                position: stackStart,
                images: innerImages,
                layout: layoutM ? layoutM[1] : '2-column',
                gap: gapM ? gapM[1] : 'medium',
                skipGallery: allSkipped,
            });
        }
    }
    
    // Helper to check if position is inside a Stack block
    const isInsideStack = (pos) => {
        return stackBlocks.some(b => pos >= b.start && pos < b.end);
    };
    
    // Extract standalone Images (not inside Stack)
    const imagePattern = /<Image[^>]*\/>/gi;
    while ((match = imagePattern.exec(content)) !== null) {
        // Skip images already captured inside a Stack
        if (isInsideStack(match.index)) continue;
        
        const srcMatch = match[0].match(/src\s*=\s*\{([^}]+)\}/i);
        if (srcMatch) {
            const varName = srcMatch[1].trim();
            const altMatch = match[0].match(/alt\s*=\s*["']([^"']+)["']/i);
            const captionMatch = match[0].match(/caption\s*=\s*["']([^"']+)["']/i);
            const skipGallery = /\bskipGallery\b/i.test(match[0]);
            const resolvedFilename = imageImports.get(varName) || null;
            visuals.push({
                type: 'image',
                position: match.index,
                src: varName,
                resolvedFilename,
                alt: altMatch ? altMatch[1] : 'Image',
                caption: captionMatch ? captionMatch[1] : null,
                skipGallery,
            });
        }
    }
    
    // Extract Tables with position
    const tablePattern = /(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)/g;
    let tableIndex = 0;
    while ((match = tablePattern.exec(content)) !== null) {
        const tableContent = match[1].trim();
        const rows = tableContent.split('\n').filter(row => row.trim());
        
        if (rows.length >= 3) {
            const headerRow = rows[0];
            const headers = splitTableRow(headerRow)
                .filter(cell => cell !== '')
                .map(cell => markdownToHtml(cell));
            
            const expectedCols = headers.length;
            
            const dataRows = rows.slice(2).map(row => {
                return parseTableRow(row, expectedCols)
                    .map(cell => markdownToHtml(cell));
            });
            
            visuals.push({
                type: 'table',
                position: match.index,
                id: `table-${tableIndex++}`,
                headers,
                rows: dataRows,
            });
        }
    }
    
    // Sort by position (order of appearance)
    visuals.sort((a, b) => a.position - b.position);
    
    return visuals;
}

/**
 * Load all visual elements (embeds, images, tables) from MDX content files
 * Returns them in order of appearance in the article
 */
export function loadAllVisualsFromMDX() {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = dirname(__filename);

    let contentDir = join(__dirname, '../content');
    if (!statSync(contentDir, { throwIfNoEntry: false })) {
        contentDir = join(__dirname, '../../src/content');
    }
    if (!statSync(contentDir, { throwIfNoEntry: false })) {
        contentDir = join(__dirname, '../../../src/content');
    }

    const allVisuals = [];
    const articleFile = join(contentDir, 'article.mdx');

    try {
        const articleContent = readFileSync(articleFile, 'utf-8');

        // Extract all visual elements from article IN ORDER (with all props)
        const articleVisuals = extractAllVisualsWithPosition(articleContent);
        articleVisuals.forEach(item => {
            item.sourceFile = 'content/article.mdx';
        });
        allVisuals.push(...articleVisuals);

        // Parse chapter order and extract from chapters
        const chapterOrder = parseArticleChapters(articleContent, contentDir);

        for (const chapterPath of chapterOrder) {
            try {
                const chapterContent = readFileSync(chapterPath, 'utf-8');
                
                // Extract all visuals IN ORDER from this chapter (with all props)
                const chapterVisuals = extractAllVisualsWithPosition(chapterContent);
                const relativePath = relative(contentDir, chapterPath);
                chapterVisuals.forEach(item => {
                    item.sourceFile = `content/${relativePath}`;
                });
                allVisuals.push(...chapterVisuals);
            } catch (error) {
                console.error(`Error reading chapter ${chapterPath}:`, error);
            }
        }

        // Process other MDX files not already handled
        const allMdxFiles = findMdxFiles(contentDir, contentDir, [], false);
        const processedFiles = new Set([articleFile, ...chapterOrder]);

        for (const filePath of allMdxFiles) {
            if (!processedFiles.has(filePath)) {
                try {
                    const rawContent = readFileSync(filePath, 'utf-8');
                    const fileVisuals = extractAllVisualsWithPosition(rawContent);
                    const relativePath = relative(contentDir, filePath);
                    fileVisuals.forEach(item => {
                        item.sourceFile = `content/${relativePath}`;
                    });
                    allVisuals.push(...fileVisuals);
                } catch (error) {
                    console.error(`Error reading ${filePath}:`, error);
                }
            }
        }
    } catch (error) {
        console.error('Error processing article:', error);
    }

    // Keep all occurrences (no deduplication)
    // Duplicates will be numbered in dataviz.astro (e.g., d3-line-chart, d3-line-chart-2)
    return allVisuals;
}
