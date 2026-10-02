#!/usr/bin/env node

/**
 * LaTeX Reference Preprocessor
 * 
 * This module cleans up LaTeX references BEFORE Pandoc conversion to ensure
 * consistent, MDX-compatible identifiers throughout the document.
 * 
 * What it does:
 * - Removes prefixes from labels: \label{sec:intro} → \label{sec-intro}
 * - Updates corresponding refs: \ref{sec:intro} → \ref{sec-intro}  
 * - Handles all reference types: sec:, fig:, eq:, table:, etc.
 * - Maintains consistency between labels and references
 */

/**
 * Extract all references from LaTeX content
 * @param {string} content - LaTeX content
 * @returns {Object} - Object with labels and refs arrays
 */
function extractReferences(content) {
    const references = {
        labels: new Set(),
        refs: new Set(),
        cites: new Set()
    };

    // Find all \label{...} commands
    const labelMatches = content.matchAll(/\\label\{([^}]+)\}/g);
    for (const match of labelMatches) {
        references.labels.add(match[1]);
    }

    // Find all \ref{...} commands
    const refMatches = content.matchAll(/\\ref\{([^}]+)\}/g);
    for (const match of refMatches) {
        references.refs.add(match[1]);
    }

    // Find all \cite{...} commands (already handled in existing code but included for completeness)
    const citeMatches = content.matchAll(/\\cite[tp]?\{([^}]+)\}/g);
    for (const match of citeMatches) {
        // Handle multiple citations: \cite{ref1,ref2,ref3}
        const citations = match[1].split(',').map(cite => cite.trim());
        citations.forEach(cite => references.cites.add(cite));
    }

    return references;
}

/**
 * Create clean identifier mapping
 * @param {Object} references - References object from extractReferences
 * @returns {Map} - Mapping from original to clean identifiers
 */
function createCleanMapping(references) {
    const mapping = new Map();

    // Create mapping for all unique identifiers
    const allIdentifiers = new Set([
        ...references.labels,
        ...references.refs
    ]);

    for (const id of allIdentifiers) {
        // Remove common prefixes and replace colons with dashes
        let cleanId = id
            .replace(/^(sec|section|ch|chapter|fig|figure|eq|equation|tab|table|lst|listing|app|appendix):/gi, '')
            .replace(/:/g, '-')
            .replace(/[^a-zA-Z0-9_-]/g, '-') // Replace any other problematic characters
            .replace(/-+/g, '-') // Collapse multiple dashes
            .replace(/^-|-$/g, ''); // Remove leading/trailing dashes

        // Ensure we don't have empty identifiers
        if (!cleanId) {
            cleanId = id.replace(/:/g, '-');
        }

        mapping.set(id, cleanId);
    }

    return mapping;
}

/**
 * Convert labels to HTML anchor spans for better MDX compatibility
 * @param {string} content - LaTeX content
 * @param {Map} mapping - Identifier mapping (original -> clean)
 * @returns {Object} - Result with content and count of conversions
 */
function convertLabelsToAnchors(content, mapping) {
    let processedContent = content;
    let anchorsCreated = 0;

    // Replace \label{...} with HTML anchor spans, but SKIP labels inside math environments
    for (const [original, clean] of mapping) {
        // Skip equation labels (they will be handled by the Lua filter)
        if (original.startsWith('eq:')) {
            continue;
        }

        const labelRegex = new RegExp(`\\\\label\\{${escapeRegex(original)}\\}`, 'g');
        const labelMatches = processedContent.match(labelRegex);

        if (labelMatches) {
            // Replace \label{original} with HTML span anchor (invisible but accessible)
            processedContent = processedContent.replace(labelRegex, `\n\n<span id="${clean}" style="position: absolute;"></span>\n\n`);
            anchorsCreated += labelMatches.length;
        }
    }

    return { content: processedContent, anchorsCreated };
}

/**
 * Convert \highlight{...} commands to HTML spans with CSS class
 * @param {string} content - LaTeX content
 * @returns {Object} - Result with content and count of conversions
 */
function convertHighlightCommands(content) {
    let processedContent = content;
    let highlightsConverted = 0;

    // Replace \highlight{...} with <span class="highlight">...</span>
    processedContent = processedContent.replace(/\\highlight\{([^}]+)\}/g, (match, text) => {
        highlightsConverted++;
        return `<span class="highlight">${text}</span>`;
    });

    return { content: processedContent, highlightsConverted };
}

/**
 * Apply mapping to LaTeX content
 * @param {string} content - Original LaTeX content
 * @param {Map} mapping - Identifier mapping
 * @returns {string} - Cleaned LaTeX content
 */
function applyMapping(content, mapping) {
    let cleanedContent = content;
    let changesCount = 0;

    // First, convert labels to anchor spans
    const anchorResult = convertLabelsToAnchors(cleanedContent, mapping);
    cleanedContent = anchorResult.content;
    const anchorsCreated = anchorResult.anchorsCreated;

    // Convert \highlight{} commands to spans
    const highlightResult = convertHighlightCommands(cleanedContent);
    cleanedContent = highlightResult.content;
    const highlightsConverted = highlightResult.highlightsConverted;

    // Then apply mapping to remaining references and equation labels
    for (const [original, clean] of mapping) {
        if (original !== clean) {
            // Replace \ref{original} with \ref{clean}
            const refRegex = new RegExp(`\\\\ref\\{${escapeRegex(original)}\\}`, 'g');
            const refMatches = cleanedContent.match(refRegex);
            if (refMatches) {
                cleanedContent = cleanedContent.replace(refRegex, `\\ref{${clean}}`);
                changesCount += refMatches.length;
            }

            // For equation labels, still clean the labels themselves (for the Lua filter)
            if (original.startsWith('eq:')) {
                const labelRegex = new RegExp(`\\\\label\\{${escapeRegex(original)}\\}`, 'g');
                const labelMatches = cleanedContent.match(labelRegex);
                if (labelMatches) {
                    cleanedContent = cleanedContent.replace(labelRegex, `\\label{${clean}}`);
                    changesCount += labelMatches.length;
                }
            }
        }
    }

    return {
        content: cleanedContent,
        changesCount: changesCount + anchorsCreated,
        highlightsConverted: highlightsConverted
    };
}

/**
 * Escape special regex characters
 * @param {string} string - String to escape
 * @returns {string} - Escaped string
 */
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Main preprocessing function
 * @param {string} latexContent - Original LaTeX content
 * @returns {Object} - Result with cleaned content and statistics
 */
export function preprocessLatexReferences(latexContent) {
    console.log('🔧 Preprocessing LaTeX references for MDX compatibility...');

    // 1. Extract all references
    const references = extractReferences(latexContent);

    console.log(`   📊 Found: ${references.labels.size} labels, ${references.refs.size} refs`);

    // 2. Create clean mapping
    const mapping = createCleanMapping(references);

    // 3. Apply mapping
    const result = applyMapping(latexContent, mapping);

    if (result.changesCount > 0) {
        console.log(`   ✅ Processed ${result.changesCount} reference(s) and created anchor spans`);

        // Show some examples of changes
        let exampleCount = 0;
        for (const [original, clean] of mapping) {
            if (original !== clean && exampleCount < 3) {
                console.log(`      ${original} → ${clean} (span + refs)`);
                exampleCount++;
            }
        }
        if (mapping.size > 3) {
            console.log(`      ... and ${mapping.size - 3} more anchor spans created`);
        }
    } else {
        console.log('   ℹ️  No reference cleanup needed');
    }

    if (result.highlightsConverted > 0) {
        console.log(`   ✨ Converted ${result.highlightsConverted} \\highlight{} command(s) to <span class="highlight">`);
    }

    return {
        content: result.content,
        changesCount: result.changesCount,
        mapping: mapping,
        references: references
    };
}
