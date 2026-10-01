/**
 * LaTeX Metadata Extractor
 * Extracts document metadata from LaTeX files for frontmatter generation
 */

/**
 * Extract metadata from LaTeX content
 * @param {string} latexContent - Raw LaTeX content
 * @returns {object} - Extracted metadata object
 */
export function extractLatexMetadata(latexContent) {
    const metadata = {};

    // Extract title
    const titleMatch = latexContent.match(/\\title\s*\{\s*([^}]+)\s*\}/s);
    if (titleMatch) {
        metadata.title = titleMatch[1]
            .replace(/\n/g, ' ')
            .trim();
    }

    // Extract authors with their specific affiliations
    const authors = [];
    const authorMatches = latexContent.matchAll(/\\authorOne\[[^\]]*\]\{([^}]+)\}/g);

    for (const match of authorMatches) {
        const fullAuthorInfo = match[1];

        // Determine affiliations based on macros present
        const affiliations = [];
        if (fullAuthorInfo.includes('\\ensps')) {
            affiliations.push(1); // École Normale Supérieure
        }
        if (fullAuthorInfo.includes('\\hf')) {
            affiliations.push(2); // Hugging Face
        }

        // Clean author name by removing macros
        let authorName = fullAuthorInfo
            .replace(/\\ensps/g, '')      // Remove École macro
            .replace(/\\hf/g, '')         // Remove Hugging Face macro
            .replace(/\s+/g, ' ')         // Normalize whitespace
            .trim();

        // Skip empty authors or placeholder entries
        if (authorName && authorName !== '...') {
            authors.push({
                name: authorName,
                affiliations: affiliations.length > 0 ? affiliations : [2] // Default to HF if no macro
            });
        }
    }

    if (authors.length > 0) {
        metadata.authors = authors;
    }

    // Extract affiliations - create the two distinct affiliations
    metadata.affiliations = [
        {
            name: "École Normale Supérieure Paris-Saclay"
        },
        {
            name: "Hugging Face"
        }
    ];

    // Extract date if available (common LaTeX patterns)
    const datePatterns = [
        /\\date\s*\{([^}]+)\}/,
        /\\newcommand\s*\{\\date\}\s*\{([^}]+)\}/,
    ];

    for (const pattern of datePatterns) {
        const dateMatch = latexContent.match(pattern);
        if (dateMatch) {
            metadata.published = dateMatch[1].trim();
            break;
        }
    }

    // Fallback to current date if no date found
    if (!metadata.published) {
        metadata.published = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: '2-digit'
        });
    }

    return metadata;
}

/**
 * Generate YAML frontmatter from metadata object
 * @param {object} metadata - Metadata object
 * @returns {string} - YAML frontmatter string
 */
export function generateFrontmatter(metadata) {
    let frontmatter = '---\n';

    // Title
    if (metadata.title) {
        frontmatter += `title: "${metadata.title}"\n`;
    }

    // Authors
    if (metadata.authors && metadata.authors.length > 0) {
        frontmatter += 'authors:\n';
        metadata.authors.forEach(author => {
            frontmatter += `  - name: "${author.name}"\n`;
            if (author.url) {
                frontmatter += `    url: "${author.url}"\n`;
            }
            frontmatter += `    affiliations: [${author.affiliations.join(', ')}]\n`;
        });
    }

    // Affiliations
    if (metadata.affiliations && metadata.affiliations.length > 0) {
        frontmatter += 'affiliations:\n';
        metadata.affiliations.forEach((affiliation, index) => {
            frontmatter += `  - name: "${affiliation.name}"\n`;
            if (affiliation.url) {
                frontmatter += `    url: "${affiliation.url}"\n`;
            }
        });
    }

    // Publication date
    if (metadata.published) {
        frontmatter += `published: "${metadata.published}"\n`;
    }

    // Additional metadata
    if (metadata.doi) {
        frontmatter += `doi: "${metadata.doi}"\n`;
    }

    if (metadata.description) {
        frontmatter += `description: "${metadata.description}"\n`;
    }

    if (metadata.licence) {
        frontmatter += `licence: >\n  ${metadata.licence}\n`;
    }

    if (metadata.tags && metadata.tags.length > 0) {
        frontmatter += 'tags:\n';
        metadata.tags.forEach(tag => {
            frontmatter += `  - ${tag}\n`;
        });
    }

    // Default Astro configuration
    frontmatter += 'tableOfContentsAutoCollapse: true\n';
    frontmatter += '---\n\n';

    return frontmatter;
}

/**
 * Extract and generate frontmatter from LaTeX content
 * @param {string} latexContent - Raw LaTeX content
 * @returns {string} - Complete YAML frontmatter
 */
export function extractAndGenerateFrontmatter(latexContent) {
    const metadata = extractLatexMetadata(latexContent);
    return generateFrontmatter(metadata);
}
