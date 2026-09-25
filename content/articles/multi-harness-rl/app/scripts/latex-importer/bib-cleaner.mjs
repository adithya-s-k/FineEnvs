#!/usr/bin/env node

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname, basename } from 'path';

/**
 * Clean a BibTeX file by removing local file references and paths
 * @param {string} inputBibFile - Path to the input .bib file
 * @param {string} outputBibFile - Path to the output cleaned .bib file
 * @returns {boolean} - Success status
 */
export function cleanBibliography(inputBibFile, outputBibFile) {
    if (!existsSync(inputBibFile)) {
        console.log('   ⚠️  No bibliography file found:', inputBibFile);
        return false;
    }

    console.log('📚 Cleaning bibliography...');
    let bibContent = readFileSync(inputBibFile, 'utf8');

    // Remove file paths and local references
    bibContent = bibContent.replace(/file = \{[^}]+\}/g, '');

    // Remove empty lines created by file removal
    bibContent = bibContent.replace(/,\s*\n\s*\n/g, '\n\n');
    bibContent = bibContent.replace(/,\s*\}/g, '\n}');

    // Clean up double commas
    bibContent = bibContent.replace(/,,/g, ',');

    // Remove trailing commas before closing braces
    bibContent = bibContent.replace(/,(\s*\n\s*)\}/g, '$1}');

    writeFileSync(outputBibFile, bibContent);
    console.log(`   📄 Clean bibliography saved: ${outputBibFile}`);

    return true;
}

/**
 * CLI for bibliography cleaning
 */
function main() {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
📚 BibTeX Bibliography Cleaner

Usage:
  node bib-cleaner.mjs [input.bib] [output.bib]
  node bib-cleaner.mjs --input=input.bib --output=output.bib

Options:
  --input=FILE     Input .bib file
  --output=FILE    Output cleaned .bib file
  --help, -h       Show this help

Examples:
  # Clean main.bib to clean.bib
  node bib-cleaner.mjs main.bib clean.bib
  
  # Using flags
  node bib-cleaner.mjs --input=references.bib --output=clean-refs.bib
`);
        process.exit(0);
    }

    let inputFile, outputFile;

    // Parse command line arguments
    if (args.length >= 2 && !args[0].startsWith('--')) {
        // Positional arguments
        inputFile = args[0];
        outputFile = args[1];
    } else {
        // Named arguments
        for (const arg of args) {
            if (arg.startsWith('--input=')) {
                inputFile = arg.split('=')[1];
            } else if (arg.startsWith('--output=')) {
                outputFile = arg.split('=')[1];
            }
        }
    }

    if (!inputFile || !outputFile) {
        console.error('❌ Both input and output files are required');
        console.log('Use --help for usage information');
        process.exit(1);
    }

    const success = cleanBibliography(inputFile, outputFile);
    if (success) {
        console.log('🎉 Bibliography cleaning completed!');
    } else {
        process.exit(1);
    }
}

// Run CLI if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
