#!/usr/bin/env node

/**
 * Release script for Research Article Template
 * Handles semantic versioning and changelog updates
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const PACKAGE_JSON_PATH = join(process.cwd(), 'app', 'package.json');
const CHANGELOG_PATH = join(process.cwd(), 'CHANGELOG.md');

function getCurrentVersion() {
    const packageJson = JSON.parse(readFileSync(PACKAGE_JSON_PATH, 'utf8'));
    return packageJson.version;
}

function updateVersion(newVersion) {
    const packageJson = JSON.parse(readFileSync(PACKAGE_JSON_PATH, 'utf8'));
    packageJson.version = newVersion;
    writeFileSync(PACKAGE_JSON_PATH, JSON.stringify(packageJson, null, 2) + '\n');
    console.log(`✅ Updated package.json to version ${newVersion}`);
}

function updateChangelog(newVersion) {
    const changelog = readFileSync(CHANGELOG_PATH, 'utf8');
    const today = new Date().toISOString().split('T')[0];

    const updatedChangelog = changelog.replace(
        '## [Unreleased]',
        `## [Unreleased]\n\n## [${newVersion}] - ${today}`
    );

    writeFileSync(CHANGELOG_PATH, updatedChangelog);
    console.log(`✅ Updated CHANGELOG.md with version ${newVersion}`);
}

function createGitTag(version) {
    try {
        execSync(`git tag -a v${version} -m "Release version ${version}"`, { stdio: 'inherit' });
        console.log(`✅ Created git tag v${version}`);
    } catch (error) {
        console.error(`❌ Failed to create git tag: ${error.message}`);
    }
}

function main() {
    const args = process.argv.slice(2);
    const versionType = args[0]; // 'major', 'minor', 'patch'

    if (!['major', 'minor', 'patch'].includes(versionType)) {
        console.error('❌ Please specify version type: major, minor, or patch');
        console.log('Usage: node scripts/release.mjs [major|minor|patch]');
        process.exit(1);
    }

    const currentVersion = getCurrentVersion();
    const [major, minor, patch] = currentVersion.split('.').map(Number);

    let newVersion;
    switch (versionType) {
        case 'major':
            newVersion = `${major + 1}.0.0`;
            break;
        case 'minor':
            newVersion = `${major}.${minor + 1}.0`;
            break;
        case 'patch':
            newVersion = `${major}.${minor}.${patch + 1}`;
            break;
    }

    console.log(`🚀 Releasing version ${newVersion} (from ${currentVersion})`);

    // Update files
    updateVersion(newVersion);
    updateChangelog(newVersion);

    // Create git tag
    createGitTag(newVersion);

    console.log(`\n🎉 Release ${newVersion} prepared!`);
    console.log('\nNext steps:');
    console.log('1. Review the changes:');
    console.log('   git diff');
    console.log('2. Commit the changes:');
    console.log(`   git add . && git commit -m "chore: release version ${newVersion}"`);
    console.log('3. Push the changes and tags:');
    console.log(`   git push && git push --tags`);
    console.log('4. Create a release on Hugging Face Spaces');
}

main();
