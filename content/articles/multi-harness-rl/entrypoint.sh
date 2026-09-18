#!/bin/bash
set -e

echo "🚀 Starting Hugging Face Space..."

# Ensure proper permissions for the app directory at runtime
chmod -R 755 /app/src/content/assets 2>/dev/null || true

# Check if Notion import is enabled and token is available
if [ "${ENABLE_NOTION_IMPORT:-false}" = "true" ] && [ -n "$NOTION_TOKEN" ] && [ -n "$NOTION_PAGE_ID" ]; then
    echo "🔄 Notion import enabled - fetching content from Notion..."
    echo "   Page ID: $NOTION_PAGE_ID"
    
    cd /app
    
    if npm run notion:import 2>&1; then
        echo "✅ Notion import completed successfully!"
        
        echo "🔨 Rebuilding site with Notion content..."
        npm run build 2>&1
        
        # PDF export only if Playwright was installed at build time
        if [ "${ENABLE_PDF_EXPORT:-false}" = "true" ] && command -v chromium > /dev/null 2>&1; then
            echo "📄 Generating PDF..."
            npm run export:pdf -- --theme=light --wait=full 2>&1 || echo "⚠️  PDF generation failed (non-critical)"
        else
            echo "⏭️  Skipping PDF generation (Playwright not available)"
        fi
        
        echo "✅ Site rebuilt and ready!"
    else
        echo "⚠️  Notion import failed - using pre-built content"
    fi
else
    if [ "${ENABLE_NOTION_IMPORT:-false}" = "true" ]; then
        echo "⚠️  Notion import enabled but NOTION_TOKEN or NOTION_PAGE_ID not found"
        echo "   NOTION_TOKEN: ${NOTION_TOKEN:+SET}${NOTION_TOKEN:-NOT SET}"
        echo "   NOTION_PAGE_ID: ${NOTION_PAGE_ID:+SET}${NOTION_PAGE_ID:-NOT SET}"
        echo "   → Using pre-built content"
    else
        echo "⏭️  Notion import disabled - serving pre-built content"
    fi
fi

echo "🌐 Starting nginx on port 8080..."
exec nginx -g 'daemon off;'
