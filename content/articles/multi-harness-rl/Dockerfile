# =============================================================================
# Research Article Template - Optimized Dockerfile
#
# Build args (PDF + LaTeX enabled by default to match previous behavior):
#   ENABLE_PDF_EXPORT=false      - Skip PDF generation (~45-90s faster)
#   ENABLE_LATEX_EXPORT=false    - Skip LaTeX export (~10s faster)
#   ENABLE_LATEX_CONVERSION=true - Convert LaTeX source to MDX before build
#   ENABLE_NOTION_IMPORT=true    - Pre-install Notion importer deps for runtime
# =============================================================================

FROM node:20-slim

ARG ENABLE_PDF_EXPORT=true
ARG ENABLE_LATEX_CONVERSION=false
ARG ENABLE_LATEX_EXPORT=true
ARG ENABLE_NOTION_IMPORT=true

# Persist build flags for entrypoint runtime decisions
ENV ENABLE_PDF_EXPORT=${ENABLE_PDF_EXPORT}
ENV ENABLE_LATEX_EXPORT=${ENABLE_LATEX_EXPORT}

# ----- System dependencies (minimal by default) -----
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----- Install npm dependencies (cached layer if package*.json unchanged) -----
COPY app/package*.json ./
RUN npm ci

# ----- Copy application source -----
COPY app/ .

# ----- Resolve public/data symlink -----
RUN set -e; \
    if [ -L public/data ] || { [ -e public/data ] && [ ! -d public/data ]; }; then rm -f public/data; fi; \
    mkdir -p public/data; \
    cp -a src/content/assets/data/. public/data/

# ----- LaTeX-to-MDX conversion (optional, before build) -----
RUN if [ "$ENABLE_LATEX_CONVERSION" = "true" ]; then \
    echo "🔄 LaTeX importer enabled - installing Pandoc..." && \
    apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -qO- https://github.com/jgm/pandoc/releases/download/3.8/pandoc-3.8-linux-amd64.tar.gz | tar xzf - -C /tmp && \
    cp /tmp/pandoc-3.8/bin/pandoc /usr/local/bin/ && \
    cp /tmp/pandoc-3.8/bin/pandoc-lua /usr/local/bin/ && \
    rm -rf /tmp/pandoc-3.8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    npm run latex:convert; \
    else echo "⏭️  LaTeX conversion disabled"; fi

# ----- Notion importer deps (optional, for runtime import) -----
RUN if [ "$ENABLE_NOTION_IMPORT" = "true" ]; then \
    echo "📦 Pre-installing Notion importer dependencies..." && \
    cd scripts/notion-importer && npm ci && cd ../..; \
    else echo "⏭️  Notion importer disabled"; fi

# ----- Build + Chromium download (parallelized) -----
# Chromium download runs in background while Astro builds, saving ~20s
RUN set -e; \
    if [ "$ENABLE_PDF_EXPORT" = "true" ]; then \
      echo "📄 Downloading Chromium (background)..."; \
      npx playwright install --with-deps chromium > /tmp/chromium.log 2>&1 & \
      CHROMIUM_PID=$!; \
    fi; \
    echo "🔨 Building Astro site..."; \
    npm run build; \
    if [ "$ENABLE_PDF_EXPORT" = "true" ]; then \
      echo "⏳ Waiting for Chromium download..."; \
      wait $CHROMIUM_PID; \
      cat /tmp/chromium.log; \
    fi

# ----- Exports: PDF + LaTeX in parallel -----
# Both run concurrently after build, saving ~15s
RUN set -e; \
    PIDS=""; \
    if [ "$ENABLE_PDF_EXPORT" = "true" ]; then \
      echo "📄 Starting PDF export..."; \
      npm run export:pdf -- --theme=light --wait=full > /tmp/pdf.log 2>&1 & \
      PIDS="$PIDS $!"; \
    fi; \
    if [ "$ENABLE_LATEX_EXPORT" = "true" ]; then \
      echo "📝 Installing Pandoc + starting LaTeX export..."; \
      ( if ! command -v pandoc > /dev/null 2>&1; then \
          apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
          wget -qO- https://github.com/jgm/pandoc/releases/download/3.8/pandoc-3.8-linux-amd64.tar.gz | tar xzf - -C /tmp && \
          cp /tmp/pandoc-3.8/bin/pandoc /usr/local/bin/ && \
          cp /tmp/pandoc-3.8/bin/pandoc-lua /usr/local/bin/ && \
          rm -rf /tmp/pandoc-3.8 && \
          apt-get clean && rm -rf /var/lib/apt/lists/*; \
        fi && \
        npm run export:latex \
      ) > /tmp/latex.log 2>&1 & \
      PIDS="$PIDS $!"; \
    fi; \
    if [ -n "$PIDS" ]; then \
      echo "⏳ Waiting for exports to complete..."; \
      FAIL=0; \
      for PID in $PIDS; do wait $PID || FAIL=1; done; \
      [ "$ENABLE_PDF_EXPORT" = "true" ] && cat /tmp/pdf.log; \
      [ "$ENABLE_LATEX_EXPORT" = "true" ] && cat /tmp/latex.log; \
      if [ "$FAIL" -ne 0 ]; then echo "❌ One or more exports failed"; exit 1; fi; \
      echo "✅ All exports completed"; \
    else \
      echo "⏭️  All exports disabled"; \
    fi

# ----- Nginx configuration -----
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ----- Permissions -----
RUN mkdir -p /var/cache/nginx /var/run /var/log/nginx /var/lib/nginx/body && \
    chmod -R 777 /var/cache/nginx /var/run /var/log/nginx /var/lib/nginx /etc/nginx/nginx.conf && \
    chmod -R 777 /app

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
