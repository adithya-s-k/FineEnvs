---
name: deploy-to-hf-spaces
description: Deploy the article to a Hugging Face Space. Use when the user asks to deploy, push, publish, or update their Space on Hugging Face.
---

# Deploy to Hugging Face Spaces

Guide for deploying and updating a research article on Hugging Face Spaces.

## Prerequisites

- A Hugging Face account with an SSH key configured (`git@hf.co`)
- Or HTTPS access with `huggingface-cli login`

## First deployment

### 1. Create the Space

Via CLI:
```bash
huggingface-cli repo create <username>/<space-name> --repo-type space --space_sdk docker
```

Or manually at https://huggingface.co/new-space (select **Docker** SDK).

### 2. Add the remote

```bash
git remote add space git@hf.co:spaces/<username>/<space-name>
```

### 3. Push

```bash
git push space main
```

The Space will build automatically using the included `Dockerfile` and `nginx.conf`. Build typically takes 3-5 minutes.

## Updating an existing Space

```bash
git add -A
git commit -m "feat: describe your changes"
git push space main
```

### Force push (when needed)

If the Space history has diverged (e.g., after a factory reset or manual edit on HF):

```bash
git push space main --force
```

## Verifying deployment

1. Go to `https://huggingface.co/spaces/<username>/<space-name>`
2. Check the **Factory** tab for build logs
3. The build is complete when status shows "Running"
4. If the page looks stale, append `?v=<timestamp>` to the URL to bypass CDN cache

### Cache busting

Hugging Face Spaces aggressively cache static assets. If changes don't appear after a successful build:

1. Check that the CSS bundle hash has changed (view page source, look for `/_astro/*.css`)
2. If unchanged, the build used a stale cache - push an empty commit to force rebuild:
   ```bash
   git commit --allow-empty -m "chore: trigger rebuild"
   git push space main
   ```
3. Wait for the new build to complete before checking again

## README tag (critical)

The `README.md` YAML frontmatter **must** contain:

```yaml
tags:
  - research-article-template
```

**NEVER remove this tag.** It is used by the [Research Article Gallery](https://huggingface.co/spaces/tfrere/research-article-gallery) to list the article. Without it, the Space becomes invisible in the gallery.

## README frontmatter reference

```yaml
---
title: "Article Title"
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
header: mini
pinned: false
tags:
  - research-article-template
---
```

| Field | Description |
|-------|-------------|
| `title` | Displayed in the Space card |
| `emoji` | Emoji shown on the Space card |
| `colorFrom` / `colorTo` | Gradient colors for the card header |
| `sdk: docker` | Required - uses the Dockerfile for build |
| `app_port: 8080` | Required - nginx serves on 8080 |
| `header: mini` | Compact header (recommended for articles) |
| `tags` | Must include `research-article-template` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | Check build logs in the Factory tab. Common issue: missing dependencies in `package.json` |
| CSS/layout looks wrong | Cache issue - force rebuild with empty commit |
| Images not showing | If using Git LFS, ensure LFS files are pushed. Otherwise check image paths |
| PDF not generated | Playwright is included in Docker build; check build logs for errors |
| Space shows old content | Wait 2-3 min after build completes for CDN propagation, then hard refresh |
