# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the ICLR 2026 Blogposts Track repository, a Jekyll-based website built using the **al-folio** academic theme. It hosts blog posts for the ICLR 2026 conference blogposts track.

## Build System and Development Commands

### Local Development (Recommended)
```bash
# Using Docker (Recommended method)
./bin/docker_run.sh
```

### Local Development (Legacy - Ruby/Bundler)
```bash
# Install dependencies
bundle install

# Serve the site locally
bundle exec jekyll serve --future
```

### Building and Deployment
```bash
# Build static site
bundle exec jekyll build

# Deploy to GitHub Pages
./bin/deploy

# Manual build with CSS purging
bundle exec jekyll build
purgecss -c purgecss.config.js
```

### Development Tools
- **Prettier**: Code formatting for Liquid templates and other files
- **Pre-commit**: Basic linting and file formatting hooks
- **Docker**: Containerized development environment (recommended)

## Architecture and Key Components

### Jekyll Structure
- **`_config.yml`**: Main configuration file for Jekyll and site settings
- **`_layouts/`**: HTML templates for different page types
- **`_includes/`**: Reusable HTML components
- **`_posts/`**: Blog posts directory
- **`_pages/`**: Static pages
- **`_data/`**: YAML data files for configuration
- **`assets/`**: Static assets (images, CSS, JS)
- **`_bibliography/`**: BibTeX files for publications

### Collections
- **`books/`**: Book collection
- **`news/`**: News items (displayed on homepage)
- **`projects/`**: Project collection (grid layout)

### Key Features
- **Distill-style blog posts**: Use `layout: distill` for academic blog posts
- **Math support**: MathJax for mathematical notation
- **Code highlighting**: Rouge syntax highlighting
- **Responsive images**: ImageMagick for WebP conversion
- **Publication system**: Jekyll Scholar for BibTeX-based publications
- **Dark mode**: Automatic light/dark theme switching

### Content Management
- **Blog posts**: Create in `_posts/` with YAML frontmatter
- **Publications**: Manage in `_bibliography/` using BibTeX format
- **Projects**: Add to `_projects/` with markdown files
- **Authors**: Configure in `_data/coauthors.yml` for automatic linking

### Configuration Highlights
- **URL structure**: `https://iclr-blogposts.github.io/2026/`
- **Analytics**: Configurable Google Analytics, Pirsch, Openpanel support
- **Comments**: Giscus integration (recommended) or Disqus (deprecated)
- **SEO**: Open Graph and Schema.org metadata support

## Working with This Repository

### Creating Blog Posts
1. Create new files in `_posts/` with format `YYYY-MM-DD-title.md`
2. Use YAML frontmatter with layout: distill for academic posts
3. Include bibliography file reference for citations
4. Enable math and code features as needed

### Managing Publications
- Edit `_bibliography/papers.bib` or additional .bib files
- Use custom BibTeX keywords for buttons (pdf, arxiv, code, etc.)
- Configure author identification in `_config.yml`

### Development Environment
- Use Docker setup for consistent development experience
- Site runs on port 8080 when using Docker
- Future posts supported via `future: true` configuration

### Publishing Workflow
1. Create content and test locally
2. Commit changes to main branch
3. GitHub Actions automatically deploy to gh-pages branch
4. Manual deployment available via `./bin/deploy`

## Important Notes

- This is a **project repository** (not user/organization site), so `baseurl: /2026` is set
- GitHub Pages deployment uses automated workflow triggered by commits
- The theme uses extensive Jekyll plugins for academic features
- Image optimization handled automatically via ImageMagick
- CSS purging used to reduce final bundle size