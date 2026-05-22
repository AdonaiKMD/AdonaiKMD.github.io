# AdonaiKMD.github.io

> From Zero to Hero — Reaching the frontier step by step

```bash
git clone https://github.com/AdonaiKMD/AdonaiKMD.github.io
cd AdonaiKMD.github.io

git add .
git commit -m "init: launch blog"
git push origin main
```
---
## How to write a new post

1. Create a file in `_posts/` named: `YYYY-MM-DD-my-title.md` (the date in the filename is a Jekyll requirement).

2. Add this at the very top:

```yaml
---
layout: post
title: "My Post Title"
---
```

3. Write in Markdown below the `---`
4. For inline math: `$f(x) = x^2$`
5. For display math: `$$\int_0^\infty e^{-x} dx = 1$$`
6. Push to GitHub — it builds automatically.

---

## File structure

```
AdonaiKMD.github.io/
├── _config.yml              ← site settings
├── _includes/
│   └── head_custom.html     ← MathJax (don't touch)
├── _posts/
│   └── 2026-05-21-linear-algebra.md   ← Chapter 1
├── index.md                 ← homepage
├── about.md                 ← about page
└── README.md                ← this file
```

## To preview locally

```bash
gem install bundler jekyll
bundle init
bundle add jekyll minima
bundle exec jekyll serve
# → open http://localhost:4000
```
