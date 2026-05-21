# AdonaiKMD.github.io

> From Zero to Hero — Rebuilding ChatGPT with my own two hands.

## How to go live (one time setup)

```bash
# 1. Clone the repo you just created on GitHub
git clone https://github.com/AdonaiKMD/AdonaiKMD.github.io
cd AdonaiKMD.github.io

# 2. Copy all these files into it, then push
git add .
git commit -m "init: launch blog"
git push origin main
```

Your site will be live at **https://AdonaiKMD.github.io** in ~2 minutes.

---

## How to write a new post

1. Create a file in `_posts/` named: `YYYY-MM-DD-your-title.md`
2. Add this at the very top:

```yaml
---
layout: post
title: "Your Post Title"
date: 2026-06-01
categories: [arc-I, foundations]
tags: [calculus, math]
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
│   └── 2026-05-21-linear-algebra.md   ← Chapter 01
├── index.md                 ← homepage
├── about.md                 ← about page
└── README.md                ← this file
```

## To preview locally (optional)

```bash
gem install bundler jekyll
bundle init
bundle add jekyll minima
bundle exec jekyll serve
# → open http://localhost:4000
```
