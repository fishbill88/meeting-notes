---
name: design-ui
description: "Design a web UI, TUI improvement, or terminal output enhancement for Meeting Notes AI using the installed ui-ux-pro-max skill. Use when asked to add a web interface, improve terminal output, or design any visual component."
---

# Design UI — Meeting Notes AI

This project uses the **ui-ux-pro-max** skill for UI/UX decisions. The skill is installed at `.claude/skills/ui-ux-pro-max/`.

## Step 1: Generate a design system for this product

```bash
python .claude/skills/ui-ux-pro-max/scripts/search.py "productivity SaaS meeting notes offline tool" --design-system -p "Meeting Notes AI"
```

For a web dashboard:
```bash
python .claude/skills/ui-ux-pro-max/scripts/search.py "productivity dashboard notes offline app" --design-system --persist -p "Meeting Notes AI" --page "dashboard"
```

## Step 2: Get stack-specific guidance

For a web UI (recommended: Next.js + shadcn/ui or plain HTML+Tailwind):

```bash
python .claude/skills/ui-ux-pro-max/scripts/search.py "dashboard notes list" --stack nextjs
python .claude/skills/ui-ux-pro-max/scripts/search.py "dashboard notes list" --stack shadcn
```

For a Python TUI (Rich-based — current stack):
- Not directly supported by ui-ux-pro-max, but consult `--domain ux` for UX principles
- Rich docs: https://rich.readthedocs.io

## Step 3: Domain-specific lookups

```bash
# Color palette for a productivity/notes tool
python .claude/skills/ui-ux-pro-max/scripts/search.py "productivity notes" --domain color

# Typography choices
python .claude/skills/ui-ux-pro-max/scripts/search.py "professional minimal clean" --domain typography

# UX guidelines for dashboards
python .claude/skills/ui-ux-pro-max/scripts/search.py "dashboard data table" --domain ux
```

## Product context for Meeting Notes AI

- **Type:** Productivity / SaaS tool
- **Primary user:** Business professionals, developers, remote workers
- **Key values:** Clean, minimal, professional, efficient
- **Recommended style:** Minimalism or Bento Grid (matches productivity tools)
- **Recommended colors:** Blue/slate palette (trust, productivity)
- **Stack default:** `html-tailwind` or `nextjs` for web

## Extending the terminal UI (Rich)

The current CLI uses `rich` for terminal output. To improve it:
- Use `rich.table.Table` for structured data (e.g., meeting summary)
- Use `rich.tree.Tree` for action items with hierarchy
- Use `rich.layout.Layout` for a split-panel view during recording
- Reference `main.py` for existing Rich usage patterns
