---
name: Technical Precision Instrument
colors:
  surface: '#111317'
  surface-dim: '#111317'
  surface-bright: '#37393e'
  surface-container-lowest: '#0c0e12'
  surface-container-low: '#1a1c20'
  surface-container: '#1e2024'
  surface-container-high: '#282a2e'
  surface-container-highest: '#333539'
  on-surface: '#e2e2e8'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#e2e2e8'
  inverse-on-surface: '#2f3035'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#ffb95f'
  on-secondary: '#472a00'
  secondary-container: '#ee9800'
  on-secondary-container: '#5b3800'
  tertiary: '#4edea3'
  on-tertiary: '#003824'
  tertiary-container: '#1bbd85'
  on-tertiary-container: '#00452e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#ffddb8'
  secondary-fixed-dim: '#ffb95f'
  on-secondary-fixed: '#2a1700'
  on-secondary-fixed-variant: '#653e00'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#111317'
  on-background: '#e2e2e8'
  surface-variant: '#333539'
typography:
  headline-xl:
    fontFamily: Geist
    fontSize: 36px
    fontWeight: '600'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Geist
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Geist
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 26px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 22px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  code-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  code-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  code-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 12px
    letterSpacing: 0.08em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-desktop: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style
The design system embodies the ethos of rigorous machine learning research: deterministic, information-dense, reproducible, and stripped of ornamental consumer fluff. It targets machine learning researchers, geospatial data scientists, and computer vision engineers analyzing remote sensing benchmarks (e.g., LoveDA). 

The emotional response is focused authority, surgical precision, and academic credibility. Drawing from modern developer tools, high-caliber research repositories, and scientific visualization environments, the interface implements an engineered **Technical Minimalist / Structured Monospace** style. Visual layers rely on precise hairline rules (1px), deep charcoal value tiers, crisp geometric typography, and strict semantic color containment. Ambient glows, skeuomorphic extrusion, and whimsical gradients are completely absent. Every layout element exists to communicate network topology, feature tensors, segmentation masks, or loss convergence metrics.

## Colors
The color system enforces strict functional zoning across model representations:

- **Canvas & Structural Neutrals:**
  - `canvas-base`: `#0f1115` (Deep foundational charcoal)
  - `surface-card`: `#16191f` (Primary container plane)
  - `surface-subtle`: `#1e222b` (Nested panels, table headers, inactive states)
  - `border-subtle`: `#232730` (Standard structural grid line and card separator)
  - `border-strong`: `#333945` (Interactive borders, active card states, divided handles)
  - `text-primary`: `#f3f4f6` (High-contrast scientific notation and headings)
  - `text-secondary`: `#9ca3af` (Metadata, parameter names, axis labels)
  - `text-muted`: `#6b7280` (Disabled toggles, inactive tab glyphs)

- **Semantic Model Pathway System:**
  - **Primary (`#06b6d4` / `#0284c7`):** Dedicated exclusively to the primary segmentation pathway, semantic mask overlays, mIoU performance metrics, and UNet/VMamba latent feature representations.
  - **Secondary (`#f59e0b` / `#d97706`):** Strictly reserved for boundary-gating mechanisms, edge-supervision heads, boundary loss coefficients (e.g., Active Contour loss, boundary F1-score), and boundary attention gates. Never use this amber accent for general warnings or casual UI decorations.
  - **Tertiary (`#10b981`):** Benchmark delta gains, pass states, evaluation validation markers.
  - **Error / Critical (`#ef4444`):** Gradient explosion alerts, NaN loss indicators, OOM flags.

## Typography
Typography is organized around a dual-engine hierarchy: **Geist/Inter** governs narrative structure and layout navigation, while **JetBrains Mono** communicates data, formulas, hyperparameters, tensor dimensions, and quantitative benchmark statistics.

- Headings utilize tight tracking (`-0.02em`) to deliver crisp structure matching technical research publications.
- Monospace labels (`label-caps`) are consistently uppercase to denote tensor dimensional states (e.g., `B×C×H×W`), layer types (`VSS_BLOCK`), and loss terms (`L_BND`).
- Tabular figures (`font-variant-numeric: tabular-nums`) must be enabled across all body and metric instances to prevent layout shifting during inference loops.

## Layout & Spacing
The layout implements a rigid 8px baseline rhythm across an asymmetrical, high-utility fluid grid:

- **Desktop Layout:** A 12-column split configuration featuring a fixed 280px left workbench (model settings, checkpoint selector, LoveDA urban/rural split controls), a fluid 7-column primary viewport (dual canvas comparison: RGB Orthophoto vs. Boundary-Gated Inference), and a 3-column right telemetry inspector (mIoU tables, boundary error maps, tensor profiling).
- **Sub-grid Partitioning:** Multi-panel visualization areas collapse boundaries into 1px shared hairline rules (`#232730`) rather than nested cards with doubled borders.
- **Micro-spacing:** Metric pills, chip bars, and tensor inputs use strict `space-xs` (4px) and `space-sm` (8px) gaps to ensure maximum information density without visual crowding.

## Elevation & Depth
Elevation is achieved purely through **tonal value shifts** and **precise 1px hairlines**, rejecting blurred drop shadows or skeuomorphic elevation.

1. **Base Layer (`#0f1115`):** Root canvas underpinning data grids and workspace viewports.
2. **Container Layer (`#16191f` with 1px border `#232730`):** Model inspector cards, inference canvases, metric summaries.
3. **Elevated Overlays (`#1e222b` with 1px border `#333945`):** Dropdowns, parameter tooltips, interactive split-pane divider bars, and checkpoint popovers.
4. **Active State Indication:** No z-index float; state change is communicated by transitioning the border color from `#232730` to either `#06b6d4` (segmentation active) or `#f59e0b` (boundary-gating active).

## Shapes
Geometry is disciplined and restrained:

- **Root Containers & Cards:** `0.5rem` (8px) radius with structural 1px inner or perimeter hairline borders.
- **Controls & Input Fields:** `0.25rem` (4px) radius for buttons, numerical stepper inputs, and toggle tabs.
- **Metric Badges & Status Chips:** `0.25rem` (4px) with subtle 1px stroke borders. No circular pill buttons (`rounded-full`), maintaining the instrument aesthetic over consumer playfulness.

## Components

- **Buttons:**
  - *Primary Segmentation Action:* `#06b6d4` background, `#0f1115` bold text, 4px radius, hover to `#22d3ee`.
  - *Boundary-Gating Trigger:* `#f59e0b` background, `#0f1115` bold text, 4px radius, hover to `#fbbf24`.
  - *Secondary / Tooling:* Surface `#1e222b`, 1px border `#232730`, text `#f3f4f6`, hover border `#333945`.
  - *Metrics Action (Compact):* JetBrains Mono font, 24px height, padding 0 8px.

- **Metric Badges & Value Chips:**
  - Monospace values accompanied by tiny uppercase labels (e.g., `mIoU: 52.84%` in cyan, `Boundary-F1: 48.12%` in amber).
  - Background is a 10% opacity tint of the respective semantic color with a matching 1px solid border at 30% opacity.

- **Segmentation & Boundary Comparative Viewport:**
  - Split-screen comparison canvas flanked by a 1px interactive slider handle (`#333945` background with an active `#06b6d4` or `#f59e0b` center node).
  - Synchronized pan/zoom controls with coordinate readouts (`X: 1204.5 Y: 340.2 Zoom: 400%`) positioned bottom-left in monospace font.

- **Checkboxes & Radios:**
  - Sharp 4px square inputs with deep `#16191f` background and `#232730` border.
  - Checked state fills with `#06b6d4` for mask classes or `#f59e0b` for boundary edge toggles, displaying a crisp white geometric icon.

- **Input Fields & Param Steppers:**
  - JetBrains Mono text, height 32px, background `#16191f`, 1px border `#232730`. Focus transitions border to `#06b6d4` without fuzzy outer rings.
  - Inline unit suffixes (e.g., `lr: 1e-4`, `SS2D scan: 4-way`) right-aligned in `#6b7280`.

- **Cards & Data Tables:**
  - Header strip features a 32px height, `#1e222b` surface, bottom 1px rule `#232730`, and monospace title.
  - Tables utilize condensed row heights (28px), hairline horizontal dividers, and right-aligned numeric data columns.