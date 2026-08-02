# Style Recipes

Load this file when `/build` is invoked or when the Style Auto-Router has selected a mood.
Use these recipes to choose components, layouts, and motion patterns **before** building from scratch.
Every recipe maps to the mood system in `SKILL.md` and feeds into the image-generator pipeline.

> Rule: recipes are a starting point, not a checklist. Adapt every element to the actual brief — never paste a recipe unchanged.

---

## dark-saas

**Identity:** Precise, uncompromising, dark-mode native. Feels like a tool that costs money and works. References: Linear, Vercel, Raycast.

### Evidence-backed expansion

When an evidence pack is available, enrich this recipe with source-derived design DNA before building:

- **Token extraction:** top colors, font families, class patterns, section tags, and detected stack become inputs to the Design Contract.
- **Layout extraction:** map repeated signals into a compositional thesis: sticky nav, asymmetric split, bento variation, canvas/SVG anchor, ruled stat rows, or scroll chaptering.
- **Motion extraction:** map `keyframes`, `transition`, `animation`, and `transform` values into a Motion Contract with trigger, target, property, easing, duration, stagger, loop, and reduced-motion fallback.
- **Anti-copy filter:** never copy exact HTML/CSS, brand assets, screenshots, or proprietary sequences. Translate patterns into original components.

A compact evidence summary is enough:

```json
{
  "reference_count": 47,
  "dominant_stacks": ["webgl/canvas", "nextjs", "webflow", "gsap", "framer-motion", "lottie"],
  "layout_signal_counts": {
    "uses_flex": 47,
    "uses_sticky_fixed_nav": 47,
    "uses_grid": 46,
    "uses_svg": 42,
    "uses_video_or_canvas": 24,
    "uses_backdrop_blur": 27
  },
  "motion_patterns": ["fade-in", "fade-in-up", "marquee", "mask reveal", "tactile hover"]
}
```

Use this to make outputs more faithful to real premium web patterns while keeping the final design original.

### Layout Moves
- Asymmetric split hero: copy left (2/5), interactive demo right (3/5)
- Bento grid for features: cells with varied sizes, not equal columns
- Ruled stat section: numbers left-aligned, labels right-aligned, thin `border-t` dividers
- Sticky section index for long pages (thin left rail, active state synced to scroll)

### Components to Prioritize
- `Spotlight Card` (ReactBits) — for feature tiles with directional lighting
- `Magic Bento` (ReactBits) — for the main feature grid
- `Count-Up` (ReactBits) — for hero or social proof stats
- `Animated List` (ReactBits) — for live activity feeds or notification demos
- `Ghost Cursor` + `Type-Delete-Retype` (motion-patterns.md §§ 1-2) — for interactive product demos
- `Aceternity card-spotlight` — for pricing or feature cards
- `Aceternity background-lines` — for hero or section atmosphere

### Motion Profile
- `MOTION_INTENSITY 6`: staggered text reveals on hero, scroll-triggered section entrances, perpetual micro-motion in bento cells
- Hero reveal: staggered `blur-text` or `split-text` (ReactBits), 40ms stagger between words
- Card hover: subtle `translateY(-2px)` + `border-white/20` (no glow)
- CTA hover: directional fill + icon shift right `4px`

### Image Mood (for image-generator.md)
- Atmosphere: deep architectural tech environments, volumetric fog, teal-to-zinc gradients
- Feature illustrations: isometric system diagrams, circuit-board macros on near-black bg
- NO lifestyle stock, NO abstract blobs, NO bright white scenes

### Avoidances
- purple/indigo accents as the main brand color
- three equal feature cards side by side
- gradient text on H1
- rounded pill buttons with glowing shadows

---

## dashboard-pro

**Identity:** Dense, operator-grade, information-first. Every pixel earns its place. References: Linear issue tracker, Grafana, Retool.

### Layout Moves
- Full-width top nav + left sidebar or top tab rail — no marketing hero
- Data grid with sticky header + row hover highlight
- Split panels: control left, live preview right
- Metric cards in a 4-column strip above the main content area
- Timeline or activity feed as a secondary right-rail component

### Components to Prioritize
- `Animated List` (ReactBits) — live activity feeds, notification streams
- `Count-Up` (ReactBits) — KPI metric cards with spring-eased number transitions
- `Autonomous Data Dashboard` (motion-patterns.md § 5) — self-running metric demos
- `Aceternity expandable-card` — collapsible row detail panels
- `Aceternity animated-tooltip` — contextual data labels on hover
- `Morphing Hero Metric` (motion-patterns.md § 6) — for live KPI cycling in hero metric strips

### Motion Profile
- `MOTION_INTENSITY 4`: utility-first motion only. No theatrical reveals.
- Row hover: `bg-zinc-800/50` background shift, `200ms` ease
- Number updates: `useSpring` counter with `stiffness: 60, damping: 20` — never instant swap
- New row entry: slide in from top `translateY(-8px)` over `300ms`
- No perpetual floating animations — motion communicates state, not decoration

### Image Mood (for image-generator.md)
- Feature illustrations: clean UI mockup crops, abstract data flow diagrams
- Hero: a realistic dashboard screenshot with data visible, or an abstract node graph
- Section atmosphere: dark structured grid patterns, geometric data visualization forms
- NO lifestyle imagery, NO blurred light blobs

### Avoidances
- marketing hero above a dashboard — start directly with the product
- generic card spam for every data point
- decorative motion that delays data visibility
- serif fonts anywhere in the interface

---

## editorial-premium

**Identity:** Typographic, slow-paced, reads like a magazine. The page feels authored, not assembled. References: Are.na, Loewe, A24, Stripe Press.

### Layout Moves
- Full-bleed hero image with overlaid headline (large, tight tracking)
- Long vertical rhythm: one idea per viewport, generous `py-40` sections
- Editorial split: narrow text column (55ch) offset left, large image right
- Ruled chapter openers: thin `border-t` + small section number + large headline
- Quote blocks: left-border accent line, large type, no card wrapper
- Wide image ledger: full-width image + caption below in mono

### Components to Prioritize
- `Scroll Float` or `Scroll Reveal` (ReactBits) — paragraph-by-paragraph reveal
- `Split Text` (ReactBits) — headline letter-by-letter entrance
- `Image Mask on Scroll` (GSAP patterns) — hero image reveals as user scrolls
- `Text Masking` (GSAP patterns) — text revealed through a shape or stripe
- `Directional Marquee` (GSAP patterns) — for logo strips or editorial pull-quotes
- `Aceternity images-slider` — for multi-image chapter transitions
- `Aceternity following-pointer` — for subtle pointer-aware image panels

### Motion Profile
- `MOTION_INTENSITY 5`: paced and elegant. No micro-interaction spam.
- Hero reveal: full-image fade in over `800ms`, then headline slides up `24px` with `600ms` ease-out
- Section entry: opacity 0 → 1 as section enters viewport, `500ms`, no translateY (no bounce)
- Hover on image: very slight scale `1.02` over `600ms` ease
- No perpetual motion. No floating loops. If it moves, it moves once with purpose.

### Image Mood (for image-generator.md)
- Hero: high-contrast editorial photography, strong light direction, subject off-center
- Feature: curated object shots, material closeups, spatial scenes
- Atmosphere: film-grain texture sections, deep shadow architectural details
- NO generic stock photos, NO abstract gradients, NO UI mockups as decoration

### Avoidances
- card grids for any content — use lists, splits, ruled rows instead
- equal-sized images side by side
- pill badges and feature icons as decoration
- bouncy spring animations — editorial moves slowly and deliberately

---

## warm-startup

**Identity:** Human, professional, approachable. Feels like a company with real people behind it. References: Notion, Loom, Linear (marketing site), Framer.

### Layout Moves
- Split-screen hero: lifestyle or team image right, copy left with generous padding
- Staggered testimonial stack: 3 cards, vertically offset, slightly different sizes
- Feature section with alternating image-left / copy-right and copy-left / image-right rows
- Process rail: numbered steps with a connecting line, honest copy, realistic timeline
- Warm footer: team photo strip, mission statement in large type, soft bg color shift

### Components to Prioritize
- `Fade Content` (ReactBits) — section entrances with warm ease
- `Glare Hover` (ReactBits) — subtle warmth on card hover
- `Count-Up` (ReactBits) — social proof stats with natural easing
- `Aceternity animated-testimonials` — for the testimonial section
- `Aceternity card-hover-effect` — for feature rows
- `Algorithm Step Visualizer` (motion-patterns.md § 3) — if product is workflow-oriented

### Motion Profile
- `MOTION_INTENSITY 5`: fluid but never theatrical.
- Hero: text stagger `300ms` per line, image fades in `600ms` after text
- Testimonials: fade in on scroll, no bounce
- Feature rows: alternating slide-in from left/right on scroll entry, `400ms` ease-out
- CTA hover: warm background fill, `200ms` ease, no glow

### Image Mood (for image-generator.md)
- Hero: warm natural light, human elements, real environments (not studio setups)
- Features: product UI crops with warm color grading, lifestyle context
- Atmosphere: soft bokeh, wood/textile textures, indoor ambient light
- NO cold blue corporate imagery, NO abstract tech visualizations

### Avoidances
- too dark — this mood lives in light or neutral mode
- neon accents or dark-mode defaults
- aggressive asymmetry that breaks the friendly composition
- clinical spacing that removes warmth

---

## product-cinematic

**Identity:** Immersive, product-as-hero, dramatic. The page feels like a product launch event. References: Apple product pages, Framer launch pages, Arc browser site.

### Layout Moves
- Cinematic full-viewport hero: product at center, copy minimal, deep dark background
- Horizontal scroll gallery for feature storytelling — each panel reveals one capability
- Sticky scroll narrative: product stays fixed while copy changes below it
- Large product screenshot or 3D render as the dominant visual in every other section
- Chapter-based structure: sections feel like acts, not feature lists

### Components to Prioritize
- `Image Sequence` (GSAP) — product rotating or assembling on scroll
- `Image Mask on Scroll` (GSAP) — cinematic reveal of product sections
- `Horizontal Scrolling Gallery` (GSAP) — feature storytelling panels
- `Aceternity hero-parallax` — layered product depth on scroll
- `Aceternity apple-cards-carousel` — for feature chapter navigation
- `Glare Hover` (ReactBits) — premium product surface reflection
- `Aceternity 3d-pin` — interactive product callout pins

### Motion Profile
- `MOTION_INTENSITY 8`: scroll choreography is the primary UX. Motion is the content.
- Hero: product fades in at `0` opacity, scales from `0.95` to `1` over `800ms` on load
- Scroll: GSAP ScrollTrigger pinned sequences — product stays as copy scrolls past
- Chapter transitions: curtain wipe or image mask reveal between acts
- Hover on product surface: 3D tilt `15deg max` with `perspective(800px)`

### Image Mood (for image-generator.md)
- Hero: dramatic product renders on deep black, rim-lit, single directional light source
- Features: precision product closeups, materials and edges in sharp focus
- Atmosphere: dark industrial or architectural environments that frame the product
- NO lifestyle stock, NO team photos, NO abstract gradients

### Avoidances
- too much copy — cinematic means the product speaks, not the copywriter
- equal-sized content sections — this mood needs dramatic pacing variation
- standard SaaS nav with a "Get Started" pill — design a nav that fits the launch feel
- standard card grids for features

---

## swiss-precision

**Identity:** Grid-led, rational, institutional confidence. Every element placed with intent. References: International Typographic Style, Stripe documentation, IBM Design Language.

### Layout Moves
- Strict 12-column grid, all elements snap to column boundaries
- Full-width ruled rows for feature lists (thin `border-t`, number column, description)
- Large display type at unusual sizes (not Tailwind defaults — use `clamp()`)
- Diagram-centric sections: process flows, system maps, comparison tables
- Typographic poster hero: large left-aligned headline, minimal color, strong grid

### Components to Prioritize
- `Draw a Path` (GSAP) — animated SVG line diagrams
- `Animate Along a Path` (GSAP) — process flow visualizations
- `Responsive Line Splits on Scroll` (GSAP) — controlled headline reveal
- `Scroll Reveal` (ReactBits) — section entrances without bounce
- `Count-Up` (ReactBits) — for data-dense stat strips
- `Aceternity background-lines` — for structural grid atmosphere
- `Aceternity layout-grid` — for modular grid feature sections

### Motion Profile
- `MOTION_INTENSITY 3`: controlled, almost invisible. Structure is the spectacle.
- Hero: headline draws in left-to-right as if typed, no scale or bounce
- Section entry: opacity 0 → 1, `400ms` linear ease (not spring)
- Hover: underline slides in, or thin border appears — no background fill, no glow
- Diagrams: lines draw progressively on scroll entry

### Image Mood (for image-generator.md)
- Hero: abstract geometric compositions on neutral backgrounds, high contrast
- Features: technical diagrams, blueprint-style renders, structured data visualizations
- NO lifestyle imagery, NO colorful gradients, NO organic shapes

### Avoidances
- decorative elements without structural purpose
- curved layouts, diagonal sections, organic shapes
- more than 2 typefaces in the entire page
- any hint of the "AI SaaS" aesthetic

---

## soft-consumer

**Identity:** Friendly, rounded, joyful, accessible. Feels like an app you actually want to open. References: Duolingo, Linear onboarding, Notion's lighter marketing pages.

### Layout Moves
- Centered friendly hero with large rounded elements and a warm illustration
- Feature section with icon + title + 2-line description in soft cards with large radii
- Social proof as an avatar cluster + count ("Trusted by 14,000 creators")
- FAQ as a soft accordion with rounded containers and subtle shadows
- Footer with a playful closing CTA and a brand-colored background strip

### Components to Prioritize
- `Bounce Cards` (ReactBits) — for playful feature cards
- `Fade Content` (ReactBits) — soft section entrances
- `Sticker Peel` (ReactBits) — for onboarding or feature callout moments
- `Aceternity background-gradient` — for soft hero atmosphere
- `Aceternity animated-testimonials` — for user stories
- `Click Spark` (ReactBits) — micro-delight on CTA interaction

### Motion Profile
- `MOTION_INTENSITY 6`: spring-based, bouncy but controlled. Motion brings joy, not spectacle.
- Hero: elements bounce in with `stiffness: 120, damping: 15` spring
- Card hover: `scale(1.03)` with spring, soft shadow intensifies
- CTA: slight bounce on hover, satisfying `scale(0.97)` on press
- All springs must feel light, not heavy — this is not a cinematic product

### Image Mood (for image-generator.md)
- Hero: warm illustrations, soft gradients, abstract friendly shapes
- Features: app UI screenshots with warm color grading, rounded device frames
- Atmosphere: pastel-adjacent color fields, soft bokeh backgrounds
- NO dark moody imagery, NO hard edges, NO clinical tech environments

### Avoidances
- dark mode by default — this mood lives in light or very soft neutral
- aggressive typography (tight tracking, bold condensed fonts)
- complex asymmetric layouts that feel unfriendly
- cinematic pacing — this mood moves quickly and cheerfully
