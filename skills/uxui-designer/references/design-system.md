# Design System Reference

## Table of Contents
- [Color Palette — Dark Mode](#color-palette--dark-mode)
- [Color Palette — Light Mode](#color-palette--light-mode)
- [Color Rules](#color-rules)
- [Typography](#typography)
- [Layout & Spacing](#layout--spacing)
- [Component Styles](#component-styles)
- [Effects & Backgrounds](#effects--backgrounds)
- [Glassmorphism](#glassmorphism)
- [Icon Libraries & SVG Rules](#icon-libraries--svg-rules)
- [Accessibility (WCAG)](#accessibility-wcag)
- [Anti-Patterns](#anti-patterns)
- [Performance & Code Quality](#performance--code-quality)

---

## ⚡ Critical Rules — Read First

> These 12 rules are placed first intentionally. They are the most violated constraints. Internalize before reading anything else.

1. **No Inter / Roboto / Arial / Space Grotesk** — use Geist, Syne, Cabinet Grotesk, DM Sans, or Bricolage Grotesque
2. **No centered hero when `DESIGN_VARIANCE > 4`** — split, asymmetric, or left-aligned only
3. **No 3 equal-width horizontal cards** — bento, zigzag, or asymmetric grid
4. **No pure black (#000000)** — use `bg-zinc-950` or `bg-[#09090b]`
5. **No neon/purple accents** — one muted accent max (emerald, deep rose, or muted electric blue)
6. **No h-screen** — use `min-h-[100dvh]`
7. **No window.addEventListener('scroll')** — use Framer Motion or GSAP ScrollTrigger
8. **No round numbers in metrics** — `47.2%` not `50%`, `11,240` not `10,000+`
9. **No vibes-only beauty** — every major visual decision must support clarity, trust, hierarchy, or task completion
10. **No overloaded first viewport** — keep above-the-fold visual complexity controlled with one dominant focal point and one primary CTA
11. **No category-breaking novelty** — experimentation is allowed only after the page type remains recognizable
12. **No motion-only meaning** — motion can enhance orientation/feedback/delight, but critical info must remain static, readable, and reduced-motion safe

---

## Color Palette — Dark Mode

| Element | Classes | Notes |
|---------|---------|-------|
| Backgrounds | `bg-zinc-950`, `bg-[#09090b]` | Never pure black (`#000000`); use Zinc/Slate off-black tokens |
| Surfaces | `bg-zinc-900/50` + `backdrop-blur-sm` | Subtle glass effect |
| Borders | `border border-white/5` or `border-white/10` | Ultra-thin and subtle |
| Headings | `text-white` | High contrast |
| Body Text | `text-zinc-400` | Readable but subdued |
| Subtle Text | `text-zinc-500`, `text-zinc-600` | Labels, captions |
| Accents | Emerald, Teal, or muted Indigo | Sparingly — glows and CTAs only |

## Color Palette — Light Mode

Only if requested by the user.

| Element | Classes | Notes |
|---------|---------|-------|
| Backgrounds | `bg-stone-50`, `bg-zinc-50`, `bg-[#f9fafb]` | Never pure white |
| Surfaces | `#ffffff` with `border border-slate-200/50` | 1px border for depth |
| Text | `#0f172a`, `text-zinc-800` | Never pure black |

## Color Rules

- Avoid saturated/neon colors (electric purple, electric blue, magenta, turquoise) — they create a cheap "Web3/AI" look
- Use off-black (`bg-zinc-950`, `bg-[#09090b]`) instead of `#000000` — pure black is too harsh and creates uncomfortable contrast with surfaces
- **AI Purple/Blue ban:** No purple button glows, no neon gradients. Use neutral bases (Zinc/Slate) with a single high-contrast accent (Emerald, Deep Rose, or Electric Blue muted)
- Max 1 accent color. Saturation < 80%
- Stick to one palette — do not mix warm and cool grays in the same project
- Color rule **60-30-10** : 60% primary (background), 30% secondary (text), 10% accent (CTA)
- Ensure WCAG AA contrast minimum (4.5:1 for normal text, 3:1 for large text)
- WCAG AAA preferred (7:1 for normal text, 4.5:1 for large text)

---

## Typography

### Multi-font system (2-3 fonts maximum)

| Usage | Font Type | Examples |
|-------|-----------|----------|
| Display/Headings | Modern geometric | Geist, Cabinet Grotesk, Satoshi, Outfit, Cal Sans, Plus Jakarta |
| Body Text | Clean, readable | DM Sans, Satoshi — avoid Inter for premium interfaces, it's too generic |
| Monospace (data/code) | Technical | Geist Mono, JetBrains Mono |

### Typography Rules
- Headlines: `text-4xl md:text-6xl tracking-tighter leading-none`
- Body: `text-base leading-relaxed max-w-[65ch]`
- Les polices Serif sont interdites sur les Dashboards/Software UIs — utilise Sans-Serif + Mono car les interfaces techniques nécessitent une lecture rapide et uniforme
- Create reusable CSS classes for each text type
- Minimum 12px font size on mobile
- Responsive font sizing (`text-5xl sm:text-6xl lg:text-7xl`)
- When `VISUAL_DENSITY > 7`: use `font-mono` for all numbers and data values — monospace aligns columns perfectly in dense data layouts

---

## Layout & Spacing

### Principles
- Mobile-first approach with Tailwind breakpoints
- Use semantic HTML5: `<header>`, `<main>`, `<section>`, `<article>`, `<aside>`, `<footer>`
- Never create "div soup" without structure
- Never use fixed pixel values for main layouts; use `%`, `rem`, `vw`, `max-w`
- Always limit reading width (`max-width`) for text content — `max-w-[65ch]` for paragraphs
- Never generate horizontal scroll
- Contain page layouts: `max-w-[1400px] mx-auto` or `max-w-7xl`

### Spacing System
- Based on 4px increments
- Generous vertical margins between sections (`py-24`, `py-32`)
- Consistent padding adapted to content
- Border-radius boomerang rule: parent radius = child radius + inner spacing

### Responsive Breakpoints
- 320px minimum (mobile)
- 768px (tablet)
- 1200px+ (desktop)

### Critical Layout Rules

- **Full-height sections:** Utilise `min-h-[100dvh]` à la place de `h-screen` — ce dernier casse sur iOS Safari car il ignore la barre d'adresse du navigateur, ce qui tronque le contenu
- **Multi-column layouts:** Pour les layouts multi-colonnes, utilise CSS Grid (`grid grid-cols-1 md:grid-cols-3 gap-6`) plutôt que le math flexbox (`calc(33% - 1rem)`) — plus lisible, plus robuste aux breakpoints, et élimine les bugs d'arrondi sous-pixel
- **Hero alignment:** When `DESIGN_VARIANCE > 4` (default: 8), centered Hero/H1 layouts are banned. Use Split Screen (50/50), Left-aligned content with Right-aligned asset, or Asymmetric whitespace instead — centered layouts are the #1 visual cliché of AI-generated pages
- **Mobile collapse:** Any asymmetric layout must aggressively fall back to `w-full px-4 py-8` single column on viewports < 768px

### Layout Types
- Use **Bento Grid** layouts (asymmetrical grids)
- Staggered columns
- Subtle diagonal sections (without Web3 extravagance)

---

## Component Styles

### Buttons
- Size: Generous (`min. px-8 py-4`) — small buttons feel cheap and are harder to tap on mobile
- Text: Bold/Semi-bold
- Hover: Marked effect (background, subtle shadow, animated icon)
- Style: Avoid standard rounded buttons
- Use "Shimmer Buttons" or "Spotlight Buttons" with subtle borders and internal glows
- On `:active` — use `-translate-y-[1px]` or `scale-[0.98]` for tactile physical push feedback

### Cards
- Background: Solid or subtle semi-transparent
- Border: 1px thin, discrete (`border-white/5` or `border-white/10`)
- Gradient: `bg-gradient-to-b from-white/5 to-transparent`
- Hover: Slight translateY/scale/shadow
- Overflow: Hidden if necessary
- If using SVG in one bento card, use SVGs in ALL cards — visual consistency across the grid is critical
- When `VISUAL_DENSITY > 7`: replace cards with `border-t` / `divide-y` dividers — boxes clutter data-dense layouts and waste vertical space

### Inputs
- Style: Minimalist
- Background: `bg-zinc-900`
- Border: `border-zinc-800`
- Focus: `focus:ring-indigo-500/20` or accent color
- Label always above input. Error text below. `gap-2` between elements.

### Interaction States [MANDATORY]

Never output only the "success/happy path" state — real UIs have loading, empty, and error states. Always include:
- **Loading:** Skeleton loaders matching the layout shape — no generic spinners (they look unfinished)
- **Empty State:** A composed empty state with a prompt to populate data
- **Error State:** Inline error message below the relevant field or action

---

## Effects & Backgrounds

Backgrounds must never be flat — depth and texture signal premium quality.

### Background Techniques
- **Noise textures** — apply **only** to `fixed inset-0 pointer-events-none` pseudo-elements, never to scrolling containers (causes GPU repaint on mobile, killing scroll performance)
- **Grid Patterns** (`bg-[linear-gradient(...)]`)
- **Animated Grid Patterns**
- **Particles** for ambience
- **Spotlights** for focus areas
- **Beams** for directional energy
- **WebGL** if necessary
- Never use simple random lines as SVG backgrounds — they look like placeholder graphics and cheapen the design

### Glows & Shadows
- No neon/fluorescent outer glows
- No heavy card shadows — use borders instead (they're sharper, more consistent across browsers, and work better in dark mode)

## Glassmorphism

Use sparingly. Beyond `backdrop-blur`, add a 1px inner border (`border-white/10`) and an inner shadow (`shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]`) to simulate physical edge refraction — this is what separates premium glass from generic blur.

- NO iOS 7-style glossy glass
- NO heavy glow effects combined with glass

---

## Icon Libraries & SVG Rules

### Icon Libraries (Choose one per project)
- `@phosphor-icons/react` **(preferred for React — flexible weights)**
- `@radix-ui/react-icons`
- `lucide-react` / Lucide Icons CDN
- Iconsax: `https://cdn.jsdelivr.net/npm/iconsax-font-icon@1.1.0/dist/icons.min.css`
- Font Awesome
- HugeIcons

**Standardize `strokeWidth` globally** — pick one value (e.g. `1.5` or `2.0`) and never mix throughout the project.

### SVG Rules
- Use SVGs for important icons — never emojis in production (they render differently across OS/browsers)
- Never use PNG images for simple icons — SVGs scale perfectly and weigh less
- Create custom SVG graphics to add creativity (small graphs, icons, pictograms)
- Create or use SVG illustrations for hero and key sections
- Optimize all images: compression + lazy-load
- Use clean placeholders if needed: `https://picsum.photos/seed/{slug}/800/600` or `https://placehold.co/400x250` — **never Unsplash URLs** (they break due to rate limiting and URL structure changes)

---

## Accessibility (WCAG)

### Contrast Targets
- **WCAG AA minimum:** 4.5:1 for normal text, 3:1 for large text
- **WCAG AAA preferred:** 7:1 for normal text, 4.5:1 for large text or bold

### Accessibility Checklist
- [ ] Meaningful `alt` text on useful images
- [ ] `aria-hidden` on decorative SVGs
- [ ] Visible focus states on all interactive elements
- [ ] Smooth keyboard navigation (logical order, clear focus)
- [ ] Descriptive links and buttons (never "click here" alone)
- [ ] Sufficient contrast everywhere
- [ ] Never black text on dark/black backgrounds
- [ ] Never white text on light backgrounds
- [ ] Never use overly deep grays that fail contrast
- [ ] Accessible hover states and icon texts

---

## Anti-Patterns

### Design Anti-Patterns ❌
- NO "Web3" or overly futuristic/cyberpunk/hacker style
- NO large gradient backgrounds
- NO saturated/neon accent colors
- NO heavy glassmorphism abuse
- NO colored glows, neon shadows
- NO ultra-rounded "floating" Web3 cards
- NO intensive gradient usage
- NO "iLook" or Glossy iOS 7 style
- NO heavy drop-shadows on cards (use borders)
- NO bright/neon gradients covering the whole screen
- NO standard Bootstrap/Material Design layouts
- NO AI-look mockups or gadgets; use simple, credible SaaS UIs
- NO random/useless SVG lines in backgrounds
- NO white shadows on dark backgrounds
- NO centered hero sections when `DESIGN_VARIANCE` > 4
- NO three equal-width cards in a horizontal row — use Zig-Zag, asymmetric grid, or horizontal scroll

### Typography Anti-Patterns ❌
- NO Inter for premium/creative UI — use Geist, Satoshi, Cabinet Grotesk, or Outfit
- NO Serif fonts on Dashboards or Software UIs
- NO emojis anywhere in production code or markup

### Technical Anti-Patterns ❌
- NO inline styles except for dynamic cases
- NO `!important` unless absolutely necessary
- NO CSS duplication
- NO layout-shifting animations (only transform/opacity)
- NO `h-screen` for hero sections — use `min-h-[100dvh]`
- NO complex flexbox percentage math — use CSS Grid
- NO arbitrary z-index values — systemic contexts only
- NO uncompressed oversized images
- NO horizontal scroll
- NO `window.addEventListener('scroll')` for scroll animations — use Framer Motion or GSAP ScrollTrigger instead (RAF batching and intersection observation handled; **GSAP ScrollTrigger requires scoped cleanup in `useEffect` return: `const ctx = gsap.context(() => { /* create ScrollTriggers here */ }, containerRef); return () => ctx.revert();`**)

### Content & Copy Anti-Patterns ❌
- NO corporate filler openers: "In today's fast-paced world", "We help teams...", "The future of..."
- NO round fake numbers: `99.99%`, `50%`, `$1M+`, `10,000+ users` — use organic specifics like `47.2%`, `$1.3M`, `11,240 users`
- NO generic startup slop names or taglines that could apply to any product
- NO CTA copy like "Learn More" or "Get Started" when a specific action is possible
- NO repeated value props saying the same thing three times in different words
- NO fake testimonials with obviously placeholder names and impossible metrics
- NO logo strip used as the only proof element — it must accompany a concrete claim

### Code & Implementation Anti-Patterns ❌
- NO `use client` on every component — isolate interactivity to leaf nodes only
- NO visible scaffold fingerprints in production output (TODO comments, placeholder function names, unused imports)
- NO dashboard screenshot in a hero section at a scale where the UI is unreadable
- NO ShadCN defaults used without customization (always override radii, colors, and shadows)
- NO nested scroll containers inside showcase cards that expose browser-default scrollbars
- NO mixing GSAP and Framer Motion inside the same component tree without isolation

---

## Performance & Code Quality

### Code Standards
- Clean, indented HTML and Tailwind
- Reusable classes
- Semantic HTML structure
- Preconnect for external fonts
- Optimized, compressed images
- Lazy loading for images
- Animations that don't cause layout shifts
- Don't stop at 500 lines; if you haven't finished, go all the way through the process

### Performance Rules
- Grain/noise textures: apply ONLY to `fixed inset-0 pointer-events-none` pseudo-elements — never to scrolling containers
- Only animate `transform` and `opacity` — never width/height/top/left (these trigger layout recalculation on every frame)
- All `useEffect` animations must include a cleanup function
- Perpetual animations (infinite loops) must be memoized with `React.memo` and isolated in their own Client Component — never trigger re-renders in a parent layout
