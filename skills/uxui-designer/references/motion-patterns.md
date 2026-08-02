# Motion Patterns Reference

## Table des matières
- [Animation Principles](#animation-principles)
- [Required Animation Types](#required-animation-types)
- [Animated Components Library](#animated-components-library)
- [High Motion Rules (MOTION_INTENSITY > 5)](#high-motion-rules-motion_intensity--5)
- [GSAP vs Framer Motion](#gsap-vs-framer-motion)
- [Z-Index Management](#z-index-management)
- [Autonomous Demo Library (MOTION_INTENSITY ≥ 7)](#autonomous-demo-library-motion_intensity--7)
- [Autonomous Animation Rules](#autonomous-animation-rules)

---

## Animation Principles

- Fluid animations with natural easing (`ease-out`, `ease-in-out`)
- Duration: 0.3s to 0.6s (never < 200ms or > 1s unless specific case)
- **Only animate `transform` and `opacity`** — animating `top`, `left`, `width`, or `height` triggers layout recalculation on every frame, killing scroll performance and causing visual jank
- Never without easing (except rotation/marquee — these need constant velocity to feel natural)

## Required Animation Types

Every project must include these animation categories:

- **Scroll animations**: fade-in, slide-in, scale-in with stagger
- **Hover effects**: Visible but subtle, creative (not just 10px shifts — use scale, shadow, border glow, or icon movement)
- **Micro-interactions**: Button states, input focus, card interactions
- **Loading states**: Skeleton screens matching the design aesthetic — not generic spinners

## Motion DNA Schema

When working from scraped references or screenshots, convert animation observations into a **motion contract** before coding. The goal is not to copy timelines exactly; it is to reproduce the *quality* of the choreography in an original implementation.

```json
{
  "motion_dna": {
    "load_choreography": {
      "sequence": ["nav", "hero_label", "headline", "cta", "visual_anchor"],
      "stagger_ms": 40,
      "duration_ms": 420,
      "easing": "cubic-bezier(.22,1,.36,1)",
      "properties": ["opacity", "transform"]
    },
    "scroll_choreography": {
      "trigger": "section enters 15% viewport",
      "patterns": ["mask reveal", "fade-up", "horizontal marquee", "sticky chapter transition"],
      "library": "Framer Motion for UI, GSAP ScrollTrigger only for pinned cinematic scenes"
    },
    "hover_choreography": {
      "buttons": ["directional fill", "icon shift 4px", "active scale .98"],
      "cards": ["border brightens", "spotlight follows pointer", "translateY -2px"],
      "nav": ["underline slide", "background glass intensifies"]
    },
    "continuous_motion": {
      "allowed": ["marquee", "slow orbital", "code typing", "metric morph", "ambient gradient drift"],
      "rest_between_cycles": "2-4s",
      "reduced_motion": "pause or replace with static final state"
    }
  }
}
```

### Evidence-derived motion heuristics

From a Supahero-style public reference scrape, common high-quality hero pages often include sticky/fixed navigation, SVG/canvas/video anchors, GSAP/Lottie/Framer-like motion stacks, keyframes such as `fade-in`, `fade-in-up`, `spin`, and understated transitions. Treat these as cues for **layered choreography**: one strong entrance, one strong scroll behavior, and tactile hover states — not motion everywhere.

## Animated Components Library

Components to consider including based on project needs:

- Marquee (infinite scrolling for logos/text)
- Terminal (MacOS style for CLI showcase)
- Hero Video Dialog
- Bento Grid with hover effects
- Globe (WebGL autorotating)
- Orbiting Circles
- Shimmer Button
- Laser Flow effects
- Spotlight effects
- Carousel with controls
- Testimonial sliders
- Parallax sections
- Sticky elements

---

## High Motion Rules (MOTION_INTENSITY > 5)

When `MOTION_INTENSITY > 5`:

- Embed continuous, infinite micro-animations (Pulse, Typewriter, Float, Shimmer) in standard components — the UI should feel alive, not static between interactions
- For magnetic buttons/cursor effects: use **exclusively** Framer Motion's `useMotionValue` + `useTransform` — using `useState` for continuous animations triggers React re-renders on every frame, destroying mobile performance
- Apply Spring Physics to all interactive elements: `type: "spring", stiffness: 100, damping: 20` — avoid linear easing which feels mechanical and robotic
- Use Framer's `layout` and `layoutId` props for smooth re-ordering and shared element transitions
- Stagger list/grid mounts: `staggerChildren` in Framer or `animation-delay: calc(var(--index) * 100ms)` in CSS. **Critical:** Parent (`variants`) and all children must live in the same Client Component tree — Framer variants don't propagate across Server/Client boundaries

---

## GSAP vs Framer Motion

Choose the right tool for the right job:

- Use **Framer Motion** for all UI interactions (cards, modals, lists, buttons) — it integrates natively with React's component lifecycle
- Use **GSAP/ScrollTrigger** only for full-page scrolltelling or ThreeJS/WebGL canvas sequences — GSAP operates outside React's render cycle and is better suited for timeline-based cinema-style sequences
- Never mix both in the same component tree — GSAP scenes must live in isolated `useEffect` blocks with strict cleanup (`return () => { timeline.kill() }`)

---

## Z-Index Management

Never use arbitrary z-index values (`z-[9999]`). Reserve z-indexes strictly for systemic contexts:

- Sticky Navbars
- Modals
- Overlays

---

## Autonomous Demo Library (MOTION_INTENSITY ≥ 7)

When `MOTION_INTENSITY ≥ 7`, use these high-impact self-running animation patterns for feature sections. Each card in a bento grid should have its **OWN** autonomous loop — no two cards animate identically.

### 1. Ghost Cursor / Autonomous Product Demo

A simulated SVG mouse cursor (glowing soft-white pointer) moves autonomously across the UI, clicking and hovering over components in an infinite loop — the product demos itself.

- **Cursor:** `position: fixed` SVG pointer with subtle drop shadow, spring-physics movement (`stiffness: 80, damping: 18`) — never linear
- **Loop:** Timed `useEffect` sequence → move → pause → click → trigger state change → wait → move to next target
- **Best for:** Feature showcases, interactive dashboards, IDE/tool demos
- **Rule:** The cursor must trigger REAL component state changes (tabs switching, dropdowns opening, values changing) — not just cosmetic movement. Fake movement without consequence feels broken.

### 2. Type-Delete-Retype (Code/Config Animation)

A code block, terminal, or config editor shows content being deleted character by character then retyped in a different language/format as the ghost cursor "clicks" language tabs.

- **Variants:** IDE tab switching (TypeScript → Python → Go), config switching (YAML → JSON → TOML), API request preview (REST → GraphQL)
- **Font:** JetBrains Mono or Geist Mono — never regular sans-serif for code (it breaks the visual metaphor of a real editor)
- **Implementation:** Custom typewriter hook with `deletePhase` + `typePhase` + `pausePhase` states
- **Best for:** Backend/API card, developer tool pages, CLI showcases

### 3. Algorithm Step Visualizer

Data structures animate step by step — a glowing highlight box sweeps an array left-to-right, binary tree nodes activate, sorting bars race, graph edges light up.

- When target is found: highlight turns green, label changes ("Searching…" → "Found!")
- **Implementation:** `useInterval` stepping through indices, Framer Motion `layoutId` for smooth highlight box repositioning
- **Best for:** Data infrastructure, developer tools, search/AI product showcases

### 4. Live Property Editor ↔ Preview

A code/CSS/JSON editor on one side autonomously types out property values. A live preview component on the other side morphs in real time (border-radius, color, box-shadow, size changing smoothly).

- Ghost cursor hovers each property value → triggers typewriter → preview component animates to match
- **Implementation:** `useMotionValue` + `useTransform` on the preview element, typewriter on editor side
- **Best for:** Design tools, no-code builders, customization-heavy platforms

### 5. Autonomous Data Dashboard

A KPI counter, bar chart, or revenue metric updates automatically as a simulated dropdown selector cycles through time periods. Numbers spring-ease to new values. Chart bars animate height smoothly.

- **Counter:** `useSpring` on a motion value, not `useState` (avoids re-renders — motion values update outside React's render cycle)
- **Chart bars:** `animate={{ height: newValue }}` with spring easing
- **Best for:** Analytics, fintech, business intelligence, SaaS metrics

### 6. Morphing Hero Metric

A single large metric ("$2.4M ARR") continuously cross-fades between different KPIs with a number scramble transition — the sense of a live, breathing dashboard.

- **Implementation:** `AnimatePresence` for exit/enter + custom character scramble hook (`Math.random()` cycling through chars before settling)
- **Best for:** Hero sections of data products, growth tools, finance platforms

---

## Autonomous Animation Rules

These rules are non-negotiable for all autonomous animation patterns:

- Always loop with `repeat: Infinity` and a natural **2–4s rest** between cycles — continuous motion without breaks feels frantic and tiring
- Cursor spring: `type: "spring", stiffness: 80, damping: 18` — never `ease` or `linear` (spring physics simulates real cursor inertia)
- Isolate every autonomous loop in its own `React.memo` Client Component — prevents re-renders from propagating to parent layout components
- Respect `prefers-reduced-motion`: wrap all autonomous animations in `@media (prefers-reduced-motion: no-preference)` — users with vestibular disorders can experience nausea from persistent motion
- Autonomous animations must feel like a real user is interacting — add micro-pauses, hesitation, natural rhythm (not robotic linear timing)
