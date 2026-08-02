# Slash Commands Reference

## `/build <page-type> <description>`

Generate a complete page from scratch.

**Page types:** `landing`, `dashboard`, `pricing`, `auth`, `settings`, `profile`, `blog`, `docs`, `changelog`

**Examples:**
```bash
/build landing AI-powered code review tool for engineering teams
/build dashboard Analytics dashboard for an e-commerce SaaS
/build pricing Three-tier pricing for a developer tool
/build auth Login + signup page for a fintech app
```

**What happens:**
1. Loads design-system, ux-audit, page-structure, and motion-patterns references
2. Runs Style Auto-Router — detects mood from brief, applies dial overrides
3. Runs **Evidence-Based UX Gate** — define task goal, first-screen clarity, visual complexity budget, accessibility constraints, and domain trust/conversion requirements
4. Loads style-recipes for the detected mood (components, layout, motion direction)
5. Selects tech stack based on complexity (React/Next.js or static HTML)
6. Runs Output Algorithm (12-step sequence before writing any code)
7. Generates all mandatory sections (minimum 5)
8. Applies motion patterns based on MOTION_INTENSITY
9. Runs pre-flight checklist
10. Runs Self-Check (5 anti-slop criteria) — if 2+ fail, fixes and re-checks before delivering
11. Invokes image-generator as final step after Self-Check passes

**Flags:**
- Add `--static` to force HTML/CSS/JS output
- Add `--react` to force React/Next.js output
- Add `--minimal` to skip non-essential sections

---

## `/polish`

Final pre-ship quality pass on existing code. Non-destructive.
Applies the **Evidence-Based UX Gate** when changes affect page UX (clarity, accessibility, trust, or conversion cues).

**What it checks and fixes:**
- Contrast ratios (adjusts colors to meet WCAG AA)
- Spacing consistency (normalizes to 4px grid)
- Copy quality (replaces generic CTAs, lorem ipsum)
- Micro-interactions (adds missing hover/focus states)
- Tactile feedback (adds active states: `scale-[0.98]`, pressed effects)
- Border consistency
- Typography hierarchy

**Example:**
```bash
/polish
/polish src/components/Hero.tsx
```

---

## `/audit`

WCAG 2.2 AA + anti-slop report. **No code changes.** Output only.

**Output format:** `file:line` references with severity (Critical/Warning/Info).

**What it checks:**
- WCAG AA contrast ratios
- Semantic HTML usage
- Focus indicators and keyboard navigation
- Loading/empty/error states
- Anti-slop patterns (centered hero, neon accents, default shadcn, etc.)
- Missing ARIA attributes

**Example:**
```bash
/audit
/audit src/app/page.tsx
```

---

## `/critique`

UX design review written as a principal designer. **No code changes.**

**Output:** Free-form review covering:
- Nielsen 10 heuristics analysis
- User flow assessment
- Cognitive load evaluation
- Visual hierarchy clarity
- Information architecture
- Interaction design quality
- Specific recommendations with priority

**Example:**
```bash
/critique
/critique src/components/Dashboard.tsx
```

---

## `/animate [intensity]`

Add motion patterns to existing code.
Applies only the accessibility/motion subset of the **Evidence-Based UX Gate** (reduced motion, readability, orientation, and no motion-only critical information).

**What it adds:**
- Scroll-triggered reveals (fade-in, slide-in, scale-in with stagger)
- Hover effects on interactive elements
- Micro-interactions (button press, input focus, card hover)
- Perpetual animations when intensity > 5 (pulse, shimmer, float) — only for `prefers-reduced-motion: no-preference`
- Spring physics on interactive elements when intensity > 6
- Autonomous demos (ghost cursor, type-delete-retype) when intensity > 7 — only for `prefers-reduced-motion: no-preference`

**Always respects `prefers-reduced-motion: reduce`.**

**Example:**
```bash
/animate
/animate 8
/animate 3
```

---

## `/imagify [mood]`

Run the Gemini image generation pipeline.
Skips the full **Evidence-Based UX Gate** by default; still enforce comprehension, trust, and accessibility checks when generated imagery affects meaning.

**Mood keywords:** `dark moody`, `clean bright`, `cinematic`, `minimal`, `warm`, `cold`, `dramatic`

**What it does:**
1. Scans the page for image zones (hero backgrounds, feature illustrations, logos, avatars)
2. Crafts cinematic prompts for each zone
3. With `GEMINI_API_KEY`: generates via `gemini-3.1-flash-image-preview`
4. Without key: falls back to curated `picsum.photos` placeholders

**Anti-slop / meaning-safety checks (still enforced when imagery affects meaning):**
- No ambiguous, misleading, or meaning-changing imagery; keep visuals consistent with page content and user task
- Do not fabricate real people, brands, testimonials, product states, or results; mark synthetic/provenance where relevant
- Preserve information hierarchy, readable overlays, alt text, and accessibility for meaningful image zones
- Reference the canonical **Evidence-Based Anti-Slop Checks** before accepting imagery that affects comprehension or trust

**Example:**
```bash
/imagify
/imagify dark moody
/imagify clean bright
```

---

## `/dials variance=N motion=N density=N`

Adjust design dials mid-session. Accepts partial sets.

**Examples:**
```bash
/dials variance=6 motion=8 density=3
/dials motion=2
/dials density=9
```

**Effects:**
- `variance=1-3`: Symmetric, grid-aligned, centered layouts
- `variance=7-10`: Asymmetric, creative, artsy layouts
- `motion=1-3`: Minimal transitions, no scroll animations
- `motion=7-10`: Cinematic springs, autonomous demos, perpetual animations
- `density=1-3`: Generous whitespace, art gallery feel
- `density=7-10`: Cockpit dense, `divide-y` instead of cards, monospace numbers

---

## `/variant <preset-name>`

Swap brand preset. Re-themes the current output.
Applies the **Evidence-Based UX Gate** when preset changes materially rewrite page UX: above-the-fold hierarchy or CTA semantics change, layout reflows roughly >20% of the first viewport, accessibility-critical structure changes, contrast drops below WCAG AA, or motion intensity/banned-pattern changes affect comprehension. Example: moving the primary CTA into a new hero composition triggers the gate; swapping accent hue while preserving contrast and layout does not.

**Available presets:**

| Name | One-line |
|------|----------|
| `vercel` | Black/white, Geist, zero accent |
| `linear` | Near-black, indigo accent, dense |
| `stripe` | Navy, prismatic gradients, generous |
| `raycast` | Dark chrome, multi-hue, rounded-XL |
| `superhuman` | Purple glow, monospace metadata, fast |
| `notion` | Off-white, serif display, editorial |
| `vs-code` | Editor-dark, syntax colors, monospace |

**What changes:** Colors, fonts, spacing, border-radius, motion intensity, banned patterns.

**Example:**
```bash
/variant linear
/variant raycast
```
