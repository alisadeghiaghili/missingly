# UX Audit Reference

## Evidence-Based UX Layer

Use this layer before visual taste decisions. The goal is not to make every interface boring or academic; it is to make premium aesthetics verifiable against research-backed usability, perception, accessibility, and conversion principles.

### Evidence Hierarchy

When evidence conflicts, apply this priority order:

1. **User task and business goal** — the interface must help the target user complete the intended action.
2. **Accessibility and safety** — WCAG 2.2 AA, keyboard access, readable contrast, reduced motion, focus states.
3. **Usability heuristics** — Nielsen/Molich principles: feedback, consistency, error prevention, recognition over recall, user control.
4. **Research-backed perception rules** — first impression, visual complexity, prototypicality, trust, cognitive load.
5. **Domain guidelines** — Baymard for e-commerce/conversion, platform design systems for component behavior.
6. **Modern inspiration references** — Supahero, Linear, Vercel, Stripe, Raycast, etc.
7. **Aesthetic experimentation** — only after the above are satisfied.

Never let novelty, motion, or visual drama override clarity, accessibility, task completion, or trust.

### Research-Backed Visual Perception Rules

Based on HCI research about website first impressions and aesthetic judgments:

| Finding | Practical rule for generation/audit |
|---|---|
| Users form durable aesthetic impressions as fast as 50ms (Lindgaard et al., 2006; Tuch et al., 2012). | The first viewport must communicate product category, value, and primary action immediately. No ambiguous abstract hero without context. |
| Visual complexity is a strong predictor of first-impression appeal. | Keep the above-the-fold area low-to-medium complexity: one dominant visual anchor, one primary CTA, limited competing decorations. |
| Prototypicality improves quick comprehension and perceived appeal. | Keep category conventions recognizable: SaaS needs clear nav/CTA/proof; dashboard needs data hierarchy; e-commerce needs product, price, trust, delivery/return cues. |
| Colorfulness matters, but usually less than perceived complexity. | Do not solve weak design by adding more gradients/colors. Improve hierarchy, spacing, alignment, and information grouping first. |
| Visual appeal influences perceived usability and trustworthiness. | Treat polish as functional: spacing, typography, alignment, and contrast affect credibility, not just beauty. |

### First-Screen Quality Gate

Before shipping a generated landing page or homepage, answer all of these:

- Can a new user understand **what this is** in 5 seconds?
- Is there exactly **one dominant focal point** above the fold?
- Is the primary CTA visually and semantically obvious?
- Does the structure look familiar enough for the category while still having an authored twist?
- Is above-the-fold visual complexity controlled, with detail increasing progressively below the fold?
- Would the page still make sense if all decorative gradients/canvas effects were removed?

If two or more answers are "no", simplify the hero and strengthen information hierarchy before adding more visuals.

### Source-Informed Design Rules

- **NN/g heuristics:** Always provide feedback for interactive states, user control for reversible/risky actions, consistent component meanings, and recognition over recall.
- **WCAG 2.2:** Include semantic structure, keyboard operability, visible focus, contrast, text alternatives, reduced motion, and robust name/role/value for custom controls.
- **Baymard-style conversion UX:** For commerce/pricing/checkout flows, show decision-critical info near the action: price, plan differences, delivery/return/cancel terms, trust/support signals, and error-preventing form guidance.
- **Design systems:** Use platform conventions for component behavior, spacing, focus, labels, disabled states, and error messaging. Visual styling can be custom; behavior should remain predictable.

### Evidence-Based Anti-Slop Checks

- Do not increase perceived quality by adding more effects; reduce noise and improve hierarchy first.
- Do not make a hero highly experimental unless the category is still recognizable.
- Do not hide essential information inside hover-only interactions.
- Do not animate elements users need to read continuously.
- Do not use decorative data/dashboard cards unless the numbers support the product story.
- Do not use trust badges, metrics, or testimonials as filler; they must answer a user objection.

### Minimal Citation Notes for Maintainers

- Lindgaard et al. (2006): rapid first impressions of website attractiveness.
- Reinecke et al. (CHI 2013): visual complexity and colorfulness predict first-impression aesthetics; complexity is especially important.
- Tuch et al. (IJHCS 2012): visual complexity and prototypicality affect first impressions within milliseconds; low complexity + high prototypicality tends to rate highly.
- Nielsen/Molich/Nielsen Norman Group: 10 usability heuristics for interface evaluation.
- W3C WCAG 2.2: perceivable, operable, understandable, robust accessibility model.
- Baymard Institute: large-scale e-commerce UX research and conversion guidelines.

---

## Nielsen's 10 Usability Heuristics

### 1. Visibility of System Status
The system should always keep users informed about what is going on.
- **Check:** Loading indicators for async operations? Progress bars for multi-step flows? Active states on navigation?
- **Fail example:** Button clicked, nothing happens for 2 seconds, no loading state.

### 2. Match Between System and Real World
Use language and concepts familiar to the user.
- **Check:** Labels use user vocabulary (not dev jargon)? Logical grouping? Familiar icons?
- **Fail example:** "Instantiate a new entity" instead of "Create project".

### 3. User Control and Freedom
Users need a clear "emergency exit" — undo, cancel, go back.
- **Check:** Cancel buttons on modals/forms? Undo for destructive actions? Back navigation?
- **Fail example:** Deleting an item with no confirmation and no undo.

### 4. Consistency and Standards
Same action = same result everywhere. Follow platform conventions.
- **Check:** Consistent button styles? Same icon meaning throughout? Standard patterns (X to close, left arrow to go back)?
- **Fail example:** Primary CTA is green on one page, blue on another.

### 5. Error Prevention
Prevent errors before they happen.
- **Check:** Confirmation for destructive actions? Input constraints (type, min, max)? Disabled buttons when form is invalid?
- **Fail example:** Delete button with no confirmation dialog.

### 6. Recognition Rather Than Recall
Minimize memory load — make options visible.
- **Check:** Recent items shown? Autocomplete on search? Tooltips on icons? Breadcrumbs?
- **Fail example:** Icon-only navigation with no labels or tooltips.

### 7. Flexibility and Efficiency of Use
Accelerators for experts, simplicity for novices.
- **Check:** Keyboard shortcuts? Bulk actions? Customizable dashboards? Command palette?
- **Fail example:** No way to select multiple items for batch operations.

### 8. Aesthetic and Minimalist Design
Every element should serve a purpose.
- **Check:** No decorative elements that distract? Content hierarchy clear? Whitespace used intentionally?
- **Fail example:** Three different card styles on the same page for the same type of data.

### 9. Help Users Recognize, Diagnose, and Recover from Errors
Error messages should be clear, specific, and suggest solutions.
- **Check:** Error messages explain what went wrong? Suggest how to fix? Inline errors near the field?
- **Fail example:** "Error 500" with no explanation or action.

### 10. Help and Documentation
Easy to search, focused on the task, concrete steps.
- **Check:** Tooltips on complex fields? Onboarding for first use? Help section accessible?
- **Fail example:** 50-page documentation with no search and no task-based structure.

---

## WCAG 2.2 AA Checklist

### Perceivable

| Rule | Requirement | How to check |
|------|------------|-------------|
| 1.1.1 | Non-text content has text alternatives | All `<img>` have meaningful `alt`. Decorative images: `alt=""` + `aria-hidden="true"` |
| 1.3.1 | Info conveyed through structure | Semantic HTML (`<h1>`-`<h6>`, `<nav>`, `<main>`, `<aside>`, `<button>`) |
| 1.3.5 | Input purpose identified | `autocomplete` attributes on common fields (name, email, address) |
| 1.4.1 | Color not sole indicator | Don't rely on color alone for errors — add icon or text |
| 1.4.3 | Contrast minimum | **4.5:1** for normal text, **3:1** for large text (18px+ or 14px+ bold) |
| 1.4.4 | Resize text | Text resizable up to 200% without assistive technology, without loss of content or functionality |
| 1.4.11 | Non-text contrast | **3:1** for UI components and graphical objects |
| 1.4.12 | Text spacing | Content usable with line-height 1.5x, paragraph spacing 2x, letter-spacing 0.12em, word-spacing 0.16em |

### Operable

| Rule | Requirement | How to check |
|------|------------|-------------|
| 2.1.1 | Keyboard accessible | All functionality available via keyboard |
| 2.1.2 | No keyboard trap | Focus can always move away from a component |
| 2.4.3 | Focus order | Tab order follows logical reading sequence |
| 2.4.4 | Link purpose | Link text describes destination (not "click here") |
| 2.4.7 | Focus visible | All interactive elements have visible focus indicator (`ring-2`) |
| 2.4.11 | Focus not obscured | Focus indicator not hidden by sticky headers or overlays |
| 2.5.5 | Target size (AAA) | Interactive targets minimum **44x44px** — Level AAA, best practice |
| 2.5.8 | Target Size Minimum (AA) | Interactive targets minimum **24x24px**. Exceptions: spacing, equivalent control, inline text, user-agent-set, essential |

### Understandable

| Rule | Requirement | How to check |
|------|------------|-------------|
| 3.1.1 | Language of page | `<html lang="en">` (or appropriate language) |
| 3.2.1 | On focus | Focus alone doesn't trigger unexpected changes |
| 3.2.2 | On input | Changing an input doesn't automatically submit or navigate |
| 3.3.1 | Error identification | Errors described in text, not just color |
| 3.3.2 | Labels or instructions | All form fields have visible labels |
| 3.3.3 | Error suggestion | Error messages suggest how to fix the problem |
| 3.3.7 | Redundant entry | Don't ask for the same info twice in a flow |

### Robust

| Rule | Requirement | How to check |
|------|------------|-------------|
| 4.1.2 | Name, Role, Value | Custom components have proper ARIA attributes |
| 4.1.3 | Status messages | `aria-live="polite"` for dynamic content updates |

---

## Anti-Slop Patterns

These are the most common AI-generated UI failures. Flag every occurrence.

### Layout
- [ ] Centered hero when design variance is high
- [ ] Three equal-width cards in a row (use asymmetric grid)
- [ ] `h-screen` instead of `min-h-[100dvh]`
- [ ] Flexbox calc for multi-column (use CSS Grid)
- [ ] No mobile responsive design
- [ ] Horizontal scroll on mobile

### Color
- [ ] Neon/saturated accent colors
- [ ] Purple button glow (AI signature)
- [ ] Mixed warm/cool grays
- [ ] `#000000` pure black backgrounds
- [ ] Failed WCAG contrast (check every text/background combo)

### Typography
- [ ] Inter as "premium" font (use Geist, Satoshi, Cabinet Grotesk)
- [ ] No typographic scale (everything same size)
- [ ] Missing `max-w-[65ch]` on reading text
- [ ] Serif on dashboards
- [ ] Round numbers in statistics (`50%` instead of `47.2%`)

### Components
- [ ] Default shadcn/ui without customization
- [ ] Generic spinner instead of skeleton loader
- [ ] Missing loading/empty/error states
- [ ] No hover effects on interactive elements
- [ ] Emojis in production UI
- [ ] "Lorem ipsum" placeholder text
- [ ] Generic CTAs ("Learn more", "Click here")

### Animation
- [ ] Layout-triggering animations (`width`, `height`, `top`, `left`)
- [ ] Missing `prefers-reduced-motion` respect
- [ ] `useState` for continuous animations (use `useMotionValue`)
- [ ] No cleanup on `useEffect` animations
- [ ] Perpetual animations not isolated in `React.memo`

### Code
- [ ] Inline styles (except dynamic values)
- [ ] `!important` usage
- [ ] Missing semantic HTML
- [ ] No `aria-hidden` on decorative elements
- [ ] Unverified 3rd-party imports

---

## Audit Output Format

When running `/audit`, output in this format:

```md
## UX/UI Audit Report

### Summary
- WCAG AA violations: X
- Anti-slop patterns: X
- Missing states: X
- Nielsen heuristic concerns: X

### Critical (must fix)
- `file.tsx:42` — Contrast ratio 2.1:1 on body text (needs 4.5:1) [WCAG 1.4.3]
- `file.tsx:87` — No loading state for async data fetch [Nielsen #1]

### Warning (should fix)
- `file.tsx:15` — Generic centered hero layout [anti-slop]
- `file.tsx:103` — Default shadcn button, no customization [anti-slop]

### Info (nice to fix)
- `file.tsx:200` — Could use skeleton loader instead of spinner
- Consider adding keyboard shortcuts [Nielsen #7]
```
