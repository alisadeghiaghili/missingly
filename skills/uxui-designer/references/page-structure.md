# Page Structure Reference

## Table des matières
- [Mandatory Sections (Minimum 5)](#mandatory-sections-minimum-5)
- [Global Layout Specs](#global-layout-specs)

---

## Mandatory Sections (Minimum 5)

### 1. Header/Navigation
- Desktop: Visible nav + primary CTA
- Mobile: Hamburger menu or single clear CTA button
- Sticky with glass effect on scroll
- Transition when changing to glass effect
- Always check and add the script for opening the hamburger menu — a non-functional mobile menu is the #1 missed detail in AI-generated pages

### 2. Hero Section
- XXL responsive title
- Badge above title (stat, insight, product label)
- Subtitle with clear, concrete benefit
- 2 CTAs (primary + secondary) well differentiated
- Social proof: avatars, logos, stats
- Subtle background (geometric shapes, discrete patterns, NO big gradients or glows)
- Scroll invitation indicator (arrow, line, label)
- Custom SVG illustration (NOT a mockup or preview) OR WebGL background or ASCII
- Handwritten text annotation and arrow pointing to the CTA
- **Asymmetric layout** — no centered text blocks (see `design-system.md` § Layout rules)

### 3. Social Proof Section
- Client testimonials with ratings
- Company logos (marquee or grid)
- Concrete statistics — use organic numbers, not round ones (`47.2%`, not `50%`)
- Use cases

### 4. Features/Solution Section
- Bento grid or creative layout
- Icon + title + description for each feature
- SVG illustrations for each card (consistent style across all cards)
- Hover animations on cards

### 5. Pricing Section (if applicable)
- Clear plan comparison
- Popular plan highlighted
- Feature lists with checkmarks
- CTA for each plan

### 6. FAQ Section
- Accordion with smooth animations
- Clear questions and answers

### 7. Final CTA Section
- Compelling headline
- Clear value proposition
- Primary action button
- Trust elements

### 8. Footer
- Logo and tagline
- Link columns (Product, Resources, Company)
- Social media icons
- Legal links
- Copyright

---

## Global Layout Specs

- Mobile-first with Tailwind breakpoints (320px, 768px, 1200px+)
- Semantic HTML5: `<header>`, `<main>`, `<section>`, `<article>`, `<aside>`, `<footer>`
- 4px increment spacing system
- Section padding: `py-24` / `py-32`
- Container: `max-w-7xl mx-auto px-6`
- No horizontal scroll, no fixed pixel values for main layouts
- Border-radius rule: parent radius = child radius + inner spacing

---

## Output Algorithm

Run this decision sequence **before generating the first component**. Each step must be resolved before moving to the next.

```
1. PROMISE      — What is the single thing this page must make the user believe or do?
2. MOOD         — Which Style Auto-Router mood fits? Lock it. Set the dial overrides.
3. HERO SYSTEM  — Which hero type? (split | offset | framed | cinematic | command | metrics)
4. NAV POSTURE  — Inline, sticky glass, floating island, or editorial index?
5. SECTION SEQ  — List all sections in order. Min 5. Each must answer a different user question.
6. PROOF ENTRY  — Where does trust appear? Must arrive before section 4.
7. IMAGE ENTRY  — Where is the first real image or diagram? Must appear in hero or first 2 sections.
8. CARDS?       — Are cards justified, or can dividers/spacing do the same job without chrome?
9. CONVERSION   — Which section is the primary conversion moment? Has explicit CTA copy.
10. FOOTER TONE — Does the footer continue the visual language or break it?
11. MOTION MAP  — Which sections get entrance motion? Which interactive elements need states?
12. IMAGE-GEN   — Invoke image-generator.md as the final step.
```

Do not skip steps. "I'll figure it out as I build" is how generic pages happen.

---

## Hero Checksum Rules

Before writing hero markup, verify all 7 conditions:

- [ ] **Focal point** — One dominant visual element (headline, image, diagram) — not two competing
- [ ] **Fixed envelope** — Hero reads as a complete scene on a 1280px screen. Does not bleed accidentally into the next section
- [ ] **Support element** — One secondary detail (proof stat, badge, sub-headline) that supports the focal point without competing
- [ ] **CTA hierarchy** — Include primary + secondary CTA with specific copy, and keep the secondary clearly lower-contrast
- [ ] **Visual anchor** — If no image: the composition must use typography scale, spatial cropping, or a diagram. An H1 on a flat background is not enough
- [ ] **Mobile order** — The hero collapses to a clean single-column layout on < 768px without losing identity
- [ ] **Nav relationship** — The nav and hero look like they belong to the same page, not like two separate templates stacked

If any box is unchecked, fix it before generating the next section.

---

## Section Rhythm Rules

These rules prevent the most common page-level pacing failures:

**Never repeat the same section grammar twice in a row.** If section 2 uses a 3-column card grid, section 3 must use a different structure (split layout, image plate, ruled list, timeline, or open prose + visual).

**Alternate density.** A visually dense section must be followed by a more spacious one. Pack-pack-pack reads as noise. Pack-breathe-pack reads as rhythm.

**Every section must answer a different user question:**
- Hero: *"What is this and why do I care?"*
- Proof: *"Can I trust this?"*
- Features: *"What does it actually do?"*
- Process: *"How does it work?"*
- Pricing: *"Is this for me and what does it cost?"*
- FAQ: *"What am I still unsure about?"*
- CTA: *"What do I do next?"*

If two adjacent sections answer the same question, merge or cut one.

**Introduce interaction before the page feels static.** By section 3, at least one element should have responded to user input (hover, scroll reveal, or interactive component).

**A footer that breaks the visual language is a missed opportunity.** The footer should feel like a quieter version of the page — same font, same color temperature, a structural echo of the nav.
