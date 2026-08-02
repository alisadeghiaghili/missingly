# Copywriting Reference

## Do's ✅

- Be concise: short, clear phrases focused on value
- Use numbers instead of words ("3 steps" not "three steps")
- Choose first 2 words carefully for each headline/CTA — they're all a scanner reads before deciding to continue
- Highlight operational benefits (time saved, cost reduction, clarity)
- Include user proof (testimonials, logos, concrete figures, use cases)
- Use action-oriented, specific CTAs
- Always add `:hover`, `:focus-visible`, and `:active` states on CTAs — hover alone is insufficient for keyboard and touch users
- Use organic, realistic data — messy numbers feel more trustworthy because real metrics are never round (`47.2%`, `+1 (312) 847-1928`, `$2.4M ARR`)

## Don'ts ❌

- NO generic/vague CTAs ("Learn more", "Click here") — they give zero information about what happens next
- NO double negatives
- NO user-blaming tone
- NO excessive text that loses user attention
- NO Lorem Ipsum if possible; use realistic SaaS/engineering copy
- NO AI filler words: "Elevate", "Seamless", "Unleash", "Next-Gen" — use concrete verbs that describe what actually happens
- NO generic placeholder names ("John Doe", "Sarah Chan") — invent believable, specific names that feel like real people
- NO startup slop brand names ("Acme", "Nexus", "SmartFlow") — create contextual, premium names that match the product domain
- NO predictable round numbers (`99.99%`, `50%`) for fake metrics — they instantly signal "fabricated" to the reader
- NO emojis anywhere in production code, markup, or copy — exception: preset-level overrides (e.g., `notion` preset allows structural emoji callouts when explicitly defined in its `DESIGN.md`)

## Good CTA Examples

- "Start free trial"
- "See demo"
- "Begin audit"
- "Discover report"
- "Deploy in 5 minutes"
- "Get started — it's free"

## Numbers & Data

Use organic, non-round numbers for all statistics and metrics:

| ❌ Avoid | ✅ Prefer |
|----------|----------|
| `50%` | `47.2%` |
| `99.99%` | `99.7%` |
| `1,000` | `1,247` |
| `$1M` | `$1.2M` |
| `555-0123` | `+1 (312) 847-1928` |

## Naming Conventions

### People Names
Invent believable, contextual names. Avoid:
- "John Doe", "Jane Smith" (placeholder classics)
- "Sarah Chan", "Alex Kim" (AI-generation clichés)

### Brand Names
Create premium, domain-specific names. Avoid:
- "Acme", "Nexus", "SmartFlow", "TechVault" (startup slop)
- Anything ending in "-ify", "-ly", "-io" by default

---

## Tone by Mood

Copywriting register must match the mood selected by the Style Auto-Router. Same product, different moods = different copy voice.

### `dark-saas`
- Sparse, precise, operator-facing. No marketing fluff in the hero.
- H1 pattern: **[Verb] [specific capability]** — "Ship without the overhead", "Observe everything. Fix what matters."
- Body: Short, declarative. "Built for teams that can't afford downtime."
- CTA: Specific action — "Open your workspace", "Deploy in 90 seconds"

### `dashboard-pro`
- Functional, credibility-first, data-backed. No adjectives that can't be measured.
- H1 pattern: **[Outcome] for [specific role]** — "Revenue intelligence for growth teams"
- Body: Lead with the metric or outcome. "47% faster pipeline reviews."
- CTA: Action + benefit — "See your data live", "Connect your stack"

### `editorial-premium`
- Authoritative, long-form capable, brand-voiced. Can afford literary phrasing.
- H1 pattern: **Striking single claim or question** — "What if good design was the default?"
- Body: One idea per paragraph. Short sentences followed by a longer one for rhythm.
- CTA: Understated — "Read the manifesto", "Explore the collection"

### `warm-startup`
- Human, service-oriented, approachable. Uses "you" and "your team" freely.
- H1 pattern: **[Relatable pain solved]** — "Your team's second brain, without the setup."
- Body: Acknowledge the problem before the solution. "Most tools add complexity. We remove it."
- CTA: Friendly + low-friction — "Try it free", "Get started today — no card needed"

### `product-cinematic`
- Minimal copy. The product IS the message. Short, mythic phrases.
- H1 pattern: **1-4 words, product-as-subject** — "Engineered for what's next", "The new standard"
- Body: 1 sentence maximum. Let the visual carry the weight.
- CTA: Authoritative — "Reserve yours", "See it in motion"

### `swiss-precision`
- Institutional, rational, fact-based. No metaphors, no emotion words.
- H1 pattern: **Descriptive, not persuasive** — "A systematic approach to design infrastructure"
- Body: Factual, structured. Bullet-point capable.
- CTA: Neutral + specific — "View documentation", "Download the spec"

### `soft-consumer`
- Warm, inclusive, benefit-led. Uses casual contractions freely.
- H1 pattern: **[Feeling or aspiration]** — "Learn anything, at your pace", "Your creative space, finally organized"
- Body: Upbeat, short sentences. Focus on how it feels to use, not what it does.
- CTA: Encouraging — "Start creating", "Join 11,240 learners"
