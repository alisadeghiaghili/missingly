# Dashboard Design Reference

## Layout Grammar

### Bento 2.0 Grid
Dashboard page regions use asymmetric CSS Grid; metric-card rows can use equal-width columns when scannability is the goal.

```css
/* Typical SaaS dashboard grid */
grid-template-columns: 280px 1fr;        /* sidebar + main */
grid-template-columns: repeat(4, 1fr);   /* metric cards */
grid-template-rows: auto 1fr;            /* header + content */
```

### Sidebar Rules
- Width: `w-64` (256px) on desktop, collapsible to `w-16` (64px) icon-only
- Background: `bg-zinc-950` or one step darker than main content
- Border: `border-r border-white/5`
- Navigation items: `py-2 px-3 rounded-lg` with `hover:bg-white/5`
- Active state: `bg-white/10 text-white` (not just color change)
- Group labels: `text-xs text-zinc-500 uppercase tracking-wider font-medium mb-2`
- Bottom section: user avatar, settings, collapse toggle

### Top Bar
- Height: `h-14` / `h-16`
- Content: breadcrumb (left), search (center or left), actions (right)
- Border: `border-b border-white/5`
- Sticky: `sticky top-0 z-10 bg-zinc-950/80 backdrop-blur-sm`

## Metric Cards

### KPI Card Anatomy
```text
┌─────────────────────────┐
│ Label        ↑ +12.3%   │  ← label: text-sm text-zinc-500
│ $47,291.84              │  ← value: text-3xl font-semibold tabular-nums
│ ▁▂▃▅▆▇█▆▅▃▂▁           │  ← sparkline (optional)
└─────────────────────────┘
```

- Numbers: `font-variant-numeric: tabular-nums` (or `tabular-nums` Tailwind) — columns align
- Trend indicator: green `↑` / red `↓` with `text-sm`
- Percentage: organic numbers (`+12.3%`, not `+10%`)
- Sparkline: SVG path, `stroke-width: 1.5`, accent color at `opacity-0.5`

### Card Grid
```txt
grid grid-cols-2 lg:grid-cols-4 gap-4
```

## Tables

### Data Table Rules
- Header: `text-xs text-zinc-500 uppercase tracking-wider bg-zinc-900/50`
- Rows: `border-b border-white/5` — no alternating row colors (use hover instead)
- Hover: `hover:bg-white/[0.02]`
- Cell padding: `px-4 py-3`
- Numbers: right-aligned, `tabular-nums`
- Status badges: colored dot + text, not background color
- Empty state: centered illustration + "No data yet" + action CTA
- Pagination: bottom-right, simple prev/next with page count

### When `VISUAL_DENSITY > 7`
- Replace cards with `divide-y divide-white/5` list rows
- Tighter cell padding: `px-3 py-2`
- Smaller text: `text-sm` everywhere
- No card borders — use dividers only

## Charts

### Color System
Primary series: accent color at full opacity.
Additional series: accent at `opacity-0.6`, `0.4`, `0.2` — or use zinc shades.
Never more than 5 colors in a single chart.

### Chart Types
- **Line charts:** `stroke-width: 2`, no fill unless area chart. Dots on hover only.
- **Bar charts:** `rounded-t-sm`, gap between bars = 40% of bar width.
- **Donut/Pie:** Max 5 segments. Center: key metric or total.

### Chart Container
```txt
bg-zinc-900/30 border border-white/5 rounded-xl p-6
```
Title top-left. Legend inline or bottom. Time range selector top-right.

## States (Mandatory)

### Loading
- Skeleton loaders matching exact card/table shape
- Pulse animation: `animate-pulse bg-zinc-800 rounded`
- Never generic spinners

### Empty
- Centered in the container
- Muted icon (32-48px) + heading + description + CTA
- Example: "No transactions yet" → "Start by connecting your bank account" → [Connect account]

### Error
- Inline red banner: `bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg p-4`
- Retry button
- Never break the layout — error replaces the data area, not the whole page

## Responsive

- < 768px: sidebar collapses to bottom navigation or hamburger
- Metric grid: `grid-cols-2` on mobile (not single column — too much scroll)
- Tables: horizontal scroll wrapper on mobile (`overflow-x-auto`)
- Charts: simplified axis labels, reduced padding
