# Image Reliability Skill

- Do not leave broken image zones in generated HTML.
- Prefer CSS gradients, inline SVG, or generated-looking placeholder panels when a real image URL is not guaranteed.
- If an `<img>` is necessary, include meaningful `alt`, stable dimensions, and an `onerror` fallback.
- Avoid hotlinking random third-party images unless the user explicitly asks for a specific external asset.
- For product mockups, dashboards, and abstract visuals, build the visual with HTML/CSS/SVG so the page works offline.
- Never rely on images for critical text or instructions.
