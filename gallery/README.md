# ThinkFlow Gallery

Place each standalone HTML showcase in `gallery/sites`.

The gallery reads these optional tags from each file:

```html
<title>Project title</title>
<meta name="description" content="A short project description">
<meta name="thinkflow:category" content="Store">
<meta name="thinkflow:prompt" content="The prompt used to create this page">
```

Filenames become public slugs, so use short names such as `modern-store.html`.

For long prompts, add a UTF-8 text file with the same name and `.prompt.txt`
suffix, for example `modern-store.prompt.txt`. The sidecar prompt takes
priority over the optional `thinkflow:prompt` meta tag.
