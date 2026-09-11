// Shared static-site enhancement. Content and navigation also work without JS.
(() => {
  const root = document.documentElement;
  const themeButton = document.getElementById('theme-toggle');
  const themes = ['light', 'moderate', 'dark'];
  const names = { light: 'Light', moderate: 'Soft', dark: 'Dark' };
  const colors = { light: '#ffffff', moderate: '#f2f5fa', dark: '#171f2a' };
  function syncTheme() {
    const theme = root.dataset.theme || 'light';
    const next = themes[(themes.indexOf(theme) + 1) % themes.length];
    themeButton?.setAttribute('aria-label', `${names[theme]} theme. Switch to ${names[next]}.`);
    themeButton?.setAttribute('title', `Switch to ${names[next]}`);
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', colors[theme]);
  }
  themeButton?.addEventListener('click', () => {
    root.dataset.theme = themes[(themes.indexOf(root.dataset.theme) + 1) % themes.length];
    try { localStorage.setItem('rabbit-theme', root.dataset.theme); } catch {}
    syncTheme();
  });
  syncTheme();

  const menuButton = document.getElementById('navToggle');
  const menu = document.getElementById('nav-sections');
  const closeMenu = () => {
    menu?.classList.remove('open');
    menuButton?.setAttribute('aria-expanded', 'false');
    menuButton?.setAttribute('aria-label', 'Open navigation');
  };
  menuButton?.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  });
  menu?.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
  document.addEventListener('click', event => {
    if (!menu?.contains(event.target) && !menuButton?.contains(event.target)) closeMenu();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menu?.classList.contains('open')) { closeMenu(); menuButton.focus(); }
  });

  const video = document.querySelector('.hero-video');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  // preload=none and no HTML autoplay let motion/data preferences take effect
  // before the first video request. Native controls still work without JS.
  const holdVideo = () => { if (reducedMotion.matches || navigator.connection?.saveData) video?.pause(); };
  if (video) {
    const replay = document.querySelector('.video-replay');
    replay.hidden = false;
    replay.addEventListener('click', () => { video.currentTime = 0; video.play().catch(() => {}); });
    video.addEventListener('error', () => { document.querySelector('.video-error').hidden = false; });
    // Source failures don't always bubble to the video when all formats fail.
    const sources = [...video.querySelectorAll('source')];
    let failedSources = 0;
    sources.forEach(source => source.addEventListener('error', () => {
      if (++failedSources === sources.length) document.querySelector('.video-error').hidden = false;
    }));
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        if (!reducedMotion.matches && !navigator.connection?.saveData) video.play().catch(() => {});
        observer.disconnect();
      }
    });
    observer.observe(video);
  }
  reducedMotion.addEventListener('change', holdVideo);
  navigator.connection?.addEventListener?.('change', holdVideo);

  const copy = document.getElementById('copy-citation');
  if (copy) {
    copy.hidden = false;
    copy.addEventListener('click', async () => {
      const citation = document.getElementById('citation-text');
      const status = document.getElementById('citation-status');
      try {
        await navigator.clipboard.writeText(citation.textContent.trim() + '\n');
        status.textContent = 'BibTeX copied.';
      } catch {
        const range = document.createRange(); range.selectNodeContents(citation);
        const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
        citation.focus();
        status.textContent = 'Citation selected. Copy it with Ctrl+C or ⌘C, or download the .bib file.';
      }
    });
  }
})();
