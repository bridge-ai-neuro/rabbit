// Native document scrolling, a live section index, and reversible edge fades.
(() => {
  const root = document.documentElement;
  const topbar = document.querySelector('nav.top');
  const mobileIndex = document.querySelector('.mobile-index');
  const picker = document.getElementById('sectionPicker');
  const progressBar = document.querySelector('.page-progress');
  const links = [...document.querySelectorAll('[data-reading-link]')];
  const ids = [...new Set(links.map(link => link.hash.slice(1)))];
  const sections = ids.map(id => document.getElementById(id)).filter(Boolean);
  const scenes = [...document.querySelectorAll('[data-scroll-scene]')].map(element => ({ element, shift: 0 }));
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0;
  let previousActive;

  const clamp = value => Math.min(1, Math.max(0, value));
  // Ease the entrance/exit without delaying or taking over the browser's scroll.
  const ease = value => value * value * (3 - 2 * value);
  function update() {
    frame = 0;
    const height = innerHeight;
    const topHeight = topbar?.offsetHeight || 0;
    const indexHeight = mobileIndex?.offsetHeight || 0;
    const readingTop = topHeight + indexHeight + 24;
    const maximum = Math.max(0, document.documentElement.scrollHeight - height);
    const progress = maximum ? clamp(scrollY / maximum) : 1;
    const focus = document.activeElement;
    // Read layout before writing styles. Undo our previous translation in measurements.
    const positions = sections.map(element => ({ element, top: element.getBoundingClientRect().top }));
    const animation = scenes.map(scene => {
      const box = scene.element.getBoundingClientRect();
      const focused = focus !== document.body && scene.element.contains(focus);
      const shift = focused ? 0 : scene.shift;
      const top = box.top - shift;
      const bottom = box.bottom - shift;
      const entrance = ease(clamp((height - top) / Math.max(1, height * 0.24)));
      const exit = ease(clamp((bottom - readingTop) / Math.max(1, height * 0.18)));
      // At the end of the document the last figure remains fully readable.
      const opacity = focused || reducedMotion.matches || progress > 0.999 ? 1 : Math.min(entrance, exit);
      const nextShift = focused || reducedMotion.matches || progress > 0.999 ? 0 : (1 - entrance) * 18 - (1 - exit) * 10;
      return { scene, opacity, shift: nextShift };
    });
    root.style.setProperty('--topbar-height', `${topHeight}px`);
    root.style.setProperty('--reading-offset', `${readingTop}px`);
    root.dataset.scrollMotion = reducedMotion.matches ? 'reduced' : 'enabled';
    for (const { scene, opacity, shift } of animation) {
      scene.element.style.setProperty('--scene-opacity', opacity.toFixed(3));
      scene.element.style.setProperty('--scene-shift', `${shift.toFixed(2)}px`);
      scene.shift = shift;
    }
    const marker = readingTop + Math.min(150, height * 0.18);
    let active = positions[0]?.element.id;
    for (const { element, top } of positions) if (top <= marker) active = element.id;
    if (progress > 0.999) active = positions.at(-1)?.element.id;
    if (active !== previousActive) {
      for (const link of links) {
        if (link.hash === `#${active}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
      const selected = links.find(link => link.hash === `#${active}`);
      document.querySelectorAll('[data-current-section]').forEach(label => { label.textContent = selected?.textContent || ''; });
      // Keep the active entry visible when the full document's index needs its own scroll.
      const sidebar = selected?.closest('.reading-nav');
      if (sidebar?.offsetWidth) {
        const item = selected.getBoundingClientRect();
        const bounds = sidebar.getBoundingClientRect();
        if (item.bottom > bounds.bottom - 12) sidebar.scrollTop += item.bottom - bounds.bottom + 12;
        else if (item.top < bounds.top + 12) sidebar.scrollTop -= bounds.top + 12 - item.top;
      }
      previousActive = active;
    }
    if (progressBar) { progressBar.hidden = false; progressBar.value = progress; }
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(update); }
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('resize', schedule, { passive: true });
  addEventListener('hashchange', schedule);
  addEventListener('pageshow', schedule);
  document.addEventListener('focusin', schedule);
  document.addEventListener('focusout', schedule);
  document.addEventListener('load', schedule, true);
  document.addEventListener('toggle', schedule, true);
  reducedMotion.addEventListener('change', schedule);
  document.fonts.ready.then(schedule);
  // Native anchors preserve browser history, modified clicks, and no-JS navigation.
  picker?.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { picker.open = false; }));
  document.addEventListener('click', event => { if (picker && !picker.contains(event.target)) picker.open = false; });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && picker?.open) { picker.open = false; picker.querySelector('summary').focus(); }
  });
  update();
})();
