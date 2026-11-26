document.addEventListener('DOMContentLoaded', function () {
  const btn = document.querySelector('.js-theme');
  if (!btn) return;

  // Clone to strip Shibuya's existing click handler
  const newBtn = btn.cloneNode(true);
  btn.parentNode.replaceChild(newBtn, btn);

  function setMode(mode) {
    const html = document.documentElement;
    html.setAttribute('data-color-mode', mode);
    html.classList.remove('light', 'dark');
    html.classList.add(mode);
    localStorage._theme = mode;
  }

  // Initialise: respect previous choice, else follow system preference
  let current;
  if (localStorage._theme === 'dark' || localStorage._theme === 'light') {
    current = localStorage._theme;
  } else {
    current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  setMode(current);

  newBtn.addEventListener('click', function () {
    current = current === 'light' ? 'dark' : 'light';
    setMode(current);
  });
});

// Right sidebar on post listing pages and post pages. On listing pages each
// link opens its post and names the post's section on the page in
// data-section; on post pages links are #anchors. Shibuya highlights the
// first section that contains the top of the viewport, which is always the
// outer section when sections are nested (so subsections never light up), and
// nothing at all while the page is above the first section. Instead, highlight
// the last TOC section whose top has scrolled past the offset (the deepest one
// for nested sections), the first one above it and the last one at the bottom.
document.addEventListener('DOMContentLoaded', function () {
  const items = document.querySelectorAll('.post-toc li');
  if (!items.length) return;

  const toc = document.querySelector('.post-toc');
  function linkFor(section) {
    const id = CSS.escape(section.id);
    return toc.querySelector('a[data-section="' + id + '"], a[href="#' + id + '"]');
  }
  const sections = Array.from(document.querySelectorAll('.yue > section section[id]')).filter(function (section) {
    return linkFor(section);
  });
  if (!sections.length) return;

  const offset = parseFloat(getComputedStyle(sections[0]).scrollMarginTop) || 80;

  function highlight() {
    const atBottom = window.innerHeight + Math.round(window.scrollY) >= document.body.offsetHeight;
    let current = sections[0];
    if (atBottom) {
      current = sections[sections.length - 1];
    } else {
      sections.forEach(function (section) {
        if (section.getBoundingClientRect().top <= offset + 1) current = section;
      });
    }
    items.forEach(function (li) { li.classList.remove('active'); });
    linkFor(current).parentNode.classList.add('active');
  }

  highlight();
  window.addEventListener('scroll', highlight);

  // Jump straight to the clicked section. Smooth scrolling would pass through
  // the sections in between, briefly highlighting each of them on the way.
  // Links that open a post (listing pages) navigate normally.
  document.querySelectorAll('.post-toc li a[href^="#"]').forEach(function (link) {
    link.addEventListener('click', function (event) {
      const target = document.getElementById(link.getAttribute('href').slice(1));
      if (!target) return;
      event.preventDefault();
      target.scrollIntoView({ behavior: 'instant', block: 'start' });
      history.pushState(null, '', link.getAttribute('href'));
      items.forEach(function (li) { li.classList.remove('active'); });
      link.parentNode.classList.add('active');
    });
  });
});