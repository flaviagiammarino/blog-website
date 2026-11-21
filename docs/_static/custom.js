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

// Post listing pages: Shibuya only highlights a post once it reaches the top of
// the viewport, so keep the first post highlighted while the page is above it.
document.addEventListener('DOMContentLoaded', function () {
  const items = document.querySelectorAll('.post-toc li');
  const first = document.querySelector('.yue > section section[id]');
  if (!items.length || !first) return;

  const offset = parseFloat(getComputedStyle(first).scrollMarginTop) || 80;

  function highlightFirst() {
    const atBottom = window.innerHeight + Math.round(window.scrollY) >= document.body.offsetHeight;
    if (first.getBoundingClientRect().top > offset && !atBottom) {
      items.forEach(function (li) { li.classList.remove('active'); });
      items[0].classList.add('active');
    }
  }

  highlightFirst();
  window.addEventListener('scroll', highlightFirst);

  // Jump straight to the clicked post. Smooth scrolling would pass through the
  // posts in between, briefly highlighting each of them on the way.
  document.querySelectorAll('.post-toc li a').forEach(function (link) {
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