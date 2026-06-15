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