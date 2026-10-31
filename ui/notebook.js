/* The Living Notebook: small page behaviors. If anything here fails, the app still works. */
(function () {
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
    return value;
  }

  window.nbToggleTheme = function () {
    var dark = document.body.classList.toggle("dark");
    store("nb-theme", dark ? "dark" : "light");
  };
  window.nbDismissTip = function () {
    var tip = document.getElementById("nb-tip");
    if (tip) tip.hidden = true;
    store("nb-tip-seen", "1");
  };

  var HINT = '<svg viewBox="0 0 150 96" aria-hidden="true"><path d="M30 18c0-6 20-10 45-10s45 4 45 10-20 10-45 10-45-4-45-10zM30 18l6 60c0 6 16 10 39 10s39-4 39-10l6-60"/></svg>' +
    "<strong>Start drawing anywhere on the page</strong>One object, clear outlines. Mouse, pen or finger.";

  function brushButton(root) {
    return root && root.querySelector('button[aria-label="Brush"]');
  }

  function armBrush(root) {
    var brush = brushButton(root);
    if (brush) brush.click();
  }

  function setHint(root, visible) {
    var hint = root.querySelector(".nb-canvas-hint");
    if (hint) hint.classList.toggle("gone", !visible);
  }

  function setupCanvas() {
    var root = document.getElementById("nb-canvas");
    if (!root || root.dataset.nbReady || !brushButton(root)) return;
    root.dataset.nbReady = "1";
    var hint = document.createElement("div");
    hint.className = "nb-canvas-hint";
    hint.innerHTML = HINT;
    root.appendChild(hint);
    armBrush(root);
    root.addEventListener("pointerdown", function (event) {
      if (event.target.closest("canvas")) setHint(root, false);
    }, true);
    root.addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (button && button.getAttribute("aria-label") === "Clear canvas") {
        setHint(root, true);
        setTimeout(function () { armBrush(root); }, 120);
      }
    }, true);
  }

  window.nbResetCanvas = function () {
    var root = document.getElementById("nb-canvas");
    if (!root) return;
    var clear = root.querySelector('button[aria-label="Clear canvas"]');
    if (clear) clear.click();
    setHint(root, true);
    setTimeout(function () { armBrush(root); }, 200);
  };

  var runWasVisible = false;

  function follow(element) {
    var calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    element.scrollIntoView({ block: "start", behavior: calm ? "auto" : "smooth" });
  }

  function tick() {
    setupCanvas();
    var run = document.querySelector(".nb-run");
    var runVisible = !!(run && run.offsetParent);
    if (runVisible && !runWasVisible) follow(run);
    runWasVisible = runVisible;
    var now = Date.now() / 1000;
    document.querySelectorAll(".nb-elapsed[data-since]").forEach(function (el) {
      el.textContent = Math.max(0, now - parseFloat(el.dataset.since)).toFixed(0) + " s";
    });
    document.querySelectorAll("[data-count]").forEach(function (el) {
      if (el.dataset.counted) return;
      el.dataset.counted = "1";
      follow(el.closest(".nb-badge") || el);
      var target = parseInt(el.dataset.count, 10);
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      var start = performance.now();
      (function step(time) {
        var done = Math.min(1, (time - start) / 900);
        el.textContent = Math.round(target * done) + "%";
        if (done < 1) requestAnimationFrame(step);
      })(start);
    });
  }

  document.addEventListener("click", function (event) {
    var link = event.target.closest && event.target.closest(".nb-dl");
    if (link) link.classList.add("nb-saved");
  });

  function applyTheme() {
    document.body.classList.toggle("dark", store("nb-theme") === "dark");
  }

  function start() {
    setInterval(function () {
      try {
        applyTheme();
        tick();
        var tip = document.getElementById("nb-tip");
        if (tip && !tip.dataset.nbReady) {
          tip.dataset.nbReady = "1";
          tip.hidden = store("nb-tip-seen") === "1";
        }
      } catch (e) { /* decoration must never break the app */ }
    }, 250);
  }

  // Gradio may add this script after the page has already finished loading
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
