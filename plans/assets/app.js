/* ===================================================
   Ambix Plan Site — app.js
   Navigation, theme, sidebar, timeline, filter
   =================================================== */

(function () {
  "use strict";

  /* ---- Theme toggle ------------------------------------------------- */
  const THEME_KEY = "ambix-theme";

  function applyTheme(theme) {
    if (theme === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else if (theme === "light") {
      document.documentElement.removeAttribute("data-theme");
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    localStorage.setItem(THEME_KEY, theme);
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent =
        theme === "dark" ? "☀ Light" : theme === "light" ? "⊙ Auto" : "☽ Dark";
    }
  }

  function cycleTheme() {
    const current =
      document.documentElement.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark-auto"
        : "light-auto");
    const stored = localStorage.getItem(THEME_KEY) || "auto";
    const next = { auto: "dark", dark: "light", light: "auto" }[stored] || "auto";
    applyTheme(next);
  }

  // Apply saved theme immediately (before paint)
  (function () {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved && saved !== "auto") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  })();

  /* ---- Sidebar active link ------------------------------------------ */
  function updateSidebarActive() {
    const links = document.querySelectorAll(".sidebar a[href^='#']");
    if (!links.length) return;
    const headings = document.querySelectorAll(".main-content h2, .main-content h3");
    let current = "";
    headings.forEach((h) => {
      const top = h.getBoundingClientRect().top;
      if (top < 120) current = "#" + h.id;
    });
    links.forEach((a) => {
      a.classList.toggle("active", a.getAttribute("href") === current);
    });
  }

  /* ---- Mobile sidebar ---------------------------------------------- */
  function initSidebarToggle() {
    const btn = document.getElementById("sidebar-toggle");
    const sidebar = document.querySelector(".sidebar");
    if (!btn || !sidebar) return;
    btn.addEventListener("click", () => sidebar.classList.toggle("open"));
    document.addEventListener("click", (e) => {
      if (!sidebar.contains(e.target) && e.target !== btn) {
        sidebar.classList.remove("open");
      }
    });
  }

  /* ---- Timeline phase expand --------------------------------------- */
  function initTimeline() {
    document.querySelectorAll(".timeline-phase").forEach((phase) => {
      phase.addEventListener("click", (e) => {
        const wasOpen = phase.classList.contains("open");
        document
          .querySelectorAll(".timeline-phase.open")
          .forEach((p) => p.classList.remove("open"));
        if (!wasOpen) phase.classList.add("open");
      });
    });
    // Close on outside click
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".timeline-phase")) {
        document
          .querySelectorAll(".timeline-phase.open")
          .forEach((p) => p.classList.remove("open"));
      }
    });
  }

  /* ---- Status table filter ----------------------------------------- */
  function initStatusFilter() {
    const bar = document.querySelector(".filter-bar");
    if (!bar) return;
    bar.querySelectorAll(".filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        bar.querySelectorAll(".filter-btn").forEach((b) =>
          b.classList.remove("active")
        );
        btn.classList.add("active");
        const filter = btn.dataset.filter;
        document.querySelectorAll(".status-row").forEach((row) => {
          if (filter === "all") {
            row.style.display = "";
          } else {
            row.style.display = row.dataset.status === filter ? "" : "none";
          }
        });
      });
    });
  }

  /* ---- Copy prompt button ------------------------------------------ */
  function initCopyButtons() {
    document.querySelectorAll(".btn-copy").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pre = btn.previousElementSibling;
        if (!pre) return;
        const text = pre.textContent;
        if (navigator.clipboard) {
          navigator.clipboard.writeText(text).then(() => {
            btn.textContent = "Copied!";
            btn.classList.add("copied");
            setTimeout(() => {
              btn.textContent = "Copy";
              btn.classList.remove("copied");
            }, 1800);
          });
        } else {
          // Fallback for older browsers
          const ta = document.createElement("textarea");
          ta.value = text;
          ta.style.position = "fixed";
          ta.style.top = "-9999px";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          btn.textContent = "Copied!";
          btn.classList.add("copied");
          setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 1800);
        }
      });
    });
  }

  /* ---- Smooth scroll for anchor links ------------------------------ */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((a) => {
      a.addEventListener("click", (e) => {
        const id = a.getAttribute("href").slice(1);
        const target = document.getElementById(id);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
    });
  }

  /* ---- DOMContentLoaded -------------------------------------------- */
  document.addEventListener("DOMContentLoaded", () => {
    // Wire theme toggle
    const themeBtn = document.getElementById("theme-toggle");
    if (themeBtn) themeBtn.addEventListener("click", cycleTheme);

    // Apply any saved theme
    const saved = localStorage.getItem(THEME_KEY);
    if (saved && saved !== "auto") applyTheme(saved);

    initSidebarToggle();
    initTimeline();
    initStatusFilter();
    initCopyButtons();
    initSmoothScroll();

    // Active sidebar on scroll
    window.addEventListener("scroll", updateSidebarActive, { passive: true });
    updateSidebarActive();
  });
})();
