// Runs before the page is painted so a saved light or dark choice does not flash the other theme first.
// (A separate file rather than an inline script, because the page's security policy forbids inline scripts.)
try {
  var saved = window.localStorage.getItem("reglens.theme");
  if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
} catch (error) {
  // storage is blocked: the page follows the system setting
}
