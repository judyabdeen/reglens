// Small helpers for building the page. Everything is created with createElement and textContent: text that comes from the
// server or from a model is never parsed as markup, so it cannot add elements, scripts or styles to the page.

const SVG_NS = "http://www.w3.org/2000/svg";

function append(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(parent, child);
    else parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

// h("a", {class: "x", href: "..."}, "text", child...) builds an element. Special properties:
//   class, text, on ({event: handler}), dataset, style (set through the CSSOM, which the page's CSP allows), checked, value.
// Any other property becomes an attribute; true means a bare attribute, false/null/undefined means none.
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "on") for (const [type, handler] of Object.entries(value)) el.addEventListener(type, handler);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style") Object.assign(el.style, value);
    else if (key === "checked" || key === "value") el[key] = value;
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

export function replaceChildren(node, ...children) {
  clear(node);
  append(node, children);
  return node;
}

// ------------------------------------------------------------------------------------------------ icons

// Simple line icons on a 24 by 24 grid. Each is a list of [element, attributes].
const ICONS = {
  close: [["path", { d: "M6 6l12 12M18 6L6 18" }]],
  info: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 11v5.5M12 7.6v.01" }]],
  sun: [["circle", { cx: 12, cy: 12, r: 4 }], ["path", { d: "M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4" }]],
  moon: [["path", { d: "M20 14.2A8.2 8.2 0 1 1 9.8 4a6.6 6.6 0 0 0 10.2 10.2z" }]],
  plus: [["path", { d: "M12 5v14M5 12h14" }]],
  download: [["path", { d: "M12 4v11M7.5 10.8L12 15.3l4.5-4.5M5 19.5h14" }]],
  copy: [["rect", { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ["path", { d: "M5 15V6a2 2 0 0 1 2-2h8" }]],
  external: [["path", { d: "M14 4h6v6M20 4l-9 9M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 5 18.5v-10A1.5 1.5 0 0 1 6.5 7H11" }]],
  chevron: [["path", { d: "M9 6l6 6-6 6" }]],
  sliders: [["path", { d: "M4 7h9M19 7h1M4 17h1M11 17h9" }], ["circle", { cx: 16, cy: 7, r: 2.4 }], ["circle", { cx: 8, cy: 17, r: 2.4 }]],
  check: [["path", { d: "M5 12.5l4.2 4.2L19 7" }]],
  arrowUp: [["path", { d: "M12 19V5.5M5.5 11.5L12 5l6.5 6.5" }]],
  stop: [["rect", { x: 6, y: 6, width: 12, height: 12, rx: 2 }]],
  warn: [["path", { d: "M12 4l9 16H3z" }], ["path", { d: "M12 10v4.5M12 17.4v.01" }]],
  book: [["path", { d: "M5 4.5h10a3 3 0 0 1 3 3v12H8a3 3 0 0 1-3-3z" }], ["path", { d: "M5 16.5a3 3 0 0 1 3-3h10" }]],
  back: [["path", { d: "M15 6l-6 6 6 6" }]],
  forward: [["path", { d: "M9 6l6 6-6 6" }]],
};

export function icon(name, className = "icon") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  for (const [tag, attributes] of ICONS[name] ?? []) {
    const part = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attributes)) part.setAttribute(key, String(value));
    svg.append(part);
  }
  return svg;
}

// ------------------------------------------------------------------------------------------------ small widgets

// A button with an icon and a text; below 40rem the text is hidden and the button keeps its accessible name.
export function withIcon(button, name, text) {
  clear(button);
  button.append(icon(name), h("span", { class: "btn-text", text }));
  return button;
}

let liveRegion = null;

// Tell screen-reader users what changed (a new answer, a copy confirmation). Setting the text again re-announces it.
export function announce(text) {
  liveRegion ??= document.getElementById("live");
  if (!liveRegion) return;
  liveRegion.textContent = "";
  window.setTimeout(() => {
    liveRegion.textContent = text;
  }, 30);
}

export function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

// Only http(s) addresses become links; anything else (a data or javascript address in a damaged manifest) is dropped.
export function safeUrl(value) {
  try {
    const url = new URL(String(value ?? ""));
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : "";
  } catch {
    return "";
  }
}

export function externalLink(url, text, className = "link") {
  const href = safeUrl(url);
  if (!href) return null;
  return h("a", { class: className, href, target: "_blank", rel: "noopener noreferrer" }, text, icon("external", "icon icon-inline"));
}

// ------------------------------------------------------------------------------------------------ dialogs

// <dialog> gives a focus trap, Escape and an inert page behind it. This adds: close buttons, a click on the backdrop,
// and returning focus to the control that opened it.
export function wireDialog(dialog) {
  let opener = null;
  for (const button of dialog.querySelectorAll("[data-close]")) button.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    if (opener?.isConnected) opener.focus();
    opener = null;
  });
  return {
    open(from = document.activeElement) {
      opener = from instanceof HTMLElement ? from : null;
      if (!dialog.open) dialog.showModal();
    },
    close: () => dialog.close(),
  };
}
