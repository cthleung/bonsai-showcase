// Running head, grow-plate insets, and ?y= scrolling for screenshots.
(() => {
  // only Chromium can run an SVG filter inside backdrop-filter; everyone else keeps the frosted fade
  const brands = navigator.userAgentData?.brands || [];
  if (brands.some((b) => /Chromium/.test(b.brand))) document.documentElement.classList.add("has-liquid");
  // the tapered glass bevel is the default (data-glass="full" on <html>);
  // ?glass=lip brings back the wobbly liquid lip for comparison
  const glass = new URLSearchParams(location.search).get("glass");
  if (glass) document.documentElement.dataset.glass = glass;

  // the bevel map for the full glass: a 1px-wide strip, stretched across the band.
  // Inside the rim, each row samples from a little above itself (green below the
  // middle), zero at the rim and zero at BEVEL px. The row at the rim matches the
  // photo above it, the rows just inside get stretched like a rounded glass edge,
  // and everything below the bevel is clear flat glass.
  const BEVEL = 72;
  const bevelMap = document.getElementById("bevel-map");
  const drawBevel = () => {
    const html = document.documentElement;
    const sheet = document.querySelector(".sheet");
    if (!bevelMap || !sheet || html.dataset.glass !== "full" || !html.classList.contains("has-liquid")) return;
    const h = Math.max(1, Math.round(parseFloat(getComputedStyle(sheet, "::before").height)));
    const c = document.createElement("canvas");
    c.width = 1;
    c.height = h;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(1, h);
    for (let y = 0; y < h; y++) {
      const t = Math.min(1, y / BEVEL);
      const v = -Math.sin(Math.PI * t);
      const i = y * 4;
      img.data[i] = 128;
      img.data[i + 1] = Math.round(128 + 127 * v);
      img.data[i + 2] = 128;
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    bevelMap.setAttribute("href", c.toDataURL());
    bevelMap.setAttribute("width", html.clientWidth);
    bevelMap.setAttribute("height", h);
  };
  drawBevel();
  addEventListener("resize", drawBevel);

  const rh = document.querySelector(".runhead");
  const label = rh.querySelector(".rh-label");
  const blocks = [...document.querySelectorAll("[data-runhead]")];

  // the running head shows whichever paper block sits under the bottom edge of the screen
  let ticking = false;
  const updateHead = () => {
    ticking = false;
    const line = innerHeight - 24;
    const current = blocks.filter((b) => {
      const r = b.getBoundingClientRect();
      return r.top < line && r.bottom > line;
    }).pop();
    if (current) label.textContent = current.dataset.runhead;
    rh.classList.toggle("is-on", Boolean(current));
  };
  addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(updateHead); } }, { passive: true });
  addEventListener("resize", updateHead);
  updateHead();

  // grow plates start framed to the text column: measure where it sits
  const setInsets = () => {
    document.querySelectorAll(".plate--grow").forEach((fig) => {
      const col = fig.closest(".chapter").querySelector(".opener");
      const r = col.getBoundingClientRect();
      const vw = document.documentElement.clientWidth;
      const ph = fig.querySelector(".ph");
      ph.style.setProperty("--l", `${Math.max(0, r.left)}px`);
      ph.style.setProperty("--r", `${Math.max(0, vw - r.left - r.width)}px`);
    });
  };
  setInsets();
  addEventListener("resize", setInsets);

  const y = new URLSearchParams(location.search).get("y");
  if (y !== null) {
    document.documentElement.style.scrollBehavior = "auto";
    const go = () => scrollTo(0, Number(y));
    document.fonts.ready.then(() => { go(); setInsets(); requestAnimationFrame(() => { go(); updateHead(); }); });
  }
})();
