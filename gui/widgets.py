"""Custom Streamlit widgets for the GUI.

``live_slider`` / ``live_select_slider`` are drop-in replacements for
``st.slider`` / ``st.select_slider`` that emit their value **continuously while
dragging** instead of only on release. They are built on a single inline
Streamlit Custom Component v2 (CCv2): an ``<input type="range">`` whose ``input``
event (fired on every drag step) calls ``setStateValue``, which triggers a
rerun. That makes the plots update live as the handle moves.

Trade-off: every drag step is a full Streamlit rerun. For the confidence
threshold that is cheap (recolour only, cached data); for K it reloads that K's
mapping and rebuilds the figures on each step, so dragging K can feel heavy on
large datasets.

``headline_plot`` renders a Plotly figure client-side (CCv2 + Plotly.js) and
handles state selection in the browser with no reruns: single click on a legend
entry or any point toggles that state; double click isolates it (or restores
all); inactive states are recoloured to a single light gray rather than hidden.
"""

from __future__ import annotations

import streamlit as st

# Theme-aware styling that mimics Streamlit's native (BaseWeb) slider: a thin
# rounded track with a primary-colour fill up to a circular thumb, the current
# value floating above the thumb, and min/max labels at the ends. ``--st-*``
# variables are injected into the shadow root, so it follows the active theme.
_CSS = """
.wrap {
  width: 100%;
  box-sizing: border-box;
  color: var(--st-text-color, inherit);
  font-family: var(--st-font, sans-serif);
}
.label {
  font-size: 0.8rem;
  opacity: 0.6;
  margin-bottom: 0.25rem;
}
.track-area {
  position: relative;
  padding-top: 1.35rem;   /* room for the floating value bubble */
}
.bubble {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
  color: var(--st-text-color, #31333f);
  font-variant-numeric: tabular-nums;
  pointer-events: none;
}
input.range {
  -webkit-appearance: none;
  appearance: none;
  display: block;
  width: 100%;
  height: 6px;
  border-radius: 9999px;
  outline: none;
  margin: 0;
  cursor: pointer;
  background: var(--st-border-color, rgba(151, 166, 195, 0.35));
}
input.range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  margin-top: -5px;   /* centre the 16px thumb on the 6px track */
  border: none;
  border-radius: 50%;
  background: var(--st-primary-color, #ff4b4b);
  cursor: pointer;
  transition: box-shadow 0.15s ease;
}
input.range::-webkit-slider-thumb:hover {
  box-shadow: 0 0 0 6px color-mix(in srgb, var(--st-primary-color, #ff4b4b) 25%, transparent);
}
input.range::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border: none;
  border-radius: 50%;
  background: var(--st-primary-color, #ff4b4b);
  cursor: pointer;
}
input.range::-moz-range-track {
  height: 6px;
  border-radius: 9999px;
  background: transparent;
}
input.range:focus-visible::-webkit-slider-thumb {
  box-shadow: 0 0 0 6px color-mix(in srgb, var(--st-primary-color, #ff4b4b) 25%, transparent);
}
.ends {
  display: flex;
  justify-content: space-between;
  margin-top: 0.3rem;
  font-size: 0.72rem;
  opacity: 0.5;
  font-variant-numeric: tabular-nums;
}
"""

_HTML = """
<div class="wrap">
  <div class="label"></div>
  <div class="track-area">
    <div class="bubble"></div>
    <input class="range" type="range" />
    <div class="ends"><span class="end-min"></span><span class="end-max"></span></div>
  </div>
</div>
"""

_JS = """
const THUMB = 16
export default function (component) {
  const { parentElement, data, setStateValue } = component
  const label = parentElement.querySelector(".label")
  const bubble = parentElement.querySelector(".bubble")
  const input = parentElement.querySelector(".range")
  const endMin = parentElement.querySelector(".end-min")
  const endMax = parentElement.querySelector(".end-max")
  if (!input) return

  const d = data || {}
  const options = d.options || null
  const decimals = d.decimals ?? 2
  const prefix = d.prefix || ""
  const min = Number(d.min)
  const max = Number(d.max)

  const fmt = (v) => {
    if (options) {
      const i = Math.round(v)
      return options[i] != null ? String(options[i]) : String(v)
    }
    return prefix + Number(v).toFixed(decimals)
  }

  label.textContent = d.label || ""
  input.min = d.min
  input.max = d.max
  input.step = d.step
  endMin.textContent = fmt(min)
  endMax.textContent = fmt(max)

  const track = "var(--st-border-color, rgba(151, 166, 195, 0.35))"
  const fill = "var(--st-primary-color, #ff4b4b)"

  // Paint the fill gradient + position the value bubble over the thumb.
  const paint = (v) => {
    const pct = max > min ? ((v - min) / (max - min)) * 100 : 0
    input.style.background =
      `linear-gradient(90deg, ${fill} ${pct}%, ${track} ${pct}%)`
    // Correct for thumb width so the bubble stays centred over the handle.
    const offset = (0.5 - pct / 100) * THUMB
    bubble.style.left = `calc(${pct}% + ${offset}px)`
    bubble.textContent = fmt(v)
  }

  // True sync: reflect Python's value, but don't fight an active drag.
  const nextValue = d.value
  if (nextValue != null && Number(input.value) !== Number(nextValue)) {
    input.value = nextValue
  }
  paint(Number(input.value))

  input.oninput = (e) => {
    const v = Number(e.target.value)
    paint(v)                    // instant local feedback
    setStateValue("value", v)   // -> rerun (live update)
  }
}
"""

_LIVE_SLIDER = st.components.v2.component(
    "aim_live_slider", html=_HTML, css=_CSS, js=_JS
)


def live_slider(
    label: str,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
    key: str,
    *,
    decimals: int = 2,
    prefix: str = "",
) -> float:
    """A numeric slider that updates live while dragging. Returns the value."""
    state = st.session_state.get(key, {})
    try:
        cur = float(state.get("value", value))
    except (TypeError, ValueError):
        cur = float(value)
    cur = min(max(cur, min_value), max_value)

    result = _LIVE_SLIDER(
        key=key,
        data={
            "label": label,
            "min": min_value,
            "max": max_value,
            "step": step,
            "value": cur,
            "decimals": decimals,
            "prefix": prefix,
            "options": None,
        },
        default={"value": cur},
        on_value_change=lambda: None,
    )
    v = cur if result.value is None else float(result.value)
    return min(max(v, min_value), max_value)


def live_select_slider(label: str, options, value, key: str):
    """A discrete slider over ``options`` that updates live while dragging.

    Emits the selected option's index and maps it back to the option value, so
    non-contiguous option sets (e.g. a K range with a step) work correctly.
    """
    options = list(options)
    if not options:
        return None
    default_idx = options.index(value) if value in options else 0

    state = st.session_state.get(key, {})
    try:
        idx = int(state.get("value", default_idx))
    except (TypeError, ValueError):
        idx = default_idx
    idx = min(max(idx, 0), len(options) - 1)

    result = _LIVE_SLIDER(
        key=key,
        data={
            "label": label,
            "min": 0,
            "max": len(options) - 1,
            "step": 1,
            "value": idx,
            "decimals": 0,
            "prefix": "",
            "options": [str(o) for o in options],
        },
        default={"value": idx},
        on_value_change=lambda: None,
    )
    sel = idx if result.value is None else int(result.value)
    sel = min(max(sel, 0), len(options) - 1)
    return options[sel]


# --------------------------------------------------------------------------- #
# Headline plot component (client-side Plotly + interaction)
# --------------------------------------------------------------------------- #
# Renders a Plotly figure (built in Python) client-side and handles state
# selection entirely in the browser (no reruns), so it can replicate the native
# legend behaviour exactly and unify it with clicks on the plotted points:
#   * single click (legend entry OR any point of a state) -> toggle that state
#   * double click                                        -> isolate that state
#     (show only it; if already isolated, restore all)
# Non-active states are all drawn in one constant light-gray colour (full
# opacity), so overlapping points never stack into odd shades.
#
# State traces are matched by their ``legendgroup`` ("state<n>"); other traces
# (grey below-threshold, continuous confidence) are left untouched. Plotly.js is
# loaded from a CDN (reused if the page already exposes it); an offline page
# shows an inline error instead of the plot.
_HEADLINE_JS = """
const CDN = "https://esm.sh/plotly.js-dist-min@2.35.2"

const stateOf = (tr) => {
  const lg = (tr && tr.legendgroup) || ""
  return lg.indexOf("state") === 0 ? parseInt(lg.slice(5), 10) : null
}

export default async function (component) {
  const { parentElement, data, key } = component

  let gd = parentElement.querySelector(".aim-gd")
  if (!gd) {
    gd = document.createElement("div")
    gd.className = "aim-gd"
    gd.style.width = "100%"
    parentElement.appendChild(gd)
  }

  let Plotly = window.Plotly || window.__aimPlotly
  if (!Plotly) {
    try {
      Plotly = (await import(CDN)).default
      window.__aimPlotly = Plotly
    } catch (e) {
      gd.textContent = "Could not load Plotly.js (needs internet): " + e
      return
    }
  }

  const fig = JSON.parse(data.figureJson)
  const offColor = data.offColor || "#dcdcdc"

  const baseColors = fig.data.map((tr) => (tr.marker && tr.marker.color) || null)
  const states = Array.from(
    new Set(fig.data.map(stateOf).filter((s) => s !== null))
  ).sort((a, b) => a - b)
  const sig = states.join(",")

  // Active-state set, kept on window per component key. Reset when the set of
  // states changes (e.g. K changed), preserved across same-K reruns.
  if (!window.__aimActive) window.__aimActive = {}
  const store = window.__aimActive
  if (!store[key] || store[key].sig !== sig) {
    store[key] = { sig, active: new Set(states) }
  }

  // Plotly.react merges the new layout over the previous one and leaves STALE
  // subplot domains when the panel count changes (e.g. toggling the shared-gene
  // UMAP flips 2<->3 panels), which overlaps the panels. Purge first whenever the
  // axis structure changes so react rebuilds from a clean layout; an unchanged
  // structure (e.g. a K drag or a threshold change) still updates smoothly in
  // place. Purge drops Plotly's event listeners, so re-wire them after.
  const axisSig = Object.keys(fig.layout || {})
    .filter((k) => /^xaxis/.test(k))
    .sort()
    .join(",")
  if (gd.__aimAxisSig && gd.__aimAxisSig !== axisSig) {
    Plotly.purge(gd)
    gd.__aimWired = false
  }
  gd.__aimAxisSig = axisSig

  await Plotly.react(gd, fig.data, fig.layout || {}, {
    responsive: true,
    displaylogo: false,
  })

  // Streamlit keeps inactive tab panels in the DOM at zero width, so a plot in a
  // background tab is laid out at (near) zero width and stays that way when the
  // tab is shown -- it renders squeezed into the left. ``responsive: true`` only
  // reacts to *window* resizes, not to the container going hidden->visible, so
  // observe the div and refit whenever it gains width.
  if (!gd.__aimResizeWired && typeof ResizeObserver !== "undefined") {
    gd.__aimResizeWired = true
    let lastW = gd.clientWidth
    new ResizeObserver(() => {
      const w = gd.clientWidth
      if (w > 0 && w !== lastW) {
        lastW = w
        Plotly.Plots.resize(gd)
      }
    }).observe(gd)
  }

  const applyColors = () => {
    const active = store[key].active
    const idx = [], cols = []
    fig.data.forEach((tr, i) => {
      const s = stateOf(tr)
      if (s !== null && baseColors[i]) {
        idx.push(i)
        cols.push(active.has(s) ? baseColors[i] : offColor)
      }
    })
    if (idx.length) Plotly.restyle(gd, { "marker.color": cols }, idx)
  }

  const toggle = (s) => {
    const a = store[key].active
    if (a.has(s)) a.delete(s)
    else a.add(s)
    applyColors()
  }
  const isolate = (s) => {
    const a = store[key].active
    if (a.size === 1 && a.has(s)) store[key].active = new Set(states)
    else store[key].active = new Set([s])
    applyColors()
  }

  applyColors()

  // Refresh the live context read by the (once-attached) event handlers.
  gd.__aimCtx = { toggle, isolate, fig }
  if (!gd.__aimWired) {
    gd.__aimWired = true

    // Legend already distinguishes single vs double (mutually exclusive events).
    gd.on("plotly_legendclick", (ev) => {
      const s = stateOf(gd.__aimCtx.fig.data[ev.curveNumber])
      if (s !== null) gd.__aimCtx.toggle(s)
      return false // prevent the native hide
    })
    gd.on("plotly_legenddoubleclick", (ev) => {
      const s = stateOf(gd.__aimCtx.fig.data[ev.curveNumber])
      if (s !== null) gd.__aimCtx.isolate(s)
      return false
    })

    // Points have no native double-click event, so time it ourselves.
    let timer = null, pending = null
    gd.on("plotly_click", (ev) => {
      const pt = ev.points && ev.points[0]
      if (!pt) return
      const s = stateOf(pt.data)
      if (s === null) return
      if (timer && pending === s) {
        clearTimeout(timer); timer = null; pending = null
        gd.__aimCtx.isolate(s)
      } else {
        if (timer) { clearTimeout(timer); gd.__aimCtx.toggle(pending) }
        pending = s
        timer = setTimeout(() => {
          timer = null; pending = null
          gd.__aimCtx.toggle(s)
        }, 300)
      }
    })
  }
}
"""

# Plotly renders more reliably in the light DOM (it injects global styles and
# hover nodes), so opt out of the shadow-root isolation for this component.
_HEADLINE_PLOT = st.components.v2.component(
    "aim_headline_plot", js=_HEADLINE_JS, isolate_styles=False
)


def headline_plot(fig, key: str, *, off_color: str = "#dcdcdc") -> None:
    """Render ``fig`` (a Plotly figure) with client-side state toggle/isolate.

    ``off_color`` is the single light-gray colour used for every deactivated
    state.
    """
    _HEADLINE_PLOT(
        key=key,
        data={"figureJson": fig.to_json(), "offColor": off_color},
    )
