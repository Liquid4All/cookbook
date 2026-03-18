# Floor Plan Panel: Implementation Plan

## Context

The left panel (`#house-panel`) is redesigned from a 2×2 grid to a **3-row × 2-column grid**.
Row 1 spans both columns and shows an interactive 2D house floor plan.
Rows 2–3 keep the existing four cards (Lights, Climate, Doors, Scenes).

```
┌───────────────────────────────────┐
│                                   │  ← row 1, col 1–2 merged
│         FLOOR PLAN                │     (~50% of panel height)
│                                   │
├──────────────────┬────────────────┤
│  LIGHTS          │  CLIMATE       │  ← row 2
├──────────────────┼────────────────┤
│  DOORS           │  SCENES        │  ← row 3
└──────────────────┴────────────────┘
```

---

## Key insight from the car cockpit reference

The car schematic uses **no SVG**. It is a positioned `div` with:

- `border-radius` shaping the car body
- Absolutely positioned `div.window-frame` children for each window
- A `.glass` child inside each frame that slides via CSS `transform`
- State toggled by adding/removing a class (`is-open`)

The house floor plan uses the **same technique**: a `div.house-body` shaped by CSS Grid,
with room `div`s as children, and door gap indicators as absolutely positioned elements.
Lighting states toggle a `.lit` class that swaps in a radial amber glow.

---

## Floor plan layout (CSS Grid inside `.house-body`)

```
grid-template-columns:  38%       24%       38%
grid-template-rows:     40%       12%       48%

  col 1          col 2        col 3
┌──────────────┬────────────┬──────────────┐  ← row 1 (40%)
│              │            │              │
│   BEDROOM    │  BATHROOM  │    OFFICE    │
│              │            │              │
├──────────────┴────────────┴──────────────┤  ← row 2 (12%)
│         ···  HALLWAY  ···                │
├──────────────┬───────────────────────────┤  ← row 3 (48%)
│              │                           │
│   KITCHEN    │       LIVING ROOM         │
│              │                           │
└──────────────┴───────────────────────────┘
```

- HALLWAY spans all 3 columns (`grid-column: 1 / 4`)
- LIVING ROOM spans columns 2–3 (`grid-column: 2 / 4`)

---

## Exterior door markers

Absolutely positioned on the house outline. Each door is a small coloured pill with a
glow dot, mirroring the car schematic's cyan window indicators.

| Door    | Position on house outline       | Locked       | Unlocked      |
|---------|---------------------------------|--------------|---------------|
| Front   | Bottom edge, center             | red glow dot | green glow dot |
| Back    | Top edge, center                | red          | green         |
| Garage  | Left edge, kitchen level        | red          | green         |
| Side    | Right edge, living room level   | red          | green         |

---

## Visual treatment (matching the car cockpit CSS)

| Element              | Car reference                                         | House equivalent                                  |
|----------------------|-------------------------------------------------------|---------------------------------------------------|
| Body background      | `#222226`                                             | same                                              |
| Wall / outline color | `#4a4a50`, 3px border                                 | same, 1px borders between rooms                   |
| Active state fill    | cyan `#00e5ff` glass sliding in/out                   | amber `radial-gradient` glow fading in            |
| Card background      | `radial-gradient(circle at top-left, rgba(cyan,0.18)` | same card style, accent follows `--gradient`      |
| Labels               | `11px uppercase letter-spacing: 0.12em`               | same, room name centered in each room div         |
| Door indicators      | pill buttons (Open/Close) outside the car             | small `●` dot + glow at wall gap, red/green       |
| Hover shimmer        | `linear-gradient(120deg, teal → purple)` on card      | same, inherited from existing `.panel` hover rule |

---

## Room light-on state (CSS only)

```css
/* lights off */
.fp-room {
  background: #151518;
}

/* lights on — toggled by JS */
.fp-room.lit {
  background: radial-gradient(
    ellipse at 50% 40%,
    rgba(245, 158, 11, 0.22),
    rgba(245, 158, 11, 0.07) 65%,
    transparent
  );
}
```

Light mode uses softer amber on a near-white room fill.

---

## Door locked / unlocked state (CSS only)

```css
.fp-door-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  transition: background 0.3s ease, box-shadow 0.3s ease;
}

.fp-door.locked   .fp-door-dot { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.7); }
.fp-door.unlocked .fp-door-dot { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.7); }
```

---

## Reactivity (JS additions)

A new `updateFloorplan(state)` function called inside the existing `refreshState()`:

```js
function updateFloorplan(state) {
  // Lights
  for (const [room, light] of Object.entries(state.lights || {})) {
    const el = document.getElementById(`fp-room-${room}`);
    if (el) el.classList.toggle("lit", light.state === "on");
  }
  // Doors
  for (const [door, doorState] of Object.entries(state.doors || {})) {
    const el = document.getElementById(`fp-door-${door}`);
    if (el) {
      el.classList.toggle("locked",   doorState === "locked");
      el.classList.toggle("unlocked", doorState === "unlocked");
    }
  }
}
```

No changes to the server, agent, or tools.

---

## Files changed

| File                  | Change                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `index.html`          | New 3-row panel structure + `.house-body` div tree + `updateFloorplan()` JS |
| `assets/style.css`    | Updated `#house-panel` grid rows, `.house-body` floor plan styles, room and door states |
