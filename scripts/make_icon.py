"""Generate the desktop app's source icon.

Produces a 1024x1024 PNG for `npm run tauri icon`, which expands it into every
size and format the bundler needs.

A placeholder, but a deliberate one: a microphone on the calm-cyan brand colour
from the design system, no text (it renders illegibly at 32px), and a shape
that stays recognisable when small.

    python scripts/make_icon.py
    cd desktop && npm run tauri icon src-tauri/icon-source.png
"""

from __future__ import annotations

import sys
from pathlib import Path

SIZE = 1024
OUT = Path(__file__).resolve().parent.parent / "desktop" / "src-tauri" / "icon-source.png"

# Brand colours, matching --primary / --primary-strong in styles.css.
TEAL_DARK = (14, 116, 144, 255)
TEAL_LIGHT = (34, 211, 238, 255)
WHITE = (255, 255, 255, 255)


def main() -> int:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("Pillow is required: pip install Pillow")

    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Rounded-square background with a vertical gradient. Drawn as rows then
    # masked, which is simpler than compositing two layers.
    gradient = Image.new("RGBA", (SIZE, SIZE))
    grad_draw = ImageDraw.Draw(gradient)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        grad_draw.line(
            [(0, y), (SIZE, y)],
            fill=tuple(
                int(TEAL_LIGHT[i] + (TEAL_DARK[i] - TEAL_LIGHT[i]) * t)
                for i in range(4)
            ),
        )

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (SIZE - 1, SIZE - 1)], radius=int(SIZE * 0.22), fill=255
    )
    image.paste(gradient, (0, 0), mask)

    # Microphone capsule.
    cx = SIZE / 2
    cap_w, cap_h = SIZE * 0.24, SIZE * 0.40
    cap_top = SIZE * 0.20
    draw.rounded_rectangle(
        [(cx - cap_w / 2, cap_top), (cx + cap_w / 2, cap_top + cap_h)],
        radius=cap_w / 2,
        fill=WHITE,
    )

    # Cradle arc.
    stroke = SIZE * 0.045
    arc_w = SIZE * 0.42
    arc_top = cap_top + cap_h * 0.42
    draw.arc(
        [(cx - arc_w / 2, arc_top), (cx + arc_w / 2, arc_top + arc_w)],
        start=0,
        end=180,
        fill=WHITE,
        width=int(stroke),
    )

    # Stand.
    stand_top = arc_top + arc_w / 2
    draw.rounded_rectangle(
        [(cx - stroke / 2, stand_top), (cx + stroke / 2, SIZE * 0.80)],
        radius=stroke / 2,
        fill=WHITE,
    )
    base_w = SIZE * 0.22
    draw.rounded_rectangle(
        [(cx - base_w / 2, SIZE * 0.78), (cx + base_w / 2, SIZE * 0.78 + stroke)],
        radius=stroke / 2,
        fill=WHITE,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, "PNG")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print("\nNext: cd desktop && npm run tauri icon src-tauri/icon-source.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
