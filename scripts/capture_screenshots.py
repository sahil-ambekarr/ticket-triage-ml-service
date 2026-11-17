"""Capture real screenshots of the demo page with Playwright.

Start the service first (any provider), then:
    python -m playwright install chromium     # once
    python -m scripts.capture_screenshots --base-url http://localhost:4103
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

SAMPLES = {
    "predict-billing": ("Billing", "#predictBtn", 30_000),
    "explain-login": ("Login", "#explainBtn", 300_000),
    "predict-ambiguous-review": ("Ambiguous", "#predictBtn", 30_000),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:4103")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    parser.add_argument("--channel", default="chrome", help="browser channel; empty string for bundled Chromium")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # channel="chrome" uses the installed Chrome; drop it to use Playwright's own Chromium.
        browser = p.chromium.launch(channel=args.channel or None)
        page = browser.new_page(viewport={"width": 1280, "height": 760}, device_scale_factor=1)
        for name, (sample, button, timeout_ms) in SAMPLES.items():
            page.goto(args.base_url + "/")
            page.wait_for_function("document.getElementById('modelHint').textContent.includes('accuracy')")
            page.get_by_role("button", name=sample, exact=True).click()
            page.click(button)
            page.wait_for_selector("#result .verdict", timeout=timeout_ms)
            if button == "#explainBtn":
                page.wait_for_selector("#result .explain", timeout=timeout_ms)
            path = args.out / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"saved {path}")
        browser.close()


if __name__ == "__main__":
    main()
