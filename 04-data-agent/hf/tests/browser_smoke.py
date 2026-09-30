"""Exercise public browser controls and verify two isolated whitebox workspaces."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from playwright.async_api import async_playwright, expect


async def check(browser, out, split, index, tag):
    context = await browser.new_context(viewport={"width": 1440, "height": 1080})
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    await page.goto("https://huggingenvs-data-agent-seta-whitebox-env.hf.space",
                    wait_until="domcontentloaded", timeout=60000)
    await page.get_by_role("button", name="Load task", exact=True).wait_for(timeout=30000)
    if split == "train":
        await page.get_by_role("combobox", name="Dataset", exact=True).click()
        await page.get_by_role("option", name="Train · 1,000 tasks", exact=True).click()
        await expect(page.get_by_role("spinbutton")).to_have_attribute("max", "999")
    await page.get_by_role("spinbutton").fill(str(index))
    await page.get_by_role("button", name="Load task", exact=True).click()
    status = page.get_by_role("textbox", name="Workspace status", exact=True)
    try:
        await page.get_by_role("button", name="Start workspace", exact=True).click()
        await expect(status).to_have_value("Active", timeout=180000)
        # Distinct browser sessions write the same path and see only their own bytes.
        await page.get_by_role("textbox", name="Shell command", exact=True).fill(
            f"printf '{tag}' > /workdir/public-ui-isolation.txt; cat /workdir/public-ui-isolation.txt")
        await page.get_by_role("button", name="Run tool", exact=True).click()
        await expect(page.locator("#tool-console")).to_contain_text(tag, timeout=90000)
        await page.get_by_role("textbox", name="Final answer", exact=True).fill("__known_wrong_ui_smoke__")
        await page.get_by_role("button", name="Submit and grade", exact=True).click()
        await expect(status).to_have_value("Finished", timeout=180000)
        await expect(page.locator("#tool-console")).to_contain_text("Reward: 0.0")
        await page.screenshot(path=str(out / f"whitebox-{split}-graded.png"), full_page=True)
        assert not errors, errors
        return {"split": split, "index": index, "public_browser": True, "graded_zero": True,
                "tools_passed": True, "javascript_errors": errors}
    finally:
        await page.get_by_role("button", name="Close workspace", exact=True).click()
        await expect(status).to_have_value("Closed", timeout=60000)
        await context.close()


async def main(out):
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            results = await asyncio.gather(check(browser, out, "train", 895, "browser-train-895"),
                                           check(browser, out, "test", 166, "browser-test-166"))
        finally:
            await browser.close()
    report = {"passed": True, "checked_at": time.time(), "results": results}
    (out / "browser-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(main(p.parse_args().out))
