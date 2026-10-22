"""Task instructions, one per (surface, mode). Served in the reset observation.

These are the environment's own statement of the task, so every harness (the
rollout script, a trainer, Claude over MCP) starts from the same wording, and a
result is only comparable within one `PROMPT_VERSION`.
"""

from __future__ import annotations

PROMPT_VERSION = "v5"  # v5: images come from files through the editor's own insert/import command; v4: GUI coordinate systems and line breaks, target shown with the task, looking is free; v3: no in-editor Reference panel; v2: HTML stylesheets

_GOAL = {
    "reference": (
        "Recreate the target design as exactly as you can. The target image is shown "
        "{where}. Match every element's position, size, stacking order, font, size and colour."
    ),
    "description": (
        "Build the design described below. There is no target image: read the brief, work out "
        "a layout that fits it, and use every asset and every line of copy it mentions.\n\n"
        "BRIEF\n{description}"
    ),
}

_CANVAS = (
    "The canvas is {width}x{height} px. Coordinates are canvas pixels from the top-left corner; "
    "x/y is an element's top-left, before rotation. Elements stack back to front in the order "
    "they are added."
)

_TEXT_RULES = (
    "Text is typed, never pasted as an image. Text boxes do not wrap: put line breaks in the "
    "text as \\n. A text box's height follows its lines; its width sets the alignment box. "
    "Only the fonts in list_fonts exist."
)

_SURFACE = {
    "tools": (
        "You edit the canvas with tool calls: add_image, add_text, add_shape, update_element, "
        "reorder_element, align_elements, delete_element, duplicate_element, set_background, "
        "undo and redo. view_assets shows every asset with its id; view_canvas shows your "
        "current render; list_elements lists what is on the canvas. " + _TEXT_RULES
    ),
    "html": (
        "You write the design as one HTML page and submit it with write_html. The page:\n"
        '<div id="canvas" style="width:{width}px;height:{height}px;background:#hex"> ... </div>\n'
        "Every element inside the canvas is absolutely positioned (left/top, or right/bottom) in "
        "px, %, em or rem; wrapper divs are fine and pass their offset on. A <style> block with "
        "tag, class and id rules works, and inline styles win. <img src=\"asset://a3\"> places an "
        "asset (width and height required). An element containing text is a text box: "
        "font-family, font-size, color, text-align, line-height, letter-spacing, font-weight, "
        "font-style, text-transform:uppercase; <span> styles part of it. Text never wraps: "
        "break lines with <br> (or newlines under white-space:pre). An empty div with a "
        "background (and optional border, border-radius; 50% for an ellipse) is a shape. "
        "transform: rotate(), translate() and opacity work; z-index orders layers. Flexbox, "
        "grid and other flow layout are not applied, and anything else unsupported is reported "
        "back as a warning. read_html returns the current page, view_canvas renders it, "
        "view_assets shows the assets. Only the fonts in list_fonts exist."
    ),
    "gui": (
        "You operate a real design editor through screenshots, with the mouse and keyboard. "
        "{coords} The assets are image files on this computer: open them with the editor's own "
        "command for adding an image (insert, place, import or upload; files already uploaded "
        "are listed in the editor's uploads), choose the file in the file "
        "window (each file is named after its asset id) and open it, or drag it from the window onto "
        "the canvas. If the image then follows the pointer, click where it goes or drag out its box. "
        "The file window can be moved by its title bar and resized from its edges. Text is added with the text tool, and "
        "every element's exact position, size, font and colour can be typed into the property "
        "panel. Double-click text to edit it; Escape finishes editing. Text never wraps inside its "
        "box, whatever its width: press Enter while editing to start a new line. Ctrl+Z undoes. "
        "Only the fonts in the editor's font list exist. The editor's layout may differ between "
        "episodes; read the screen rather than assuming where controls are."
    ),
}

_FINISH = {
    "tools": "When the design is finished call submit. You have at most {max_steps} actions; looking "
             "(view_canvas, view_reference, view_assets, list_elements, list_fonts) is free.",
    "html": "When the design is finished call submit. You have at most {max_steps} actions; looking "
            "(read_html, view_canvas, view_reference, view_assets, list_fonts) is free.",
    "gui": "When the design is finished, export it with the editor's Export button; exporting hands it in "
           "(the submit tool does the same). You have at most {max_steps} actions (clicks, drags, typing, "
           "keys); screenshots are free.",
    "gui_app": "When the design is finished call the submit tool; that hands in the current document. You have "
               "at most {max_steps} actions (clicks, drags, typing, keys); screenshots are free.",
    "gui_app_saved": "When the design is finished, save it, then call the submit tool; that hands in the current "
                     "document. You have at most {max_steps} actions (clicks, drags, typing, keys); screenshots are "
                     "free.",
}


# said instead of the action budget when an episode does not announce it (budget_note="efficient")
_EFFICIENT = "Work efficiently, in as few actions as the design needs"

_COORDS = {
    "pixels": "Screen coordinates are pixels of the {vw}x{vh} screenshot.",
    "normalized": "Screen coordinates are on a 0-1000 grid over the screenshot: [0, 0] is its top-left corner "
                  "and [1000, 1000] its bottom-right, whatever the image size.",
}


def _app(surface: str, skin: str | None):
    """The real app an episode runs in (realapps/registry.py), or None for the built-in editor."""
    if surface != "gui" or not skin:
        return None
    try:
        from ..realapps import registry
    except ImportError:  # flat layout inside the Docker image
        from realapps import registry
    return registry.apps().get(skin)


# How the task's images are found on "this computer": the default (asset-id names in Pictures) and a workspace
# (core.workspace: everyday names, an everyday folder, other pictures and documents around them).
_FILES_PLAIN = ("The task's images are files named after their asset ids (a1.png, a2.png, ...) in the Pictures folder "
                "of your home directory:")
_FILES_WORKSPACE = ("The task's images are the files listed under Images below. They are somewhere in your home "
                    "folders (Downloads, Desktop, Pictures, Documents, or folders inside them), among other pictures "
                    "and documents, not necessarily all in one place: find them, and look at each (in the file "
                    "window's preview, where it has one, or once it is in) to see which part of the target it is. To "
                    "bring each in:")
# an app whose own file window shows no preview (a plain list): the desktop's file manager shows the pictures
_FILES_APP = ("Its file window shows no preview: the Files app (Super+E) shows the pictures in your folders as "
              "thumbnails, so you can see which is which before you bring one in.")
_MOCK_FILES_PLAIN = "choose the file in the file window (each file is named after its asset id) and open it"
_MOCK_FILES_WORKSPACE = ("find the file in the file window (the images are the files listed under Images below, "
                         "somewhere in your home folders among other pictures and documents, not necessarily all in "
                         "one place; look at each to see which part of the target it is) and open it")


def instructions(surface: str, mode: str, info: dict, viewport=(1280, 800), coordinates: str = "pixels",
                 skin: str | None = None) -> str:
    canvas = info["canvas"]
    where = {
        "tools": "with the task and any time from view_reference",
        "html": "with the task and any time from view_reference",
        "gui": "with the task and any time from view_reference (it is not shown in the editor)",
    }[surface]
    parts = [
        f"You are a graphic designer. The task is a {info['format']} titled "
        f"\"{info['title']}\".",
        _GOAL[mode].format(where=where, description=info.get("description", "")),
        _CANVAS.format(width=canvas["width"], height=canvas["height"]),
    ]
    app = _app(surface, skin)
    surface_text = app.prompt if app is not None and app.prompt else _SURFACE[surface]
    if info.get("workspace"):  # the images under everyday names, in everyday folders, among other files
        if app is not None and app.extra.get("prompt_workspace"):  # an app whose images otherwise come another way
            surface_text = app.extra["prompt_workspace"]
        surface_text = surface_text.replace(_FILES_PLAIN + " add one", _FILES_WORKSPACE + " add it").replace(
            _FILES_PLAIN, _FILES_WORKSPACE).replace(_MOCK_FILES_PLAIN, _MOCK_FILES_WORKSPACE)
        if app is not None and app.extra.get("files_app"):
            surface_text += " " + _FILES_APP
    parts.append(surface_text.format(width=canvas["width"], height=canvas["height"],
                                     coords=_COORDS[coordinates].format(vw=viewport[0], vh=viewport[1])))
    if info.get("workspace"):  # files on a computer: their names are what there is to go on
        files = ", ".join(a["file"] for a in info["assets"])
        parts.append(f"Images: {files or 'none'}.")
    else:
        assets = ", ".join(f"{a['id']} ({a['kind']}, aspect {a['aspect_ratio']:.2f})"
                           for a in info["assets"])
        parts.append(f"Assets: {assets or 'none'}.")
    parts.append(f"Fonts used by this design: {', '.join(info['fonts_used'])}.")
    finish = (app.finish or _FINISH["gui_app"]) if app is not None else _FINISH[surface]
    if info.get("workspace") and app is not None:  # a lived-in computer: the work is saved before it is handed in
        finish = app.extra.get("finish_workspace") or _FINISH["gui_app_saved"]
    if info.get("budget_note") == "efficient":  # the limit stays, unannounced: ask for few actions instead
        finish = finish.replace("You have at most {max_steps} actions", _EFFICIENT)
    parts.append(finish.format(max_steps=info["max_steps"]))
    return "\n\n".join(parts)
