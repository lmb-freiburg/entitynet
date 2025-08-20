import os
import webbrowser
from pathlib import Path

from packg.log import logger


def dump_dataframe(
    dataframe, out_path, base_filename, as_html=True, open_in_browser=True, as_csv=True
) -> list[str]:
    """
    Dump dataframe to html and then opens in browser and returns list of urls to print and click on.
    """
    strs = []
    out_path = Path(out_path)
    if as_html:
        fn = out_path / f"{base_filename}.html"
        os.makedirs(fn.parent, exist_ok=True)
        fn_str = fn.as_posix()
        dataframe.to_html(fn_str)
        strs.append(f"file://{fn_str}")
        if open_in_browser:
            webbrowser_open_maybe(fn_str)
    if as_csv:
        fn = out_path / f"{base_filename}.csv"
        os.makedirs(fn.parent, exist_ok=True)
        dataframe.to_csv(fn)
        strs.append(fn.as_posix())
    return strs


def webbrowser_open_maybe(url, using=None, ignore_text_browser=True):
    try:
        browser = webbrowser.get(using=using)
    except webbrowser.Error:
        logger.error(f"No browser found to open url {url}")
        return
    ignore_browsers = set(["www-browser", "w3m", "lynx", "links", "elinks"])
    if browser.name in ignore_browsers and using is None and ignore_text_browser:
        logger.error(f"Browser {browser.name} in ignore list, not opening url {url}")
        return
    browser.open(url)
