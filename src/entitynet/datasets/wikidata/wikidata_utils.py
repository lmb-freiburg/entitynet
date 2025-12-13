import re


def strip_label(name: str):
    assert name.startswith('"')
    if name.endswith('"@en'):
        name = name[1:-4]
    elif name.endswith('"@mul'):
        name = name[1:-5]
    elif name.endswith('"'):
        name = name[1:-1]
    else:
        raise ValueError(f"Unexpected name format: {name}")
    return name


def strip_label_no_quotes(name: str, assert_has_lang: bool = True):
    if name.endswith("@en"):
        name = name[:-3]
    elif name.endswith("@mul"):
        name = name[:-4]
    elif assert_has_lang:
        raise ValueError(f"Unexpected name format: {name=} missing @lang suffix")
    return name


_QUOTED_WITH_LANG = re.compile(r'^"(?P<label>.+)"@(?P<lang>[a-z]+)$')
_QUOTED_WITHOUT_LANG = re.compile(r'^"(?P<label>.+)"$')
_UNQUOTED_WITH_LANG = re.compile(r"^(?P<label>.+)@(?P<lang>[a-z]+)$")


def strip_label_allow_multilang(name: str):
    """
    Regex-based alternative that extracts labels with or without language markers.
    """
    match = _QUOTED_WITH_LANG.match(name)
    if match:
        return match.group("label")
    match = _QUOTED_WITHOUT_LANG.match(name)
    if match:
        return match.group("label")
    raise ValueError(f"Unexpected name format (regex variant): {name}")


def strip_label_no_quotes_allow_multilang(name: str, assert_has_lang: bool = True):
    """
    Regex-based alternative that extracts labels missing the leading quote.
    """
    match = _UNQUOTED_WITH_LANG.match(name)
    if match:
        return match.group("label")
    if assert_has_lang:
        raise ValueError(f"Unexpected name format (regex variant): {name=} missing @lang suffix")
    return name
