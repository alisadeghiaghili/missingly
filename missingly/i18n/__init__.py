"""Lightweight internationalisation (i18n) for missingly.

Provides :class:`Translator`, a simple JSON-based lookup class that maps
dot-separated keys to translated strings.  Every user-facing string in the
package should be retrieved through this class so that reports and messages
can be rendered in any supported locale.

Supported locales live as ``<locale>.json`` files inside this package
directory.  English (``en``) is the canonical source of truth and is always
used as a fallback when a key is missing from the requested locale.

Example
-------
>>> from missingly.i18n import Translator
>>> tr = Translator("en")
>>> tr.t("overview.section_title")
'Dataset Overview'
>>> tr.t("mcar.detail_mcar", p_value=0.12)
'p = 0.1200 > 0.05: cannot reject MCAR. Simple imputation strategies are statistically justified.'
>>> Translator.available_locales()      # doctest: +SKIP
['en', 'fa']
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["Translator"]

_LOCALE_DIR = Path(__file__).parent
_FALLBACK_LOCALE = "en"


class Translator:
    """JSON-backed i18n lookup with English fallback.

    Parameters
    ----------
    locale : str
        BCP-47-style locale code, e.g. ``"en"``, ``"fa"``.
        If the locale file does not exist a :exc:`ValueError` is raised
        immediately so callers discover the problem early.

    Raises
    ------
    ValueError
        If *locale* has no corresponding ``<locale>.json`` file.

    Notes
    -----
    * Keys are dot-separated paths into the JSON tree, e.g.
      ``"mcar.detail_mcar"``.
    * Format arguments (``{name}``) are interpolated via :meth:`str.format`
      when keyword arguments are supplied to :meth:`t`.
    * Missing keys fall back to English; if the key is also absent in English
      the raw key string is returned so the app never crashes.
    * ``_meta`` keys (``direction``, ``font_family``, ``locale_name``) are
      conventional — templates and callers may use them for layout hints
      (e.g. RTL direction for Persian).
    """

    def __init__(self, locale: str = "en") -> None:
        self._locale = locale
        self._data: dict[str, Any] = self._load(locale)
        if locale != _FALLBACK_LOCALE:
            self._fallback: dict[str, Any] = self._load(_FALLBACK_LOCALE)
        else:
            self._fallback = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(locale: str) -> dict[str, Any]:
        path = _LOCALE_DIR / f"{locale}.json"
        if not path.exists():
            available = [p.stem for p in _LOCALE_DIR.glob("*.json")]
            raise ValueError(
                f"Locale {locale!r} is not available. "
                f"Supported locales: {sorted(available)}"
            )
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _resolve(data: dict[str, Any], key: str) -> str | None:
        """Walk *data* along dot-separated *key*; return str value or None."""
        node: Any = data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node if isinstance(node, str) else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def locale(self) -> str:
        """The active locale code, e.g. ``'en'``."""
        return self._locale

    @property
    def direction(self) -> str:
        """Text direction: ``'ltr'`` or ``'rtl'``."""
        return self._resolve(self._data, "_meta.direction") or "ltr"

    @property
    def font_family(self) -> str:
        """Preferred font-family string for this locale."""
        return (
            self._resolve(self._data, "_meta.font_family")
            or "system-ui, sans-serif"
        )

    def t(self, key: str, **kwargs: Any) -> str:
        """Return the translated string for *key*, with English fallback.

        Parameters
        ----------
        key : str
            Dot-separated path into the locale JSON tree,
            e.g. ``"overview.section_title"``.
        **kwargs
            Named format arguments interpolated into the translated string
            via :meth:`str.format`.  Unused kwargs are silently ignored.

        Returns
        -------
        str
            Translated (and formatted) string, English fallback string, or
            the raw *key* if the key is missing from both locales.

        Examples
        --------
        >>> tr = Translator("en")
        >>> tr.t("overview.section_title")
        'Dataset Overview'
        >>> tr.t("mcar.detail_mcar", p_value=0.031)
        'p = 0.0310 ≤ 0.05: MCAR is rejected. Model-based methods are preferred.'
        """
        value = self._resolve(self._data, key)
        if value is None:
            value = self._resolve(self._fallback, key)
        if value is None:
            return key  # last resort — never crash
        if kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value  # return unformatted rather than crash
        return value

    @staticmethod
    def available_locales() -> list[str]:
        """Return sorted list of locale codes with a JSON file present.

        Returns
        -------
        list[str]
            E.g. ``['en', 'fa']``.
        """
        return sorted(p.stem for p in _LOCALE_DIR.glob("*.json"))
