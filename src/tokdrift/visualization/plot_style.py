
_NAMING_DISPLAY_MAP = {
    "camel_case-snake_case": "N1",
    "camel_case-pascal_case": "N2",
    "camel_case-screaming_snake_case": "N3",
    "snake_case-camel_case": "N4",
    "snake_case-pascal_case": "N5",
    "snake_case-screaming_snake_case": "N6",
}

_SPACING_DISPLAY_MAP = {
    "rparentheses_semicolon": "S11",
    "period_name": "S15",
    "lparentheses_rparentheses": "S14",
    "lparentheses_name": "S16",
    "rparentheses_rparentheses": "S13",
    "period_asterisk": "S9",
    "double_plus_rparentheses": "S8",
    "rparentheses_period": "S3",
    "rparentheses_colon": "S10",
    "lsquarebracket_name": "S7",
    "rsquarebracket_rparentheses": "S4",
    "op_semicolon": "S12",
    "op_lparentheses": "S6",
    "op_lsquarebracket": "S2",
    "op_rsquarebracket": "S5",
    "op_dash": "S1",
    "op_name": "S17",
    "op_all": "S18",
}

_EXTRA_DISPLAY_MAP = {
    "baseline": "baseline",
}


def _build_column_display_map():
    mapping = {}

    # Naming variants
    for key, label in _NAMING_DISPLAY_MAP.items():
        mapping[key] = label

    # Spacing and operator variants
    mapping.update(_SPACING_DISPLAY_MAP)

    # Additional explicit entries
    mapping.update(_EXTRA_DISPLAY_MAP)

    return mapping


_COLUMN_DISPLAY_MAP = _build_column_display_map()


def get_variant_order() -> list:
    """Canonical ordering for naming and spacing variants."""
    return [
        'camel_case-snake_case',
        'camel_case-pascal_case',
        'camel_case-screaming_snake_case',
        'snake_case-camel_case',
        'snake_case-pascal_case',
        'snake_case-screaming_snake_case',
        'op_dash',
        'op_lsquarebracket',
        'rparentheses_period',
        'rsquarebracket_rparentheses',
        'op_rsquarebracket',
        'op_lparentheses',
        'lsquarebracket_name',
        'double_plus_rparentheses',
        'period_asterisk',
        'rparentheses_colon',
        'rparentheses_semicolon',
        'op_semicolon',
        'rparentheses_rparentheses',
        'lparentheses_rparentheses',
        'period_name',
        'lparentheses_name',
        'op_name',
        'op_all',
    ]


def get_column_display_name(column: str) -> str:
    """Return a human-friendly label for a metric column."""
    if column in _COLUMN_DISPLAY_MAP:
        return _COLUMN_DISPLAY_MAP[column]

    return column.replace('_', ' ')
