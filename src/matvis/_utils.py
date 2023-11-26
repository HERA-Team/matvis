import logging


def no_op(fnc):
    """No-op function."""
    return fnc


def ceildiv(a: int, b: int) -> int:
    """Ceiling division for integers.

    From https://stackoverflow.com/a/17511341/1467820
    """
    return -(a // -b)


def human_readable_size(size, decimal_places=2, indicate_sign=False):
    """Get a human-readable data size.

    From: https://stackoverflow.com/a/43690506/1467820
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if abs(size) < 1024.0:
            break
        if unit != "PiB":
            size /= 1024.0

    if indicate_sign:
        return f"{size:+.{decimal_places}f} {unit}"
    else:
        return f"{size:.{decimal_places}f} {unit}"
