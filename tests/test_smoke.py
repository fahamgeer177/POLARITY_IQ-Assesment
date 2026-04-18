from __future__ import annotations


def test_package_imports() -> None:
    import polarity_iq  # noqa: F401
    from polarity_iq import cli  # noqa: F401


def test_cli_parser_builds() -> None:
    from polarity_iq.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.cmd == "doctor"
