from __future__ import annotations

import contextlib
import io
import unittest

from hummel_submit.cli import _parser, main


class HelpTests(unittest.TestCase):
    def test_help_subcommand_matches_top_level_help(self) -> None:
        expected = _parser().format_help()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = main(["help"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
