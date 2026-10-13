"""
Multi-Command Parser Tests (Step 19)

Covers split_commands(): basic separators, command-aware splitting
(never splitting inside arguments, queries, or paths), edge cases
(empty/leading/trailing/repeated separators, whitespace), typo-tolerant
command-start detection, and the safety limits.

The parser only splits text — these tests never execute anything.
"""

import os
import sys
import unittest

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.command_registry import get_all_commands
from core.exceptions import JarvisError, MultiCommandError
from core.multi_command_parser import (
    COMMAND_START_WORDS,
    COMMAND_TOO_LONG_MESSAGE,
    MAX_COMMAND_LENGTH,
    MAX_COMMANDS,
    TOO_MANY_COMMANDS_MESSAGE,
    split_commands,
)


class TestBasicSplitting(unittest.TestCase):
    """The four documented separators split command boundaries."""

    def test_single_command_returns_one_segment(self):
        """PART 7: single-command input is completely unaffected."""
        self.assertEqual(split_commands("open chrome"), ["open chrome"])

    def test_and_splits_two_commands(self):
        self.assertEqual(
            split_commands("open chrome and open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_then_splits_two_commands(self):
        self.assertEqual(
            split_commands("open notepad then open calculator"),
            ["open notepad", "open calculator"],
        )

    def test_after_that_splits_two_commands(self):
        self.assertEqual(
            split_commands(
                "create folder projects after that open file explorer"
            ),
            ["create folder projects", "open file explorer"],
        )

    def test_next_splits_two_commands(self):
        self.assertEqual(
            split_commands("open chrome next open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_three_commands_split_in_order(self):
        self.assertEqual(
            split_commands(
                "open chrome and open calculator and open notepad"
            ),
            ["open chrome", "open calculator", "open notepad"],
        )

    def test_search_command_after_separator(self):
        """PART 2's flagship example."""
        self.assertEqual(
            split_commands("open chrome and search python tutorials"),
            ["open chrome", "search python tutorials"],
        )

    def test_help_and_commands_split(self):
        self.assertEqual(
            split_commands("help and commands"),
            ["help", "commands"],
        )


class TestPunctuationSeparators(unittest.TestCase):
    """Conservative punctuation support (PART 28)."""

    def test_semicolon_splits(self):
        self.assertEqual(
            split_commands("open chrome; open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_semicolon_without_space_splits(self):
        self.assertEqual(
            split_commands("open chrome;open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_comma_then_splits(self):
        self.assertEqual(
            split_commands("open chrome, then open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_bare_comma_splits_when_command_follows(self):
        self.assertEqual(
            split_commands("open chrome,open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_comma_inside_query_does_not_split(self):
        """Commas can be valid argument content."""
        self.assertEqual(
            split_commands("search web cats, dogs and more"),
            ["search web cats, dogs and more"],
        )

    def test_semicolon_without_command_after_does_not_split(self):
        self.assertEqual(
            split_commands("open chrome; powershell Remove-Item x"),
            ["open chrome; powershell Remove-Item x"],
        )


class TestNoFalseSplits(unittest.TestCase):
    """A separator only splits when a Jarvis command follows (PART 4)."""

    def test_search_query_with_and_stays_whole(self):
        self.assertEqual(
            split_commands("search web cats and dogs"),
            ["search web cats and dogs"],
        )

    def test_search_query_with_and_machine_learning(self):
        self.assertEqual(
            split_commands("search web python and machine learning"),
            ["search web python and machine learning"],
        )

    def test_search_query_with_and_places_stays_whole(self):
        self.assertEqual(
            split_commands(
                "search web best restaurants in Jaipur and Delhi"
            ),
            ["search web best restaurants in Jaipur and Delhi"],
        )

    def test_file_name_with_and_stays_whole(self):
        self.assertEqual(
            split_commands("open rock and roll.mp3"),
            ["open rock and roll.mp3"],
        )

    def test_windows_path_containing_and_stays_whole(self):
        self.assertEqual(
            split_commands("open C:\\Users\\Test\\and\\data.txt"),
            ["open C:\\Users\\Test\\and\\data.txt"],
        )

    def test_windows_path_with_spaced_and_stays_whole(self):
        """The text after 'and' is not a command start."""
        self.assertEqual(
            split_commands("open C:\\Users\\Test\\ and data.txt"),
            ["open C:\\Users\\Test\\ and data.txt"],
        )

    def test_then_inside_query_does_not_split(self):
        self.assertEqual(
            split_commands("search web to be or not then to be"),
            ["search web to be or not then to be"],
        )

    def test_word_containing_and_is_not_a_separator(self):
        self.assertEqual(
            split_commands("open android studio"),
            ["open android studio"],
        )

    def test_non_command_after_separator_does_not_split(self):
        self.assertEqual(
            split_commands("open chrome and dogs"),
            ["open chrome and dogs"],
        )


class TestCommandAwareBoundaries(unittest.TestCase):
    """Command-start detection reuses the Command Registry (PART 6)."""

    def test_start_words_come_from_the_registry(self):
        """Every start word is the first word of a registered alias —
        there is no duplicated command list in this module."""
        registry_first_words = set()
        for definition in get_all_commands().values():
            for alias in definition.aliases:
                parts = alias.split()
                if parts:
                    registry_first_words.add(parts[0].lower())
        self.assertTrue(COMMAND_START_WORDS)
        self.assertTrue(COMMAND_START_WORDS <= registry_first_words)

    def test_registered_start_words_present(self):
        for word in ("open", "search", "create", "delete", "lock",
                     "shutdown", "restart", "help", "commands"):
            self.assertIn(word, COMMAND_START_WORDS)

    def test_typo_command_start_splits(self):
        """PART 31: 'opn chrme and serch python' must split."""
        self.assertEqual(
            split_commands("opn chrme and serch python"),
            ["opn chrme", "serch python"],
        )

    def test_typo_segments_are_not_spell_corrected_here(self):
        """Spell correction happens per command in the pipeline."""
        segments = split_commands("opn chrme and serch python")
        self.assertIn("opn", segments[0])
        self.assertIn("serch", segments[1])

    def test_leading_separator_is_dropped(self):
        self.assertEqual(split_commands("and open chrome"),
                         ["open chrome"])
        self.assertEqual(split_commands("then open calculator"),
                         ["open calculator"])

    def test_separator_only_word_leaves_no_empty_command(self):
        self.assertEqual(split_commands("and"), [])
        self.assertEqual(split_commands("then"), [])


class TestEdgeCases(unittest.TestCase):
    """Empty input, whitespace, and separator noise are handled safely."""

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(split_commands(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(split_commands("   "), [])

    def test_non_string_returns_empty_list(self):
        self.assertEqual(split_commands(None), [])
        self.assertEqual(split_commands(42), [])

    def test_no_empty_segments_are_created(self):
        for segment in split_commands("open chrome and open calculator"):
            self.assertTrue(segment.strip())

    def test_whitespace_is_trimmed_and_collapsed(self):
        self.assertEqual(
            split_commands("  open    chrome  "), ["open chrome"]
        )

    def test_interior_segment_whitespace_is_cleaned(self):
        segments = split_commands(
            "open   chrome   and   open   calculator"
        )
        self.assertEqual(segments, ["open chrome", "open calculator"])

    def test_repeated_separators_do_not_create_empty_commands(self):
        self.assertEqual(
            split_commands("open chrome and and open calculator"),
            ["open chrome", "open calculator"],
        )

    def test_trailing_separator_is_dropped(self):
        self.assertEqual(split_commands("open chrome and"),
                         ["open chrome"])
        self.assertEqual(split_commands("open chrome then"),
                         ["open chrome"])
        self.assertEqual(split_commands("open chrome;"),
                         ["open chrome"])

    def test_dangling_separator_chain_is_dropped(self):
        self.assertEqual(split_commands("open chrome and then"),
                         ["open chrome"])
        self.assertEqual(split_commands("open chrome and then next"),
                         ["open chrome"])

    def test_uppercase_separators_split(self):
        self.assertEqual(
            split_commands("OPEN CHROME AND OPEN CALCULATOR"),
            ["OPEN CHROME", "OPEN CALCULATOR"],
        )

    def test_segment_case_is_preserved(self):
        """The pipeline lowercases later — splitting must not."""
        self.assertEqual(
            split_commands("Open Chrome and Open Notepad"),
            ["Open Chrome", "Open Notepad"],
        )


class TestCommandLimits(unittest.TestCase):
    """Chain size and segment length are bounded (PART 10/11)."""

    def _chain(self, count):
        return " and ".join(f"open app{i}" for i in range(count))

    def test_at_max_commands_is_allowed(self):
        segments = split_commands(self._chain(MAX_COMMANDS))
        self.assertEqual(len(segments), MAX_COMMANDS)

    def test_over_max_commands_raises_with_safe_message(self):
        with self.assertRaises(MultiCommandError) as ctx:
            split_commands(self._chain(MAX_COMMANDS + 1))
        self.assertEqual(str(ctx.exception), TOO_MANY_COMMANDS_MESSAGE)
        self.assertIn(str(MAX_COMMANDS), str(ctx.exception))

    def test_too_many_commands_error_is_user_safe(self):
        """The message never leaks internals."""
        with self.assertRaises(MultiCommandError) as ctx:
            split_commands(self._chain(MAX_COMMANDS + 1))
        message = str(ctx.exception)
        self.assertNotIn("Traceback", message)
        self.assertNotIn("File", message)

    def test_multicommand_error_is_a_jarvis_error(self):
        self.assertTrue(issubclass(MultiCommandError, JarvisError))

    def test_single_segment_at_length_limit_is_allowed(self):
        # "open " is 5 characters → 5 + 495 = 500 exactly.
        segments = split_commands("open " + "a" * (MAX_COMMAND_LENGTH - 5))
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]), MAX_COMMAND_LENGTH)

    def test_single_segment_over_length_limit_raises(self):
        # 5 + 496 = 501 characters.
        with self.assertRaises(MultiCommandError) as ctx:
            split_commands("open " + "a" * (MAX_COMMAND_LENGTH - 4))
        self.assertEqual(str(ctx.exception), COMMAND_TOO_LONG_MESSAGE)

    def test_multi_segment_over_length_limit_raises(self):
        long_target = "a" * (MAX_COMMAND_LENGTH + 1)
        with self.assertRaises(MultiCommandError) as ctx:
            split_commands(f"open chrome and open {long_target}")
        self.assertEqual(str(ctx.exception), COMMAND_TOO_LONG_MESSAGE)


class TestTransparency(unittest.TestCase):
    """The splitter is invisible for ordinary single commands."""

    def test_ordinary_single_commands_return_one_segment(self):
        for text in (
            "open chrome",
            "search web python tutorials",
            "help",
            "commands",
            "lock computer",
            "shutdown computer",
            "delete file test.txt",
            "create folder projects",
        ):
            self.assertEqual(split_commands(text), [text])

    def test_single_command_output_is_trimmed(self):
        self.assertEqual(split_commands(" open chrome "), ["open chrome"])


if __name__ == "__main__":
    unittest.main()
