"""Unit tests for the Markdown widget."""

from __future__ import annotations

from pathlib import Path

import pytest

from textual import on
from textual.app import App, ComposeResult
from textual.content import Span
from textual.style import Style
from textual.widgets import Markdown
from textual.widgets._markdown import MarkdownBlock


class MarkdownApp(App[None]):
    def __init__(self, markdown: str) -> None:
        super().__init__()
        self._markdown = markdown

    def compose(self) -> ComposeResult:
        yield Markdown(self._markdown)


@pytest.mark.parametrize(
    ["document", "expected_types"],
    [
        # Basic markup.
        ("", []),
        ("# Hello", ["heading"]),
        ("## Hello", ["heading"]),
        ("### Hello", ["heading"]),
        ("#### Hello", ["heading"]),
        ("##### Hello", ["heading"]),
        ("###### Hello", ["heading"]),
        ("---", ["hr"]),
        ("Hello", ["paragraph"]),
        ("Hello\nWorld", ["paragraph"]),
        ("- One\n-Two", ["paragraph"]),
        (
            "1. One\n2. Two",
            ["paragraph", "paragraph"],
        ),
        ("    1", ["fence"]),
        ("```\n1\n```", ["fence"]),
        ("```python\n1\n```", ["fence"]),
        ("""| One | Two |\n| :- | :- |\n| 1 | 2 |""", ["table"]),
    ],
)
async def test_markdown_block_types(
    document: str, expected_types: list[str]
) -> None:
    """A Markdown document should parse into the expected block type list."""
    async with MarkdownApp(document).run_test() as pilot:
        await pilot.pause()
        markdown = pilot.app.query_one(Markdown)
        assert [block.block_type for block in markdown._blocks] == expected_types


async def test_heading_levels() -> None:
    """Heading levels should be correctly parsed."""
    document = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
    async with MarkdownApp(document).run_test() as pilot:
        await pilot.pause()
        markdown = pilot.app.query_one(Markdown)
        headings = [b for b in markdown._blocks if b.block_type == "heading"]
        assert [h.level for h in headings] == [1, 2, 3, 4, 5, 6]


async def test_softbreak_split_links_rendered_correctly() -> None:
    """Test for https://github.com/Textualize/textual/issues/2805"""

    document = """\
My site [has
this
URL](https://example.com)\
"""
    async with MarkdownApp(document).run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        paragraphs = [b for b in markdown._blocks if b.block_type == "paragraph"]
        assert len(paragraphs) == 1
        paragraph = paragraphs[0]
        assert paragraph.content.plain == "My site has\nthis\nURL"

        expected_spans = [
            Span(8, 20, Style.from_meta({"@click": "link('https://example.com')"})),
        ]

    assert paragraph.content.spans == expected_spans


async def test_load_non_existing_file() -> None:
    """Loading a file that doesn't exist should result in the obvious error."""
    async with MarkdownApp("").run_test() as pilot:
        with pytest.raises(FileNotFoundError):
            await pilot.app.query_one(Markdown).load(
                Path("---this-does-not-exist---.it.is.not.a.md")
            )


@pytest.mark.parametrize(
    ("anchor", "found"),
    [
        ("hello-world", False),
        ("hello-there", True),
    ],
)
async def test_goto_anchor(anchor: str, found: bool) -> None:
    """Going to anchors should return a boolean: whether the anchor was found."""
    document = "# Hello There\n\nGeneral.\n"
    async with MarkdownApp(document).run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        assert markdown.goto_anchor(anchor) is found


async def test_update_of_document_posts_table_of_content_update_message() -> None:
    """Updating the document should post a TableOfContentsUpdated message."""

    messages: list[str] = []

    class TableOfContentApp(App[None]):
        def compose(self) -> ComposeResult:
            yield Markdown("# One\n\n#Two\n")

        @on(Markdown.TableOfContentsUpdated)
        def log_table_of_content_update(
            self, event: Markdown.TableOfContentsUpdated
        ) -> None:
            nonlocal messages
            messages.append(event.__class__.__name__)

    async with TableOfContentApp().run_test() as pilot:

        assert messages == ["TableOfContentsUpdated"]
        await pilot.app.query_one(Markdown).update("")
        await pilot.pause()
        assert messages == ["TableOfContentsUpdated", "TableOfContentsUpdated"]


async def test_markdown_quoting():
    # https://github.com/Textualize/textual/issues/3350
    links = []

    class MyApp(App):
        def compose(self) -> ComposeResult:
            self.md = Markdown(markdown="[tété](tété)", open_links=False)
            yield self.md

        def on_markdown_link_clicked(self, message: Markdown.LinkClicked):
            links.append(message.href)

    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.click(Markdown, offset=(3, 0))
    assert links == ["tété"]


async def test_table_of_contents() -> None:
    """Table of contents should be generated from headings."""
    document = "# First\n\n## Second\n\n### Third\n"
    async with MarkdownApp(document).run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        toc = markdown.table_of_contents
        assert len(toc) == 3
        levels = [level for level, _, _ in toc]
        assert levels == [1, 2, 3]
        names = [name for _, name, _ in toc]
        assert names == ["First", "Second", "Third"]


async def test_empty_markdown_update() -> None:
    """Updating with empty markdown should clear blocks."""
    async with MarkdownApp("# Hello").run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        assert len(markdown._blocks) > 0
        await markdown.update("")
        assert len(markdown._blocks) == 0


async def test_markdown_source_property() -> None:
    """The source property should return the current markdown."""
    async with MarkdownApp("# Hello").run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        assert markdown.source == "# Hello"
        await markdown.update("## World")
        assert markdown.source == "## World"


async def test_markdown_fence_content() -> None:
    """Fence blocks should preserve code content."""
    document = "```python\nprint('hello')\n```"
    async with MarkdownApp(document).run_test() as pilot:
        markdown = pilot.app.query_one(Markdown)
        fences = [b for b in markdown._blocks if b.block_type == "fence"]
        assert len(fences) == 1
        assert "print('hello')" in fences[0].content.plain
