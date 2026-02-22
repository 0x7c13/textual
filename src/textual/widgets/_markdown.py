from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePath
from typing import Callable, Iterable, Optional
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text
from typing_extensions import TypeAlias

from textual._cells import cell_len
from textual._slug import TrackedSlugs, slug_for_tcss_id
from textual.app import ComposeResult
from textual.await_complete import AwaitComplete
from textual.cache import LRUCache
from textual.content import Content, Span
from textual.events import Mount
from textual.geometry import Region, Size
from textual.highlight import highlight
from textual.message import Message
from textual.reactive import reactive, var
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions
from textual.widget import Widget
from textual.widgets import Tree

TableOfContentsType: TypeAlias = "list[tuple[int, str, str | None]]"
"""Information about the table of contents of a markdown document.

The triples encode the level, the label, and the optional block id of each heading.
"""

BULLETS = ["• ", "▪ ", "‣ ", "⭑ ", "◦ "]
"""Unicode bullets used for unordered lists."""


class MarkdownStream:
    """An object to manage streaming markdown.

    This will accumulate markdown fragments if they can't be rendered fast enough.

    This object is typically created by the [Markdown.get_stream][textual.widgets.Markdown.get_stream] method.

    """

    def __init__(self, markdown_widget: Markdown) -> None:
        """
        Args:
            markdown_widget: Markdown widget to update.
        """
        self.markdown_widget = markdown_widget
        self._task: asyncio.Task | None = None
        self._new_markup = asyncio.Event()
        self._pending: list[str] = []
        self._stopped = False

    def start(self) -> None:
        """Start the updater running in the background.

        No need to call this, if the object was created by [Markdown.get_stream][textual.widgets.Markdown.get_stream].

        """
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the stream and await its finish."""
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None
            self._stopped = True

    async def write(self, markdown_fragment: str) -> None:
        """Append or enqueue a markdown fragment.

        Args:
            markdown_fragment: A string to append at the end of the document.
        """
        if self._stopped:
            raise RuntimeError("Can't write to the stream after it has stopped.")
        if not markdown_fragment:
            return
        self._pending.append(markdown_fragment)
        self._new_markup.set()
        await asyncio.sleep(0)

    async def _run(self) -> None:
        """Run a task to append markdown fragments when available."""
        try:
            while await self._new_markup.wait():
                new_markdown = "".join(self._pending)
                self._pending.clear()
                self._new_markup.clear()
                await asyncio.shield(self.markdown_widget.append(new_markdown))
        except asyncio.CancelledError:
            pass

        new_markdown = "".join(self._pending)
        if new_markdown:
            await self.markdown_widget.append(new_markdown)


class Navigator:
    """Manages a stack of paths like a browser."""

    def __init__(self) -> None:
        self.stack: list[Path] = []
        self.index = 0

    @property
    def location(self) -> Path:
        """The current location.

        Returns:
            A path for the current document.
        """
        if not self.stack:
            return Path(".")
        return self.stack[self.index]

    @property
    def start(self) -> bool:
        """Is the current location at the start of the stack?"""
        return self.index == 0

    @property
    def end(self) -> bool:
        """Is the current location at the end of the stack?"""
        return self.index >= len(self.stack) - 1

    def go(self, path: str | PurePath) -> Path:
        """Go to a new document.

        Args:
            path: Path to new document.

        Returns:
            New location.
        """
        location, anchor = Markdown.sanitize_location(str(path))
        if location == Path(".") and anchor:
            current_file, _ = Markdown.sanitize_location(str(self.location))
            path = f"{current_file}#{anchor}"
        new_path = self.location.parent / Path(path)
        self.stack = self.stack[: self.index + 1]
        new_path = new_path.absolute()
        self.stack.append(new_path)
        self.index = len(self.stack) - 1
        return new_path

    def back(self) -> bool:
        """Go back in the stack.

        Returns:
            True if the location changed, otherwise False.
        """
        if self.index:
            self.index -= 1
            return True
        return False

    def forward(self) -> bool:
        """Go forward in the stack.

        Returns:
            True if the location changed, otherwise False.
        """
        if self.index < len(self.stack) - 1:
            self.index += 1
            return True
        return False


# ---------------------------------------------------------------------------
# Block types -- lightweight data objects that represent parsed markdown
# ---------------------------------------------------------------------------


@dataclass
class MarkdownBlock:
    """Data object representing a parsed markdown block element."""

    block_type: str
    """The type of block (e.g. 'paragraph', 'heading', 'fence', 'hr', etc.)."""
    content: Content
    """The rendered Content for this block."""
    level: int = 0
    """The heading level (1-6) for heading blocks, or nesting level for lists."""
    block_id: str | None = None
    """An optional ID for the block (used for heading anchors)."""
    source_range: tuple[int, int] = (0, 0)
    """The (start_line, end_line) range in the source document."""
    style_name: str = ""
    """CSS component class name for styling."""
    top_margin: int = 0
    """Lines of margin above this block."""
    bottom_margin: int = 1
    """Lines of margin below this block."""
    indent: int = 0
    """Left indentation in cells."""
    prefix: str = ""
    """A prefix string (e.g. bullet character) to render before the first line."""
    padding_top: int = 0
    """Lines of padding above content (rendered with block style, unlike margin)."""
    padding_bottom: int = 0
    """Lines of padding below content (rendered with block style, unlike margin)."""
    padding_left: int = 0
    """Cells of padding to the left of content."""
    padding_right: int = 0
    """Cells of padding to the right of content (inside block background)."""
    border_left: str = ""
    """Character to render as a left border on every content line."""
    bq_depth: int = 0
    """Blockquote nesting depth (0 = not in blockquote)."""
    text_align: str = "left"
    """Text alignment: 'left', 'center', or 'right'."""
    code_language: str = ""
    """Language for code blocks."""
    is_header_row: bool = False
    """Whether this is a table header row."""
    table_headers: list[Content] | None = None
    """Header contents for table blocks."""
    table_rows: list[list[Content]] | None = None
    """Row contents for table blocks."""


class MarkdownFence:
    """Compatibility shim - references MarkdownBlock internally.

    This class exists to preserve the public API for code that checks
    `isinstance(block, MarkdownFence)`.
    """

    pass


# ---------------------------------------------------------------------------
# Token to Content conversion
# ---------------------------------------------------------------------------

def _token_to_content(
    token: Token, get_style: Callable[[str], Style] | None = None
) -> Content:
    """Convert an inline token to Textual Content.

    Args:
        token: A markdown token.
        get_style: Optional callable to resolve component class styles.

    Returns:
        Content instance.
    """
    if token.children is None:
        return Content("")

    tokens: list[str] = []
    spans: list[Span] = []
    style_stack: list[tuple[Style | str, int]] = []
    position: int = 0

    def add_content(text: str) -> None:
        nonlocal position
        tokens.append(text)
        position += len(text)

    def add_style(style: Style | str) -> None:
        style_stack.append((style, position))

    def close_tag() -> None:
        style, start = style_stack.pop()
        spans.append(Span(start, position, style))

    for child in token.children:
        child_type = child.type
        if child_type == "text":
            add_content(re.sub(r"\s+", " ", child.content))
        if child_type == "hardbreak":
            add_content("\n")
        if child_type == "softbreak":
            add_content(" ")
        elif child_type == "code_inline":
            add_style(".code_inline")
            add_content(child.content)
            close_tag()
        elif child_type == "em_open":
            add_style(".em")
        elif child_type == "strong_open":
            add_style(".strong")
        elif child_type == "s_open":
            add_style(".s")
        elif child_type == "link_open":
            href = child.attrs.get("href", "")
            action = f"link({href!r})"
            add_style(Style(underline=True) + Style.from_meta({"@click": action}))
        elif child_type == "image":
            href = child.attrs.get("src", "")
            alt = child.attrs.get("alt", "")
            action = f"link({href!r})"
            add_style(Style(underline=True) + Style.from_meta({"@click": action}))
            add_content("🖼  ")
            if alt:
                add_content(f"({alt})")
            if child.children is not None:
                for grandchild in child.children:
                    add_content(grandchild.content)
            close_tag()
        elif child_type.endswith("_close"):
            close_tag()

    content = Content("".join(tokens), spans=spans)
    return content


# ---------------------------------------------------------------------------
# Markdown parser - converts markdown-it tokens into MarkdownBlock objects
# ---------------------------------------------------------------------------

HEADING_STYLES = {
    1: "markdown--h1",
    2: "markdown--h2",
    3: "markdown--h3",
    4: "markdown--h4",
    5: "markdown--h5",
    6: "markdown--h6",
}


def _bq_style_name(depth: int) -> str:
    """Return the component class name for a given blockquote depth."""
    clamped = min(depth, 3)
    return f"markdown--bq-depth-{clamped}"


def _get_list_indent(stack: list[dict]) -> int:
    """Get the indent from the nearest list_item in the stack."""
    for parent in reversed(stack):
        if parent["type"] == "list_item":
            return parent.get("indent", 0)
    return 0


def _parse_tokens(
    tokens: Iterable[Token],
    unhandled_token: Callable[[Token], MarkdownBlock | None] | None = None,
) -> list[MarkdownBlock]:
    """Parse markdown-it tokens into a flat list of MarkdownBlock objects.

    Args:
        tokens: Iterable of markdown-it tokens.
        unhandled_token: Optional callback for unhandled token types.

    Returns:
        A list of MarkdownBlock data objects.
    """

    blocks: list[MarkdownBlock] = []

    # Stack for tracking nested structures
    stack: list[dict] = []
    # For tracking list nesting
    list_stack: list[dict] = []

    heading_slugs = TrackedSlugs()

    for token in tokens:
        token_type = token.type
        source_range = (
            (token.map[0], token.map[1]) if token.map is not None else (0, 0)
        )

        if token_type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
            stack.append(
                {
                    "type": "heading",
                    "level": level,
                    "source_range": source_range,
                }
            )

        elif token_type == "heading_close":
            ctx = stack.pop()
            block_id = ctx.get("block_id")
            content = ctx.get("content", Content(""))
            # Generate the id from the heading text
            slug_text = content.plain
            block_id = f"heading-{slug_for_tcss_id(slug_text)}"

            top_margin = 2 if ctx["level"] <= 2 else 1
            blocks.append(
                MarkdownBlock(
                    block_type="heading",
                    content=content,
                    level=ctx["level"],
                    block_id=block_id,
                    source_range=ctx["source_range"],
                    style_name=HEADING_STYLES.get(ctx["level"], "markdown--h1"),
                    top_margin=top_margin,
                    bottom_margin=1,
                    text_align="center" if ctx["level"] == 1 else "left",
                )
            )

        elif token_type == "paragraph_open":
            stack.append(
                {"type": "paragraph", "source_range": source_range}
            )

        elif token_type == "paragraph_close":
            ctx = stack.pop()
            content = ctx.get("content", Content(""))
            indent = 0
            prefix = ""
            style_name = "markdown--paragraph"
            border_left = ""
            in_list = False
            bq_depth = sum(1 for p in stack if p["type"] == "blockquote")
            if bq_depth > 0:
                border_left = "▌ " * bq_depth
                style_name = _bq_style_name(bq_depth)
                # Also check for list item indent
                for parent in reversed(stack):
                    if parent["type"] == "list_item":
                        indent = parent.get("indent", 0)
                        in_list = True
                        if not parent.get("first_para_done"):
                            prefix = parent.get("prefix", "")
                            parent["first_para_done"] = True
                        break
            else:
                for parent in reversed(stack):
                    if parent["type"] == "list_item":
                        indent = parent.get("indent", 0)
                        in_list = True
                        if not parent.get("first_para_done"):
                            prefix = parent.get("prefix", "")
                            parent["first_para_done"] = True
                        break

            in_blockquote = bq_depth > 0
            blocks.append(
                MarkdownBlock(
                    block_type="paragraph",
                    content=content,
                    source_range=ctx["source_range"],
                    style_name=style_name,
                    bottom_margin=0 if (in_list or in_blockquote) else 1,
                    indent=indent,
                    prefix=prefix,
                    border_left=border_left,
                    bq_depth=bq_depth,
                )
            )

        elif token_type == "blockquote_open":
            bq_depth = sum(1 for p in stack if p["type"] == "blockquote")
            if bq_depth > 0:
                # Check if content was emitted at this depth
                parent_bq = None
                for s in reversed(stack):
                    if s["type"] == "blockquote":
                        parent_bq = s
                        break
                if parent_bq and len(blocks) > parent_bq.get("block_count_at_open", 0):
                    list_indent = _get_list_indent(stack)
                    blocks.append(
                        MarkdownBlock(
                            block_type="blockquote_spacer",
                            content=Content(" "),
                            source_range=source_range,
                            style_name=_bq_style_name(bq_depth),
                            border_left="▌ " * bq_depth,
                            bq_depth=bq_depth,
                            indent=list_indent,
                            bottom_margin=0,
                        )
                    )
            stack.append(
                {
                    "type": "blockquote",
                    "source_range": source_range,
                    "block_count_at_open": len(blocks),
                }
            )

        elif token_type == "blockquote_close":
            stack.pop()
            bq_depth = sum(1 for p in stack if p["type"] == "blockquote")
            if bq_depth > 0:
                list_indent = _get_list_indent(stack)
                blocks.append(
                    MarkdownBlock(
                        block_type="blockquote_spacer",
                        content=Content(" "),
                        source_range=source_range,
                        style_name=_bq_style_name(bq_depth),
                        border_left="▌ " * bq_depth,
                        bq_depth=bq_depth,
                        indent=list_indent,
                        bottom_margin=0,
                    )
                )
            else:
                # Ensure spacing after top-level blockquotes
                if blocks:
                    blocks[-1].bottom_margin = 1

        elif token_type == "bullet_list_open":
            depth = sum(
                1 for s in list_stack if s["type"] in ("bullet_list", "ordered_list")
            )
            list_stack.append(
                {
                    "type": "bullet_list",
                    "depth": depth,
                    "item_count": 0,
                }
            )
            stack.append({"type": "bullet_list", "source_range": source_range})

        elif token_type == "bullet_list_close":
            list_stack.pop()
            stack.pop()
            # Ensure spacing after top-level lists
            if not list_stack and blocks:
                blocks[-1].bottom_margin = 1

        elif token_type == "ordered_list_open":
            depth = sum(
                1 for s in list_stack if s["type"] in ("bullet_list", "ordered_list")
            )
            list_stack.append(
                {
                    "type": "ordered_list",
                    "depth": depth,
                    "item_count": 0,
                    "start": int(token.attrGet("start") or 1),
                }
            )
            stack.append({"type": "ordered_list", "source_range": source_range})

        elif token_type == "ordered_list_close":
            list_stack.pop()
            stack.pop()
            # Ensure spacing after top-level lists
            if not list_stack and blocks:
                blocks[-1].bottom_margin = 1

        elif token_type == "list_item_open":
            if not list_stack:
                continue
            list_ctx = list_stack[-1]
            list_ctx["item_count"] += 1

            depth = list_ctx["depth"]
            indent = 4 + depth * 2

            if list_ctx["type"] == "bullet_list":
                bullet_idx = depth % len(BULLETS)
                prefix = BULLETS[bullet_idx]
            else:
                number = list_ctx["start"] + list_ctx["item_count"] - 1
                prefix = f"{number}. "

            stack.append(
                {
                    "type": "list_item",
                    "indent": indent,
                    "prefix": prefix,
                    "first_para_done": False,
                    "source_range": source_range,
                }
            )

        elif token_type == "list_item_close":
            stack.pop()

        elif token_type == "hr":
            blocks.append(
                MarkdownBlock(
                    block_type="hr",
                    content=Content(""),
                    source_range=source_range,
                    style_name="markdown--hr",
                    top_margin=2,
                    bottom_margin=1,
                )
            )

        elif token_type == "table_open":
            stack.append(
                {
                    "type": "table",
                    "headers": [],
                    "rows": [],
                    "current_row": [],
                    "in_head": False,
                    "source_range": source_range,
                }
            )

        elif token_type == "thead_open":
            if stack and stack[-1]["type"] == "table":
                stack[-1]["in_head"] = True

        elif token_type == "thead_close":
            if stack and stack[-1]["type"] == "table":
                stack[-1]["in_head"] = False

        elif token_type == "tbody_open":
            pass

        elif token_type == "tbody_close":
            pass

        elif token_type == "tr_open":
            if stack and stack[-1]["type"] == "table":
                stack[-1]["current_row"] = []

        elif token_type == "tr_close":
            if stack and stack[-1]["type"] == "table":
                if stack[-1]["in_head"]:
                    stack[-1]["headers"] = stack[-1]["current_row"][:]
                else:
                    stack[-1]["rows"].append(stack[-1]["current_row"][:])
                stack[-1]["current_row"] = []

        elif token_type in ("th_open", "td_open"):
            stack.append({"type": "table_cell"})

        elif token_type in ("th_close", "td_close"):
            ctx = stack.pop()
            content = ctx.get("content", Content(""))
            # Find the parent table
            for parent in reversed(stack):
                if parent["type"] == "table":
                    parent["current_row"].append(content)
                    break

        elif token_type == "table_close":
            ctx = stack.pop()
            headers = ctx.get("headers", [])
            rows = ctx.get("rows", [])
            blocks.extend(
                _build_table_blocks(headers, rows, ctx["source_range"])
            )

        elif token_type == "inline":
            if stack:
                content = _token_to_content(token)
                stack[-1]["content"] = content

        elif token_type in ("fence", "code_block"):
            code = token.content.rstrip()
            language = token.info or ""
            highlighted = highlight(code, language=language or None)

            indent = 0
            prefix = ""
            bq_depth = sum(1 for p in stack if p["type"] == "blockquote")
            border_left = "▌ " * bq_depth if bq_depth > 0 else ""
            for parent in reversed(stack):
                if parent["type"] == "list_item":
                    indent = parent.get("indent", 0)
                    break

            blocks.append(
                MarkdownBlock(
                    block_type="fence",
                    content=highlighted,
                    source_range=source_range,
                    style_name="markdown--fence",
                    top_margin=1,
                    bottom_margin=1,
                    code_language=language,
                    indent=indent,
                    prefix=prefix,
                    padding_top=1,
                    padding_bottom=1,
                    padding_left=2,
                    padding_right=1,
                    border_left=border_left,
                    bq_depth=bq_depth,
                )
            )

        else:
            if unhandled_token is not None:
                external = unhandled_token(token)
                if external is not None:
                    blocks.append(external)

    return blocks


def _build_table_blocks(
    headers: list[Content],
    rows: list[list[Content]],
    source_range: tuple[int, int],
) -> list[MarkdownBlock]:
    """Build MarkdownBlock objects for a table.

    Stores raw table data; actual rendering is done at layout time
    when the available width is known.
    """
    if not headers:
        return []

    return [
        MarkdownBlock(
            block_type="table",
            content=Content(""),
            source_range=source_range,
            style_name="markdown--table",
            bottom_margin=1,
            table_headers=headers,
            table_rows=rows,
        )
    ]


def _wrap_cell_text(text: str, width: int) -> list[str]:
    """Wrap text to fit within width cells (CJK-aware).

    Args:
        text: The text to wrap.
        width: Maximum cell width per line.

    Returns:
        A list of wrapped lines.
    """
    if width <= 0:
        return [text]
    result: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        current_width = 0
        for char in paragraph:
            cw = cell_len(char)
            if current_width + cw > width and current:
                result.append(current)
                current = char
                current_width = cw
            else:
                current += char
                current_width += cw
        result.append(current)
    return result if result else [""]


def _cell_ljust(text: str, target_cells: int) -> str:
    """Left-justify text to a target cell width (CJK-aware)."""
    text_cells = cell_len(text)
    if text_cells >= target_cells:
        return text
    return text + " " * (target_cells - text_cells)


# ---------------------------------------------------------------------------
# Line cache for virtualization
# ---------------------------------------------------------------------------


@dataclass
class _BlockLineInfo:
    """Cached rendering information for a markdown block."""

    block_index: int
    """Index into the blocks list."""
    start_line: int
    """First virtual line occupied by this block."""
    height: int
    """Total height in virtual lines (including margins)."""
    content_height: int
    """Height of the content itself (without margins)."""
    top_margin: int
    """Top margin lines."""
    bottom_margin: int
    """Bottom margin lines."""


NUMERALS = " ⅠⅡⅢⅣⅤⅥ"


class Markdown(ScrollView, can_focus=True):
    """A Markdown widget that uses virtualization for efficient rendering.

    This widget parses markdown and renders it using the Line API,
    only rendering visible lines for optimal performance with large documents.
    """

    COMPONENT_CLASSES = {
        "markdown--h1",
        "markdown--h2",
        "markdown--h3",
        "markdown--h4",
        "markdown--h5",
        "markdown--h6",
        "markdown--paragraph",
        "markdown--fence",
        "markdown--hr",
        "markdown--table",
        "markdown--table-header",
        "markdown--block-quote",
        "markdown--block-quote-border",
        "markdown--bq-depth-1",
        "markdown--bq-depth-2",
        "markdown--bq-depth-3",
        "markdown--bullet",
        "code_inline",
        "em",
        "strong",
        "s",
    }

    DEFAULT_CSS = """
    Markdown {
        color: $foreground;
        overflow-y: auto;
        overflow-x: hidden;
        background: $surface;
        padding: 0 0 0 2;

        & > .markdown--h1 {
            color: $markdown-h1-color;
            background: $markdown-h1-background;
            text-style: $markdown-h1-text-style;
            content-align: center middle;
        }
        & > .markdown--h2 {
            color: $markdown-h2-color;
            background: $markdown-h2-background;
            text-style: $markdown-h2-text-style;
        }
        & > .markdown--h3 {
            color: $markdown-h3-color;
            background: $markdown-h3-background;
            text-style: $markdown-h3-text-style;
        }
        & > .markdown--h4 {
            color: $markdown-h4-color;
            background: $markdown-h4-background;
            text-style: $markdown-h4-text-style;
        }
        & > .markdown--h5 {
            color: $markdown-h5-color;
            background: $markdown-h5-background;
            text-style: $markdown-h5-text-style;
        }
        & > .markdown--h6 {
            color: $markdown-h6-color;
            background: $markdown-h6-background;
            text-style: $markdown-h6-text-style;
        }
        & > .markdown--fence {
            background: black 10%;
            color: rgb(210, 210, 210);
        }
        &:light > .markdown--fence {
            background: white 30%;
        }
        & > .markdown--hr {
            color: $secondary;
        }
        & > .markdown--block-quote {
        }
        &:dark > .markdown--bq-depth-1 {
            background: $foreground 7%;
        }
        &:dark > .markdown--bq-depth-2 {
            background: $foreground 5%;
        }
        &:dark > .markdown--bq-depth-3 {
            background: $foreground 3%;
        }
        &:light > .markdown--bq-depth-1 {
            background: $foreground 5%;
        }
        &:light > .markdown--bq-depth-2 {
            background: $foreground 3%;
        }
        &:light > .markdown--bq-depth-3 {
            background: $foreground 2%;
        }
        &:dark > .markdown--block-quote-border {
            color: $text-primary 50%;
        }
        &:light > .markdown--block-quote-border {
            color: $text-secondary;
        }
        &:dark > .markdown--bullet {
            color: $text-primary;
        }
        &:light > .markdown--bullet {
            color: $text-secondary;
        }
        & > .markdown--table {
        }
        &:light > .markdown--table {
            background: white 30%;
        }
        & > .markdown--table-header {
            color: $primary;
            text-style: bold;
        }
        &:dark > .code_inline {
            background: $warning 10%;
            color: $text-warning 95%;
        }
        &:light > .code_inline {
            background: $error 5%;
            color: $text-error 95%;
        }
        & > .em {
            text-style: italic;
        }
        & > .strong {
            text-style: bold;
        }
        & > .s {
            text-style: strike;
        }
    }
    """

    class TableOfContentsUpdated(Message):
        """The table of contents was updated."""

        def __init__(
            self, markdown: Markdown, table_of_contents: TableOfContentsType
        ) -> None:
            super().__init__()
            self.markdown: Markdown = markdown
            """The `Markdown` widget associated with the table of contents."""
            self.table_of_contents: TableOfContentsType = table_of_contents
            """Table of contents."""

        @property
        def control(self) -> Markdown:
            """The `Markdown` widget associated with the table of contents."""
            return self.markdown

    class TableOfContentsSelected(Message):
        """An item in the TOC was selected."""

        def __init__(self, markdown: Markdown, block_id: str) -> None:
            super().__init__()
            self.markdown: Markdown = markdown
            """The `Markdown` widget where the selected item is."""
            self.block_id: str = block_id
            """ID of the block that was selected."""

        @property
        def control(self) -> Markdown:
            """The `Markdown` widget where the selected item is."""
            return self.markdown

    class LinkClicked(Message):
        """A link in the document was clicked."""

        def __init__(self, markdown: Markdown, href: str) -> None:
            super().__init__()
            self.markdown: Markdown = markdown
            """The `Markdown` widget containing the link clicked."""
            self.href: str = unquote(href)
            """The link that was selected."""

        @property
        def control(self) -> Markdown:
            """The `Markdown` widget containing the link clicked."""
            return self.markdown

    def __init__(
        self,
        markdown: str | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        parser_factory: Callable[[], MarkdownIt] | None = None,
        open_links: bool = True,
    ):
        """A Markdown widget.

        Args:
            markdown: String containing Markdown or None to leave blank for now.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes of the widget.
            parser_factory: A factory function to return a configured MarkdownIt instance. If `None`, a "gfm-like" parser is used.
            open_links: Open links automatically. If you set this to `False`, you can handle the [`LinkClicked`][textual.widgets.markdown.Markdown.LinkClicked] events.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._initial_markdown: str | None = markdown
        self._markdown = ""
        self._parser_factory = parser_factory
        self._table_of_contents: TableOfContentsType | None = None
        self._open_links = open_links
        self._last_parsed_line = 0

        # Virtualization state
        self._blocks: list[MarkdownBlock] = []
        """Parsed markdown blocks."""
        self._block_line_info: list[_BlockLineInfo] = []
        """Line information for each block."""
        self._total_lines: int = 0
        """Total virtual lines."""
        self._line_cache: LRUCache[tuple[int, int], Strip] = LRUCache(maxsize=2048)
        """Cache of rendered strips, keyed by (line_number, width)."""
        self._table_strips: dict[int, list[Strip]] = {}
        """Pre-computed strips for table blocks, keyed by block index."""
        self._width_at_last_layout: int = 0
        """Width when blocks were last laid out."""

    @property
    def table_of_contents(self) -> TableOfContentsType:
        """The document's table of contents."""
        if self._table_of_contents is None:
            self._table_of_contents = [
                (block.level, block.content.plain, block.block_id)
                for block in self._blocks
                if block.block_type == "heading"
            ]
        return self._table_of_contents

    @property
    def source(self) -> str:
        """The markdown source."""
        return self._markdown or ""

    async def _on_mount(self, _: Mount) -> None:
        initial_markdown = self._initial_markdown
        self._initial_markdown = None
        await self.update(initial_markdown or "")

        if initial_markdown is None:
            self.post_message(
                Markdown.TableOfContentsUpdated(
                    self, self._table_of_contents
                ).set_sender(self)
            )

    @classmethod
    def get_stream(cls, markdown: Markdown) -> MarkdownStream:
        """Get a [MarkdownStream][textual.widgets.markdown.MarkdownStream] instance.

        Args:
            markdown: A [Markdown][textual.widgets.Markdown] widget instance.

        Returns:
            The Markdown stream object.
        """
        updater = MarkdownStream(markdown)
        updater.start()
        return updater

    def on_markdown_link_clicked(self, event: LinkClicked) -> None:
        if self._open_links:
            self.app.open_url(event.href)

    @staticmethod
    def sanitize_location(location: str) -> tuple[Path, str]:
        """Given a location, break out the path and any anchor.

        Args:
            location: The location to sanitize.

        Returns:
            A tuple of the path to the location cleaned of any anchor, plus
            the anchor (or an empty string if none was found).
        """
        location, _, anchor = location.partition("#")
        return Path(location), anchor

    def goto_anchor(self, anchor: str) -> bool:
        """Try and find the given anchor in the current document.

        Args:
            anchor: The anchor to try and find.

        Returns:
            True when the anchor was found, False otherwise.
        """
        if self._table_of_contents is None:
            return False
        unique = TrackedSlugs()
        for _, title, header_id in self._table_of_contents:
            if unique.slug(title) == anchor:
                # Find the block and scroll to it
                for info in self._block_line_info:
                    block = self._blocks[info.block_index]
                    if block.block_id == header_id:
                        self.scroll_to(y=info.start_line, animate=False)
                        return True
                return True
        return False

    async def load(self, path: Path) -> None:
        """Load a new Markdown document.

        Args:
            path: Path to the document.

        Raises:
            OSError: If there was some form of error loading the document.
        """
        path, anchor = self.sanitize_location(str(path))
        data = await asyncio.get_running_loop().run_in_executor(
            None, partial(path.read_text, encoding="utf-8")
        )
        await self.update(data)
        if anchor:
            self.goto_anchor(anchor)

    def _build_blocks(self, markdown: str) -> list[MarkdownBlock]:
        """Parse markdown source into blocks.

        Args:
            markdown: Markdown document string.

        Returns:
            A list of MarkdownBlock objects.
        """
        parser = (
            MarkdownIt("gfm-like")
            if self._parser_factory is None
            else self._parser_factory()
        )
        tokens = parser.parse(markdown)
        return _parse_tokens(tokens)

    def _layout_blocks(self) -> None:
        """Compute the line layout for all blocks.

        This calculates the virtual line positions for each block based on
        the current widget width, and updates virtual_size.
        """
        width = self.scrollable_content_region.width
        if width <= 0:
            width = 80  # Fallback

        self._width_at_last_layout = width
        self._block_line_info.clear()
        self._line_cache.clear()
        self._table_strips.clear()

        current_line = 0
        last_bottom_margin = 0

        for index, block in enumerate(self._blocks):
            # Compute top margin (collapse with previous bottom margin)
            top_margin = max(block.top_margin, last_bottom_margin) - last_bottom_margin
            if index == 0:
                top_margin = 1

            # Calculate content height
            border_width = len(block.border_left) if block.border_left else 0
            content_width = width - block.indent - block.padding_left - block.padding_right - border_width
            if content_width <= 0:
                content_width = 1

            if block.block_type == "hr":
                content_height = 1
            elif block.block_type == "table" and block.table_headers is not None:
                table_strips = self._build_table_strips(block, content_width)
                self._table_strips[index] = table_strips
                content_height = len(table_strips)
            else:
                content_height = block.content.get_height({}, content_width)

            # Add padding to content height
            content_height += block.padding_top + block.padding_bottom

            total_height = top_margin + content_height + block.bottom_margin

            self._block_line_info.append(
                _BlockLineInfo(
                    block_index=index,
                    start_line=current_line,
                    height=total_height,
                    content_height=content_height,
                    top_margin=top_margin,
                    bottom_margin=block.bottom_margin,
                )
            )

            current_line += total_height
            last_bottom_margin = block.bottom_margin

        self._total_lines = current_line + 1  # +1 for bottom spacing
        self.virtual_size = Size(width, self._total_lines)

    def _find_block_at_line(self, line: int) -> tuple[int, _BlockLineInfo] | None:
        """Find which block contains a given virtual line.

        Uses binary search for efficiency.

        Args:
            line: Virtual line number.

        Returns:
            Tuple of (block_index, BlockLineInfo) or None.
        """
        if not self._block_line_info:
            return None

        # Binary search
        infos = self._block_line_info
        lo, hi = 0, len(infos) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            info = infos[mid]
            if line < info.start_line:
                hi = mid - 1
            elif line >= info.start_line + info.height:
                lo = mid + 1
            else:
                return mid, info
        return None

    def _render_block_line(
        self, block: MarkdownBlock, info: _BlockLineInfo, line: int, width: int
    ) -> Strip:
        """Render a single virtual line from a block.

        Args:
            block: The markdown block.
            info: Line info for the block.
            line: The virtual line number.
            width: Width to render at.

        Returns:
            A Strip for the line.
        """
        # Determine where within the block this line falls
        local_line = line - info.start_line

        base_style = self.visual_style
        block_style = self._get_block_style(block)

        # Top margin (uses base/parent style, not block style)
        if local_line < info.top_margin:
            return Strip.blank(width, base_style.rich_style)

        # Bottom margin (uses base/parent style, not block style)
        content_end = info.top_margin + info.content_height
        if local_line >= content_end:
            return Strip.blank(width, base_style.rich_style)

        # Content area line (relative to content start, includes padding)
        content_line = local_line - info.top_margin

        # Top padding (blank line with block style/background)
        if content_line < block.padding_top:
            return self._render_padding_line(block, base_style, block_style, width)

        # Bottom padding (blank line with block style/background)
        actual_content_end = info.content_height - block.padding_bottom
        if content_line >= actual_content_end:
            return self._render_padding_line(block, base_style, block_style, width)

        # Actual content line (within padding)
        actual_line = content_line - block.padding_top

        if block.block_type == "hr":
            # Horizontal rule
            rule_char = "─"
            return Strip(
                [Segment(rule_char * width, block_style.rich_style)],
                width,
            )

        # Render the content
        content = block.content
        border_width = len(block.border_left) if block.border_left else 0
        content_width = width - block.indent - block.padding_left - block.padding_right - border_width
        if content_width <= 0:
            content_width = 1

        # Table blocks use pre-computed strips
        if block.block_type == "table" and info.block_index in self._table_strips:
            table_strips = self._table_strips[info.block_index]
            if actual_line < len(table_strips):
                strip = table_strips[actual_line]
            else:
                strip = Strip.blank(content_width, block_style.rich_style)
        else:
            render_options = RenderOptions(
                self._get_style,
                self.styles,
            )
            strips = content.render_strips(
                content_width,
                None,
                block_style,
                render_options,
            )

            if actual_line < len(strips):
                strip = strips[actual_line]
            else:
                strip = Strip.blank(content_width, block_style.rich_style)

        # Center-align content for headings (e.g., H1)
        if block.text_align == "center":
            strip_len = strip.cell_length
            if strip_len < content_width:
                pad_left = (content_width - strip_len) // 2
                pad_right = content_width - strip_len - pad_left
                strip = Strip(
                    [Segment(" " * pad_left, block_style.rich_style)]
                    + strip._segments
                    + [Segment(" " * pad_right, block_style.rich_style)],
                    content_width,
                )

        # Apply indent, border_left, prefix, and padding_left
        left_offset = block.indent + block.padding_left + border_width
        if left_offset > 0 or block.prefix:
            segments: list[Segment] = []
            # Indent area uses base style (not block style) so block
            # backgrounds (e.g. code fence) don't bleed into the indent
            indent_style = base_style.rich_style
            if block.indent > 0:
                indent_width = block.indent
                if actual_line == 0 and block.prefix:
                    prefix_text = block.prefix
                    prefix_len = len(prefix_text)
                    # Get bullet style if available
                    bullet_style = self.get_visual_style("markdown--bullet")
                    pad = max(0, indent_width - prefix_len)
                    segments.append(Segment(" " * pad, indent_style))
                    segments.append(Segment(prefix_text, bullet_style.rich_style))
                else:
                    segments.append(Segment(" " * indent_width, indent_style))
            if block.bq_depth > 0:
                segments.extend(self._render_bq_border_segments(block.bq_depth))
            elif block.border_left:
                segments.append(Segment(block.border_left, block_style.rich_style))
            if block.padding_left > 0:
                segments.append(Segment(" " * block.padding_left, block_style.rich_style))
            segments.extend(strip._segments)
            if block.padding_right > 0:
                segments.append(Segment(" " * block.padding_right, block_style.rich_style))
            strip = Strip(segments)

        # Pad strip to full width — use a style without text decorations
        # so underlines etc. don't extend into the padding area
        pad_rich_style = block_style.rich_style
        if pad_rich_style.underline or pad_rich_style.overline or pad_rich_style.strike:
            pad_rich_style = RichStyle(
                color=pad_rich_style.color,
                bgcolor=pad_rich_style.bgcolor,
                bold=pad_rich_style.bold,
                dim=pad_rich_style.dim,
                italic=pad_rich_style.italic,
            )
        strip = strip.extend_cell_length(width, pad_rich_style)

        # Apply link metadata
        if block.block_type in ("paragraph", "heading", "table"):
            meta = {"block": block.block_type}
            strip = strip.apply_meta(meta)

        return strip

    def _get_block_style(self, block: MarkdownBlock) -> Style:
        """Get the visual style for a block.

        Args:
            block: The markdown block.

        Returns:
            A Style instance.
        """
        if block.style_name:
            return self.get_visual_style(block.style_name)
        return self.visual_style

    def _render_bq_border_segments(self, bq_depth: int) -> list[Segment]:
        """Render blockquote border segments with per-depth backgrounds.

        Each ▌ uses the border style, and the space after each ▌ uses
        that depth level's background style for a layered visual effect.
        """
        bq_border_style = self.get_visual_style("markdown--block-quote-border")
        segments: list[Segment] = []
        for d in range(1, bq_depth + 1):
            segments.append(Segment("▌", bq_border_style.rich_style))
            depth_bg_style = self.get_visual_style(_bq_style_name(d))
            segments.append(Segment(" ", depth_bg_style.rich_style))
        return segments

    def _render_padding_line(
        self, block: MarkdownBlock, base_style: Style, block_style: Style, width: int
    ) -> Strip:
        """Render a padding line (e.g. top/bottom padding of a code fence).

        These lines use the exact same left-side construction as content lines
        so that indent and padding_left are consistent.
        """
        border_width = len(block.border_left) if block.border_left else 0
        left_offset = block.indent + block.padding_left + border_width
        if left_offset > 0:
            segments: list[Segment] = []
            if block.indent > 0:
                segments.append(Segment(" " * block.indent, base_style.rich_style))
            if block.bq_depth > 0:
                segments.extend(self._render_bq_border_segments(block.bq_depth))
            elif block.border_left:
                segments.append(Segment(block.border_left, block_style.rich_style))
            if block.padding_left > 0:
                segments.append(Segment(" " * block.padding_left, block_style.rich_style))
            remaining = width - left_offset
            if remaining > 0:
                segments.append(Segment(" " * remaining, block_style.rich_style))
            return Strip(segments, width)
        return Strip.blank(width, block_style.rich_style)

    def _build_table_strips(
        self, block: MarkdownBlock, content_width: int
    ) -> list[Strip]:
        """Build pre-rendered strips for a table block.

        Produces a fully-bordered table with cell wrapping, styled headers,
        and thin box-drawing keylines matching the old grid-based rendering.
        """
        headers = block.table_headers or []
        rows = block.table_rows or []
        if not headers:
            return []

        col_count = len(headers)
        cell_pad = 1  # 1 space padding on each side of cell content

        # Styles
        block_style = self._get_block_style(block)
        header_style = self.get_visual_style("markdown--table-header")
        cell_rs = block_style.rich_style
        header_rs = header_style.rich_style
        # Border style: use foreground at reduced intensity
        border_rs = RichStyle(
            color=block_style.rich_style.color,
            bgcolor=block_style.rich_style.bgcolor,
            dim=True,
        )

        # Calculate overhead: │ pad content pad │ pad content pad │ ...
        # = 1 (left border) + col_count * (cell_pad + col_width + cell_pad) + (col_count - 1) * 1 (separators) + 1 (right border)
        # = 2 + col_count * (2 * cell_pad) + (col_count - 1)
        # = col_count * (2 * cell_pad + 1) + 1
        overhead = col_count * (2 * cell_pad + 1) + 1
        available = content_width - overhead
        if available < col_count:
            available = col_count

        # Natural column widths (CJK-aware)
        nat_widths = [max(cell_len(h.plain), 1) for h in headers]
        for row in rows:
            for i, cell_content in enumerate(row):
                if i < col_count:
                    nat_widths[i] = max(nat_widths[i], cell_len(cell_content.plain))

        total_nat = sum(nat_widths)
        if total_nat <= available:
            col_widths = nat_widths[:]
        else:
            # Shrink proportionally
            col_widths = [max(1, int(w * available / total_nat)) for w in nat_widths]
            # Fix rounding errors
            diff = available - sum(col_widths)
            for i in range(abs(diff)):
                idx = i % col_count
                if diff > 0:
                    col_widths[idx] += 1
                elif col_widths[idx] > 1:
                    col_widths[idx] -= 1

        # Wrap cell text
        wrapped_headers = [
            _wrap_cell_text(h.plain, col_widths[i])
            for i, h in enumerate(headers)
        ]
        header_height = max(
            (len(lines) for lines in wrapped_headers), default=1
        )

        wrapped_rows: list[list[list[str]]] = []
        row_heights: list[int] = []
        for row in rows:
            wrapped_row = []
            for i in range(col_count):
                if i < len(row):
                    wrapped_row.append(
                        _wrap_cell_text(row[i].plain, col_widths[i])
                    )
                else:
                    wrapped_row.append([""])
            rh = max((len(lines) for lines in wrapped_row), default=1)
            wrapped_rows.append(wrapped_row)
            row_heights.append(rh)

        # Build strips
        strips: list[Strip] = []

        def h_border(left: str, mid: str, right: str) -> Strip:
            segs: list[Segment] = [Segment(left, border_rs)]
            for i, w in enumerate(col_widths):
                segs.append(Segment("─" * (w + 2 * cell_pad), border_rs))
                if i < col_count - 1:
                    segs.append(Segment(mid, border_rs))
            segs.append(Segment(right, border_rs))
            return Strip(segs)

        def data_row_strips(
            wrapped_cells: list[list[str]], height: int, is_header: bool
        ) -> list[Strip]:
            text_rs = header_rs if is_header else cell_rs
            result: list[Strip] = []
            for line_idx in range(height):
                segs: list[Segment] = [Segment("│", border_rs)]
                for col_idx in range(col_count):
                    cell_lines = wrapped_cells[col_idx]
                    if line_idx < len(cell_lines):
                        text = cell_lines[line_idx]
                        padded = _cell_ljust(text, col_widths[col_idx])
                    else:
                        padded = " " * col_widths[col_idx]
                    segs.append(Segment(" " * cell_pad, cell_rs))
                    segs.append(Segment(padded, text_rs))
                    segs.append(Segment(" " * cell_pad, cell_rs))
                    segs.append(Segment("│", border_rs))
                result.append(Strip(segs))
            return result

        # Top border
        strips.append(h_border("┌", "┬", "┐"))
        # Header
        strips.extend(data_row_strips(wrapped_headers, header_height, True))
        # Header separator
        strips.append(h_border("├", "┼", "┤"))
        # Data rows with separators between them
        for row_idx, (wrapped_row, rh) in enumerate(
            zip(wrapped_rows, row_heights)
        ):
            strips.extend(data_row_strips(wrapped_row, rh, False))
            if row_idx < len(wrapped_rows) - 1:
                strips.append(h_border("├", "┼", "┤"))
        # Bottom border
        strips.append(h_border("└", "┴", "┘"))

        return strips

    def render_line(self, y: int) -> Strip:
        """Render a line of content for the Line API.

        Args:
            y: Y coordinate of line relative to the scroll view.

        Returns:
            A rendered line.
        """
        width = self.scrollable_content_region.width
        if width <= 0:
            return Strip.blank(0)

        # Recompute layout if width changed
        if width != self._width_at_last_layout and self._blocks:
            self._layout_blocks()

        line_number = self.scroll_offset.y + y

        # Check cache
        cache_key = (line_number, width)
        cached = self._line_cache.get(cache_key)
        if cached is not None:
            return cached

        if line_number >= self._total_lines or line_number < 0:
            strip = Strip.blank(width, self.visual_style.rich_style)
            self._line_cache[cache_key] = strip
            return strip

        result = self._find_block_at_line(line_number)
        if result is None:
            strip = Strip.blank(width, self.visual_style.rich_style)
            self._line_cache[cache_key] = strip
            return strip

        _block_idx, info = result
        block = self._blocks[info.block_index]
        strip = self._render_block_line(block, info, line_number, width)
        self._line_cache[cache_key] = strip
        return strip

    def render_lines(self, crop: Region) -> list[Strip]:
        """Render visible lines."""
        if self._blocks and self._width_at_last_layout != self.scrollable_content_region.width:
            self._layout_blocks()
        return super().render_lines(crop)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self.virtual_size.width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.virtual_size.height

    def update(self, markdown: str) -> AwaitComplete:
        """Update the document with new Markdown.

        Args:
            markdown: A string containing Markdown.

        Returns:
            An optionally awaitable object.
        """
        self._markdown = markdown
        self._table_of_contents = None
        self._line_cache.clear()
        self._table_strips.clear()

        async def await_update() -> None:
            async with self.lock:
                blocks = await asyncio.get_running_loop().run_in_executor(
                    None, self._build_blocks, markdown
                )
                self._blocks = blocks
                self._layout_blocks()

                lines = markdown.splitlines()
                self._last_parsed_line = len(lines) - (
                    1 if lines and lines[-1] else 0
                )
                self.refresh()
                self.post_message(
                    Markdown.TableOfContentsUpdated(
                        self, self.table_of_contents
                    ).set_sender(self)
                )

        return AwaitComplete(await_update())

    def append(self, markdown: str) -> AwaitComplete:
        """Append markdown to the document.

        Args:
            markdown: A fragment of markdown to be appended.

        Returns:
            An optionally awaitable object.
        """
        self._markdown = self.source + markdown
        self._table_of_contents = None
        self._line_cache.clear()
        self._table_strips.clear()

        async def await_append() -> None:
            async with self.lock:
                blocks = await asyncio.get_running_loop().run_in_executor(
                    None, self._build_blocks, self._markdown
                )
                self._blocks = blocks
                self._layout_blocks()

                lines = self._markdown.splitlines()
                self._last_parsed_line = len(lines) - (
                    1 if lines and lines[-1] else 0
                )
                self.refresh()

                any_headers = any(
                    block.block_type == "heading" for block in blocks
                )
                if any_headers:
                    self.post_message(
                        Markdown.TableOfContentsUpdated(
                            self, self.table_of_contents
                        ).set_sender(self)
                    )

        return AwaitComplete(await_append())

    def scroll_to_block_id(self, block_id: str) -> None:
        """Scroll to a block by its ID.

        Args:
            block_id: The block ID to scroll to.
        """
        for info in self._block_line_info:
            block = self._blocks[info.block_index]
            if block.block_id == block_id:
                self.scroll_to(y=info.start_line, animate=False)
                return

    async def action_link(self, href: str) -> None:
        """Called on link click."""
        self.post_message(Markdown.LinkClicked(self, href))


class MarkdownTableOfContents(Widget, can_focus_children=True):
    """Displays a table of contents for a markdown document."""

    DEFAULT_CSS = """
    MarkdownTableOfContents {
        width: auto;
        height: 1fr;
        background: $panel;
        &:focus-within {
            background-tint: $foreground 5%;
        }
    }
    MarkdownTableOfContents > Tree {
        padding: 1;
        width: auto;
        height: 1fr;
        background: $panel;
    }
    """

    table_of_contents = reactive[Optional[TableOfContentsType]](None, init=False)
    """Underlying data to populate the table of contents widget."""

    def __init__(
        self,
        markdown: Markdown,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Initialize a table of contents.

        Args:
            markdown: The Markdown document associated with this table of contents.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes for the widget.
            disabled: Whether the widget is disabled or not.
        """
        self.markdown: Markdown = markdown
        """The Markdown document associated with this table of contents."""
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)

    def compose(self) -> ComposeResult:
        tree: Tree = Tree("TOC")
        tree.show_root = False
        tree.show_guides = True
        tree.guide_depth = 4
        tree.auto_expand = False
        yield tree

    def watch_table_of_contents(self, table_of_contents: TableOfContentsType) -> None:
        """Triggered when the table of contents changes."""
        self.rebuild_table_of_contents(table_of_contents)

    def rebuild_table_of_contents(self, table_of_contents: TableOfContentsType) -> None:
        """Rebuilds the tree representation of the table of contents data.

        Args:
            table_of_contents: Table of contents.
        """
        tree = self.query_one(Tree)
        tree.clear()
        root = tree.root
        for level, name, block_id in table_of_contents:
            node = root
            for _ in range(level - 1):
                if node._children:
                    node = node._children[-1]
                    node.expand()
                    node.allow_expand = True
                else:
                    node = node.add(NUMERALS[level], expand=True)
            node_label = Text.assemble((f"{NUMERALS[level]} ", "dim"), name)
            node.add_leaf(node_label, {"block_id": block_id})

    async def _on_tree_node_selected(self, message: Tree.NodeSelected) -> None:
        node_data = message.node.data
        if node_data is not None:
            await self._post_message(
                Markdown.TableOfContentsSelected(self.markdown, node_data["block_id"])
            )
        message.stop()


class MarkdownViewer(Widget, can_focus=False, can_focus_children=True):
    """A Markdown viewer widget."""

    SCOPED_CSS = False

    DEFAULT_CSS = """
    MarkdownViewer {
        height: 1fr;
        layout: horizontal;
        background: $surface;
        & > Markdown {
            width: 1fr;
            height: 1fr;
        }
        & > MarkdownTableOfContents {
            display: none;
            dock:left;
        }
    }

    MarkdownViewer.-show-table-of-contents > MarkdownTableOfContents {
        display: block;
    }
    """

    show_table_of_contents = reactive(True)
    """Show the table of contents?"""
    top_block = reactive("")

    navigator: var[Navigator] = var(Navigator)

    class NavigatorUpdated(Message):
        """Navigator has been changed (clicked link etc)."""

    def __init__(
        self,
        markdown: str | None = None,
        *,
        show_table_of_contents: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        parser_factory: Callable[[], MarkdownIt] | None = None,
        open_links: bool = True,
    ):
        """Create a Markdown Viewer object.

        Args:
            markdown: String containing Markdown, or None to leave blank.
            show_table_of_contents: Show a table of contents in a sidebar.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes of the widget.
            parser_factory: A factory function to return a configured MarkdownIt instance. If `None`, a "gfm-like" parser is used.
            open_links: Open links automatically. If you set this to `False`, you can handle the [`LinkClicked`][textual.widgets.markdown.Markdown.LinkClicked] events.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.show_table_of_contents = show_table_of_contents
        self._markdown = markdown
        self._parser_factory = parser_factory
        self._open_links = open_links

    @property
    def document(self) -> Markdown:
        """The [`Markdown`][textual.widgets.Markdown] document widget."""
        return self.query_one(Markdown)

    @property
    def table_of_contents(self) -> MarkdownTableOfContents:
        """The table of contents widget."""
        return self.query_one(MarkdownTableOfContents)

    async def _on_mount(self, _: Mount) -> None:
        await self.document.update(self._markdown or "")

    async def go(self, location: str | PurePath) -> None:
        """Navigate to a new document path."""
        path, anchor = self.document.sanitize_location(str(location))
        if path == Path(".") and anchor:
            self.document.goto_anchor(anchor)
        else:
            await self.document.load(self.navigator.go(location))
            self.post_message(self.NavigatorUpdated())

    async def back(self) -> None:
        """Go back one level in the history."""
        if self.navigator.back():
            await self.document.load(self.navigator.location)
            self.post_message(self.NavigatorUpdated())

    async def forward(self) -> None:
        """Go forward one level in the history."""
        if self.navigator.forward():
            await self.document.load(self.navigator.location)
            self.post_message(self.NavigatorUpdated())

    async def _on_markdown_link_clicked(self, message: Markdown.LinkClicked) -> None:
        message.stop()
        await self.go(message.href)

    def watch_show_table_of_contents(self, show_table_of_contents: bool) -> None:
        self.set_class(show_table_of_contents, "-show-table-of-contents")

    def compose(self) -> ComposeResult:
        markdown = Markdown(
            parser_factory=self._parser_factory, open_links=self._open_links
        )
        yield markdown
        yield MarkdownTableOfContents(markdown)

    def _on_markdown_table_of_contents_updated(
        self, message: Markdown.TableOfContentsUpdated
    ) -> None:
        self.query_one(MarkdownTableOfContents).table_of_contents = (
            message.table_of_contents
        )
        message.stop()

    def _on_markdown_table_of_contents_selected(
        self, message: Markdown.TableOfContentsSelected
    ) -> None:
        self.document.scroll_to_block_id(message.block_id)
        message.stop()
