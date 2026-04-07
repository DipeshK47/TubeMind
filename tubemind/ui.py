"""Server-rendered UI builders for the board-based TubeMind app.

Sprint 4 — UI Overhaul (Shrutika Yadav)
----------------------------------------
Replaced the old dual-screen seed/query interface with a single
ChatGPT-style chat experience. The layout mirrors modern AI chat apps:
- Left sidebar for board/topic switching
- Right panel is a full-height chat window
- Messages scroll in the middle
- Input bar is fixed at the bottom
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fasthtml.common import (
    A, Button, Div, Form, H1, H2, H3, Iframe, Img, Input,
    Label, Option, P, Pre, Script, Select, Span, Textarea, Title,
)

from tubemind.auth import (
    ERROR_MESSAGES, begin_oauth_session, google_auth_url,
    list_note_chunks, list_note_queries,
)
from tubemind.config import (
    DEFAULT_QUERY_MODE, DEMO_AUTH_ENABLED,
    GOOGLE_AUTH_ENABLED, QUERY_MODE_LABELS,
)
from tubemind.models import BoardWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, limit: int = 220) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def format_timestamp(ms: int) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000).strftime("%b %d, %Y at %I:%M %p")


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------

def render_user_badge(user: dict[str, Any]) -> Any:
    avatar = (
        Img(src=user["picture"], cls="user-avatar", alt=user["name"])
        if user.get("picture")
        else Span(
            (user.get("name") or user.get("email") or "U")[:1].upper(),
            cls="user-avatar user-avatar-fallback",
        )
    )
    return Div(
        avatar,
        Span(user.get("name") or user.get("email"), cls="user-name"),
        A("Logout", href="/logout", cls="logout-link"),
        cls="user-badge",
    )


def render_theme_toggle() -> Any:
    return Button(
        Span("Theme", cls="theme-toggle-copy"),
        Span(
            Span("", cls="theme-toggle-knob"),
            cls="theme-toggle-track",
            **{"aria-hidden": "true"},
        ),
        Span("Light", cls="theme-toggle-state"),
        cls="theme-toggle",
        type="button",
        **{
            "data-theme-toggle": "",
            "aria-label": "Switch to dark mode",
            "aria-pressed": "false",
            "title": "Switch to dark mode",
        },
    )


def render_page_topbar(user: Optional[dict[str, Any]] = None) -> Any:
    actions = [render_theme_toggle()]
    if user:
        actions.append(render_user_badge(user))
    return Div(
        Div(*actions, cls="topbar-actions"),
        cls="page-topbar",
    )


# ---------------------------------------------------------------------------
# Login page (unchanged)
# ---------------------------------------------------------------------------

def render_login_page(session, error: str = "") -> Any:
    error_msg = ERROR_MESSAGES.get(error, "")
    actions: list[Any] = []

    if GOOGLE_AUTH_ENABLED:
        state = begin_oauth_session(session)
        actions.append(
            A("Sign in with Google", href=google_auth_url(state), role="button", cls="signin-btn")
        )
    if DEMO_AUTH_ENABLED:
        actions.append(
            A("Enter Demo Workspace", href="/auth/demo", role="button", cls="signin-btn signin-btn-secondary")
        )

    login_copy = "Ask a question, keep the evidence, and let each board stay anchored to one evolving topic."
    if DEMO_AUTH_ENABLED and not GOOGLE_AUTH_ENABLED:
        login_copy = "Demo mode is enabled. Enter the workspace without Google OAuth."
    elif DEMO_AUTH_ENABLED and GOOGLE_AUTH_ENABLED:
        login_copy = "Use Google sign-in or enter the demo workspace."

    return Title("TubeMind - Sign in"), Div(
        render_page_topbar(),
        Div(
            Div(
                Span("Research boards for YouTube knowledge", cls="login-badge"),
                H2("TubeMind", cls="login-title"),
                P(login_copy, cls="login-copy"),
                Div(error_msg, cls="login-error") if error_msg else "",
                Div(*actions, cls="login-actions") if actions else P(
                    "No login method configured. Set DEMO_AUTH_ENABLED=true or configure Google OAuth.",
                    cls="login-copy",
                ),
                P(
                    "Built for slower thinking, clearer summaries, and source-backed notes that still feel easy to scan.",
                    cls="login-footnote",
                ),
                cls="login-card",
            ),
            cls="login-shell",
        ),
        cls="app-shell app-shell-login",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(boards: list[dict[str, Any]], active_board_id: int | None) -> Any:
    board_links = [
        A(
            Div(
                P(str(board.get("title", "") or "Untitled board"), cls="sidebar-board-title"),
                P(
                    str(
                        board.get("summary", "")
                        or format_timestamp(int(board.get("updated_at", 0) or 0))
                        or "No notes yet."
                    ),
                    cls="sidebar-board-copy",
                ),
                cls=f"sidebar-board {'is-active' if int(board.get('id', 0) or 0) == int(active_board_id or 0) else ''}",
            ),
            href=f"/boards/{int(board.get('id', 0) or 0)}",
            cls="sidebar-board-link",
        )
        for board in boards
    ]

    return Div(
        Div(
            Span("TubeMind", cls="sidebar-brand"),
            P("Topic-bound boards", cls="sidebar-copy"),
            cls="sidebar-head",
        ),
        Form(
            Button("+ New Board", type="submit", cls="sidebar-create-btn"),
            _hx_post="/api/boards",
            _hx_target="#workspace-root",
            _hx_swap="outerHTML",
        ),
        (
            Div(*board_links, cls="sidebar-board-list")
            if board_links
            else Div(
                P("Ask your first question or create a board to get started.", cls="sidebar-empty-copy"),
                cls="sidebar-empty",
            )
        ),
        cls="sidebar-shell",
    )


# ---------------------------------------------------------------------------
# ✨ NEW Chat UI — Sprint 4 (Shrutika Yadav)
# ---------------------------------------------------------------------------

def render_chat_bubble_user(question: str) -> Any:
    """Right-aligned user message bubble."""
    return Div(
        Div(
            P(question, cls="cb-text"),
            cls="cb cb-user",
        ),
        cls="cb-row cb-row-user",
    )


def render_chat_bubble_bot(note: dict[str, Any]) -> Any:
    """Left-aligned TubeMind answer bubble with source chips."""
    note_id = int(note.get("id", 0) or 0)
    chunk_count = len(list_note_chunks(note_id))
    created = format_timestamp(int(note.get("created_at", 0) or 0))

    return Div(
        Div(
            Div(
                Span("🎬", cls="cb-icon"),
                Span("TubeMind", cls="cb-name"),
                cls="cb-header",
            ),
            P(str(note.get("answer", "") or ""), cls="cb-text"),
            Div(
                Span(f"📼  {chunk_count} source clip(s)", cls="cb-chip"),
                Span(created, cls="cb-chip cb-chip-muted") if created else "",
                A("View sources →", href=f"/notes/{note_id}", cls="cb-chip cb-chip-link"),
                cls="cb-footer",
            ),
            cls="cb cb-bot",
        ),
        cls="cb-row cb-row-bot",
    )


def render_chat_thread(notes: list[dict[str, Any]]) -> Any:
    """Full scrollable conversation thread."""
    if not notes:
        return Div(
            Div(
                P("🎬", cls="chat-empty-icon"),
                P("Ask anything about YouTube videos", cls="chat-empty-title"),
                P(
                    "TubeMind searches relevant YouTube videos, reads their transcripts, "
                    "and gives you a cited answer with timestamps — "
                    "so you don't have to watch hours of content.",
                    cls="chat-empty-sub",
                ),
                cls="chat-empty-state",
            ),
            cls="chat-thread",
            id="chat-thread",
        )

    rows = []
    for note in notes:
        rows.append(render_chat_bubble_user(str(note.get("question", "") or "")))
        rows.append(render_chat_bubble_bot(note))
    return Div(*rows, cls="chat-thread", id="chat-thread")


def render_chat_input(active_board: Optional[dict[str, Any]]) -> Any:
    """Sticky composer bar at the bottom — Enter sends, Shift+Enter = newline."""
    board_id_val = str(int(active_board.get("id", 0) or 0)) if active_board else ""

    return Div(
        Form(
            Input(type="hidden", name="board_id", value=board_id_val),
            Input(type="hidden", name="mode", value=DEFAULT_QUERY_MODE),
            Div(
                Textarea(
                    "",
                    name="question",
                    placeholder="Message TubeMind…",
                    rows=1,
                    id="tm-input",
                    cls="tm-textarea",
                    **{"onkeydown": "tmKey(event)"},
                ),
                Button("↑", type="submit", cls="tm-send", title="Send"),
                cls="tm-composer",
            ),
            Span("⏳ Searching YouTube…", cls="htmx-indicator tm-thinking"),
            _hx_post="/api/questions",
            _hx_target="#workspace-root",
            _hx_swap="outerHTML",
            _hx_indicator=".tm-thinking",
        ),
        P("TubeMind searches YouTube videos and cites timestamps.", cls="tm-disclaimer"),
        Script("""
function tmKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        e.target.closest('form').requestSubmit();
    }
}
(function () {
    function boot() {
        var ta = document.getElementById('tm-input');
        if (!ta) return;
        ta.style.height = 'auto';
        ta.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 180) + 'px';
        });
        ta.focus();
    }
    function scrollDown() {
        var t = document.getElementById('chat-thread');
        if (t) t.scrollTop = t.scrollHeight;
    }
    document.addEventListener('DOMContentLoaded', function () { boot(); scrollDown(); });
    document.addEventListener('htmx:afterSwap',   function () { boot(); scrollDown(); });
}());
"""),
        cls="tm-input-area",
    )


# render_question_form is called by routes.py — keep the name
def render_question_form(active_board: Optional[dict[str, Any]]) -> Any:
    return render_chat_input(active_board)


# ---------------------------------------------------------------------------
# Main workspace — full-height chat layout
# ---------------------------------------------------------------------------

def render_workspace(workspace: BoardWorkspace, user: dict[str, Any]) -> Any:
    board = workspace.active_board
    board_name = str(board.get("title", "") or "New board") if board else "TubeMind"
    board_status = str(board.get("status", "") or "").upper() if board else ""

    notice_block = Div(workspace.notice, cls="notice-banner") if workspace.notice else ""
    warning_block = Div(workspace.warning, cls="warning-banner") if workspace.warning else ""

    return Div(
        render_page_topbar(user),
        Div(
            # ── Sidebar ────────────────────────────────────────────────────
            render_sidebar(
                workspace.boards,
                int(board.get("id", 0) or 0) if board else None,
            ),
            # ── Chat window ────────────────────────────────────────────────
            Div(
                # top bar inside chat window
                Div(
                    Div(
                        Span(board_status, cls="cw-status") if board_status else "",
                        Span(board_name, cls="cw-title"),
                        cls="cw-title-group",
                    ),
                    cls="cw-topbar",
                ),
                notice_block,
                warning_block,
                # scrollable messages
                render_chat_thread(workspace.notes),
                # sticky input
                render_chat_input(board),
                cls="chat-window",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
        id="workspace-root",
    )


# ---------------------------------------------------------------------------
# Note detail page (unchanged)
# ---------------------------------------------------------------------------

def render_note_detail_page(
    user: dict[str, Any],
    boards: list[dict[str, Any]],
    note: dict[str, Any],
) -> Any:
    board = note.get("board") or {}
    chunks = list_note_chunks(int(note.get("id", 0) or 0))
    queries = list_note_queries(int(note.get("id", 0) or 0))

    query_items = [
        Div(
            P(str(item.get("youtube_query", "") or ""), cls="detail-query"),
            P(str(item.get("reason", "") or "Generated to extend the board corpus."), cls="detail-query-reason"),
            cls="detail-query-card",
        )
        for item in queries
    ]

    chunk_items = [
        Div(
            Div(
                Span(chunk.get("start_label", "") or "0:00", cls="chunk-time"),
                A(
                    chunk.get("video_title", "") or "Source video",
                    href=chunk.get("source_url", "#"),
                    target="_blank",
                    rel="noreferrer",
                    cls="chunk-video-link",
                ),
                cls="chunk-head",
            ),
            (
                Iframe(
                    src=chunk.get("embed_url", ""),
                    title=f"Video source for chunk {index}",
                    loading="lazy",
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share",
                    allowfullscreen="true",
                    cls="chunk-embed-frame",
                )
                if chunk.get("embed_url")
                else ""
            ),
            Div(
                A("Watch source", href=chunk.get("source_url", "#"), target="_blank", rel="noreferrer", cls="chunk-open-link"),
                A("Open board", href=f"/boards/{int(board.get('id', 0) or 0)}", cls="chunk-open-link"),
                cls="chunk-actions",
            ),
            Pre(str(chunk.get("content", "") or ""), cls="chunk-copy"),
            cls="chunk-card",
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    return Div(
        render_page_topbar(user),
        Div(
            render_sidebar(boards, int(board.get("id", 0) or 0)),
            Div(
                Div(
                    A("← Back to board", href=f"/boards/{int(board.get('id', 0) or 0)}", cls="back-link"),
                    H1(str(note.get("question", "") or ""), cls="detail-title"),
                    P(f"Asked {format_timestamp(int(note.get('created_at', 0) or 0))}", cls="detail-meta"),
                    cls="detail-head",
                ),
                Div(
                    H3("Answer", cls="detail-section-title"),
                    Pre(str(note.get("answer", "") or ""), cls="detail-answer"),
                    cls="detail-panel",
                ),
                Div(
                    H3("Generated YouTube queries", cls="detail-section-title"),
                    (
                        Div(*query_items, cls="detail-query-list")
                        if query_items
                        else P("TubeMind answered from the existing board corpus.", cls="detail-muted")
                    ),
                    cls="detail-panel",
                ),
                Div(
                    H3("Supporting video clips", cls="detail-section-title"),
                    (
                        Div(*chunk_items, cls="chunk-list")
                        if chunk_items
                        else P("No chunk previews found for this note.", cls="detail-muted")
                    ),
                    cls="detail-panel",
                ),
                cls="detail-main",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
    )