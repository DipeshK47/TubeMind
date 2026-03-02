from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from dotenv import load_dotenv
from fasthtml.common import *
from monsterui.all import *


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / ".local" / "wiki_graph_app"
RAG_STORAGE_DIR = APP_ROOT / "rag_storage"
STATE_FILE = APP_ROOT / "state.json"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TubeMind/0.1 (https://github.com/mailf/TubeMind)"
INDEX_SEARCH_LIMIT = 5
DEFAULT_QUERY_MODE = "mix"
QUERY_MODES = ("mix", "hybrid", "local", "global", "naive")


def load_environment() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in .env")


@dataclass
class CorpusState:
    indexed: bool = False
    seed_query: str = ""
    indexed_page_ids: list[int] = field(default_factory=list)
    indexed_titles: list[str] = field(default_factory=list)
    page_urls: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "CorpusState":
        if not STATE_FILE.exists():
            return cls()
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return cls(
            indexed=bool(data.get("indexed", False)),
            seed_query=str(data.get("seed_query", "")),
            indexed_page_ids=[int(page_id) for page_id in data.get("indexed_page_ids", [])],
            indexed_titles=[str(title) for title in data.get("indexed_titles", [])],
            page_urls={str(k): str(v) for k, v in data.get("page_urls", {}).items()},
        )

    def save(self) -> None:
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "indexed": self.indexed,
            "seed_query": self.seed_query,
            "indexed_page_ids": self.indexed_page_ids,
            "indexed_titles": self.indexed_titles,
            "page_urls": self.page_urls,
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class WikiGraphApp:
    def __init__(self) -> None:
        load_environment()
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        self.state = CorpusState.load()
        self.lock = threading.RLock()
        self.rag = self._create_rag()

    def _create_rag(self):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        llm_model = partial(
            openai_complete_if_cache,
            "gpt-5.1",
            reasoning_effort="none",
        )

        return LightRAG(
            working_dir=str(RAG_STORAGE_DIR),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def startup(self) -> None:
        await self.rag.initialize_storages()

    async def shutdown(self) -> None:
        await self.rag.finalize_storages()

    def seed_corpus(self, topic: str) -> dict[str, Any]:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ValueError("Enter a topic to search Wikipedia.")

        with self.lock:
            if self.state.indexed:
                return {
                    "inserted_titles": [],
                    "skipped_titles": list(self.state.indexed_titles),
                    "message": "The corpus is already indexed. Use the follow-up query form below.",
                }

            search_results = wikipedia_search(normalized_topic, limit=INDEX_SEARCH_LIMIT)
            if not search_results:
                raise ValueError("Wikipedia returned no matching articles for that topic.")

            page_ids = [result["pageid"] for result in search_results]
            articles = wikipedia_fetch_articles(page_ids)
            if not articles:
                raise ValueError("Wikipedia search worked, but article content could not be fetched.")

            documents: list[str] = []
            ids: list[str] = []
            file_paths: list[str] = []
            inserted_titles: list[str] = []

            for article in articles:
                page_id = int(article["pageid"])
                title = article["title"]
                text = article["extract"].strip()
                canonical_url = article["fullurl"]
                if not text or page_id in self.state.indexed_page_ids:
                    continue

                documents.append(
                    "\n\n".join(
                        [
                            f"Title: {title}",
                            f"Source: {canonical_url}",
                            text,
                        ]
                    )
                )
                ids.append(f"wikipedia:{page_id}")
                file_paths.append(canonical_url)
                inserted_titles.append(title)
                self.state.indexed_page_ids.append(page_id)
                self.state.indexed_titles.append(title)
                self.state.page_urls[title] = canonical_url

            if not documents:
                raise ValueError("No new Wikipedia article text was available to index.")

            self.rag.insert(documents, ids=ids, file_paths=file_paths)
            self.state.indexed = True
            self.state.seed_query = normalized_topic
            self.state.save()

            return {
                "inserted_titles": inserted_titles,
                "skipped_titles": [],
                "message": f"Indexed {len(inserted_titles)} Wikipedia articles for '{normalized_topic}'.",
            }

    def query_corpus(self, query: str, mode: str = DEFAULT_QUERY_MODE) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Enter a question for the indexed corpus.")
        if not self.state.indexed:
            raise ValueError("Index the Wikipedia corpus first.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        with self.lock:
            from lightrag import QueryParam

            return str(
                self.rag.query(
                    normalized_query,
                    param=QueryParam(mode=mode, response_type="Multiple Paragraphs"),
                )
            )


def wikipedia_request(params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({**params, "format": "json", "formatversion": "2"})
    request = UrlRequest(
        f"{WIKIPEDIA_API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wikipedia_search(topic: str, limit: int = INDEX_SEARCH_LIMIT) -> list[dict[str, Any]]:
    data = wikipedia_request(
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": limit,
            "srprop": "",
        }
    )
    return data.get("query", {}).get("search", [])


def wikipedia_fetch_articles(page_ids: list[int]) -> list[dict[str, Any]]:
    data = wikipedia_request(
        {
            "action": "query",
            "prop": "extracts|info",
            "pageids": "|".join(str(page_id) for page_id in page_ids),
            "inprop": "url",
            "explaintext": 1,
            "redirects": 1,
        }
    )
    pages = data.get("query", {}).get("pages", [])
    return [
        page
        for page in pages
        if page.get("missing") is None and page.get("extract") and page.get("fullurl")
    ]


app_state = WikiGraphApp()

app, rt = fast_app(
    title="TubeMind Wikipedia GraphRAG",
    pico=False,
    hdrs=(
        *Theme.zinc.headers(
            mode="light",
            radii=ThemeRadii.lg,
            shadows=ThemeShadows.md,
            font=ThemeFont.default,
        ),
        Style(
            """
            :root {
                --page-glow: radial-gradient(circle at top left, hsla(var(--primary), 0.24), transparent 34%);
                --page-glow-secondary: radial-gradient(circle at top right, rgba(255, 255, 255, 0.82), transparent 38%);
                --page-base: linear-gradient(180deg, #fffdf8 0%, #f5f1e7 100%);
                --panel-border: rgba(73, 57, 31, 0.10);
                --panel-shadow: 0 22px 60px rgba(73, 57, 31, 0.10);
            }
            body {
                min-height: 100vh;
                background:
                    var(--page-glow),
                    var(--page-glow-secondary),
                    var(--page-base);
                color: #201811;
            }
            main.tubemind-shell {
                max-width: 980px;
                margin: 0 auto;
                padding: 2rem 1.25rem 4rem;
            }
            .tm-app-grid,
            .tm-panel {
                border: 1px solid var(--panel-border);
                border-radius: 1.5rem;
                background: rgba(255, 255, 255, 0.74);
                backdrop-filter: blur(14px);
                box-shadow: 0 14px 40px rgba(73, 57, 31, 0.08);
            }
            .tm-app-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1.1rem;
            }
            .tm-panel {
                padding: 1.4rem;
            }
            .tm-panel h2 {
                margin-bottom: 0.5rem;
                font-size: 1.45rem;
                line-height: 1.1;
            }
            .tm-panel p {
                color: rgba(32, 24, 17, 0.72);
            }
            .tm-panel form {
                display: grid;
                gap: 0.9rem;
                margin-top: 1rem;
            }
            .tm-panel textarea,
            .tm-panel select {
                border-radius: 1rem;
                border: 1px solid rgba(73, 57, 31, 0.14);
                background: rgba(255, 252, 247, 0.92);
            }
            .tm-panel textarea {
                min-height: 138px;
            }
            .tm-panel button {
                justify-content: center;
                min-height: 3rem;
                font-weight: 700;
            }
            #response-area pre {
                margin-top: 1rem;
                padding: 1.15rem;
                white-space: pre-wrap;
                border-radius: 1.2rem;
                background: rgba(27, 23, 18, 0.94);
                color: #fff7ea;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
            }
            .tm-response {
                margin-top: 1.5rem;
                padding: 1.5rem;
            }
            .tm-response details {
                margin-top: 1rem;
                overflow: hidden;
                border-radius: 1rem;
                background: rgba(255, 255, 255, 0.68);
            }
            .tm-response ul {
                margin: 0;
                padding: 0 1rem 1rem 2.2rem;
            }
            .htmx-indicator {
                display: none;
            }
            .htmx-request.htmx-indicator,
            .htmx-request .htmx-indicator {
                display: block;
            }
            @media (max-width: 920px) {
                .tm-app-grid {
                    grid-template-columns: 1fr;
                }
            }
            """
        ),
    ),
    on_startup=[app_state.startup],
    on_shutdown=[app_state.shutdown],
)


def page_main(*content: Any):
    """Render the full application shell.

    This function intentionally renders only the interactive application
    surface: the two workflow forms and the response area. HTMX swaps replace
    the full `main` element, so this wrapper keeps the form layout stable
    without reintroducing decorative hero content or nonessential controls.
    """
    state = app_state.state
    seed_button_label = "Index corpus and answer" if not state.indexed else "Corpus already indexed"
    query_button_cls = "uk-btn uk-btn-primary"
    seed_button_cls = "uk-btn uk-btn-primary"
    return Main(
        Div(
            Card(
                H2("Seed the knowledge base"),
                P(
                    "Search Wikipedia, pull the top articles into LightRAG, and generate the first answer from the newly built graph."
                ),
                Form(
                    Label("Wikipedia topic or search phrase", _for="seed_query"),
                    Textarea(
                        state.seed_query or "",
                        id="seed_query",
                        name="seed_query",
                        placeholder="Example: Ada Lovelace and the analytical engine",
                        disabled=state.indexed,
                    ),
                    Small("Indexing is intentionally one-time for this demo so follow-up questions stay fast and deterministic."),
                    Button(
                        seed_button_label,
                        type="submit",
                        disabled=state.indexed,
                        cls=seed_button_cls,
                        hx_post="/seed",
                        hx_target="main",
                        hx_swap="outerHTML",
                        hx_indicator="#seed-indicator",
                    ),
                ),
                Div(
                    P("Indexing in progress. The page will refresh with the answer when the corpus is ready.", aria_busy="true"),
                    id="seed-indicator",
                    cls="htmx-indicator mt-3 text-sm text-[#6a5744]",
                ),
                cls="tm-panel",
            ),
            Card(
                H2("Query the existing graph"),
                P("Use the saved corpus for repeated analysis without paying the indexing cost again."),
                Form(
                    Label("Question", _for="query"),
                    Textarea(
                        "",
                        id="query",
                        name="query",
                        placeholder="Ask about the indexed Wikipedia corpus",
                    ),
                    Label("Retrieval mode", _for="mode"),
                    Select(
                        *[
                            Option(mode.upper(), value=mode, selected=mode == DEFAULT_QUERY_MODE)
                            for mode in QUERY_MODES
                        ],
                        id="mode",
                        name="mode",
                        disabled=not state.indexed,
                    ),
                    Small("`mix` is the default because it balances graph context with direct chunk retrieval."),
                    Button(
                        "Query graph",
                        type="submit",
                        disabled=not state.indexed,
                        cls=query_button_cls,
                        hx_post="/query",
                        hx_target="main",
                        hx_swap="outerHTML",
                        hx_indicator="#query-indicator",
                    ),
                ),
                Div(
                    P("Query in progress. Results will replace the response panel below.", aria_busy="true"),
                    id="query-indicator",
                    cls="htmx-indicator mt-3 text-sm text-[#6a5744]",
                ),
                cls="tm-panel",
            ),
            cls="tm-app-grid",
        ),
        *content if content else (response_panel(),),
        cls="tubemind-shell",
    )


def layout(*content: Any):
    """Wrap the rendered page body with the document title.

    Keeping title generation separate from `page_main` lets both full-page and
    HTMX responses share the same content builder while only full navigations
    emit document-level metadata.
    """
    return Title("TubeMind Wikipedia GraphRAG"), page_main(*content)


def status_panel() -> Any:
    """Render the compact corpus status panel.

    This panel exists outside the forms so both workflows can reference the
    current application state at a glance. It highlights whether indexing has
    already happened, what seed query produced the corpus, and how many source
    articles are available for retrieval.
    """
    state = app_state.state
    if state.indexed:
        return Card(
            H2("Corpus status"),
            P("Indexing is complete. Every follow-up question now reuses the existing LightRAG corpus."),
            Div(
                Div(
                    Strong("State"),
                    Span("Ready"),
                    cls="tm-status-pill",
                ),
                Div(
                    Strong("Seed query"),
                    Span(state.seed_query),
                    cls="tm-status-pill",
                ),
                Div(
                    Strong("Stored articles"),
                    Span(str(len(state.indexed_titles))),
                    cls="tm-status-pill",
                ),
                cls="tm-status-row",
            ),
            cls="tm-panel mt-6",
        )
    return Card(
        H2("Corpus status"),
        P("No Wikipedia corpus has been indexed yet. Seed the graph once to unlock the follow-up query workflow."),
        Div(
            Div(
                Strong("State"),
                Span("Waiting"),
                cls="tm-status-pill",
            ),
            Div(
                Strong("Seed query"),
                Span("Not set"),
                cls="tm-status-pill",
            ),
            Div(
                Strong("Stored articles"),
                Span("0"),
                cls="tm-status-pill",
            ),
            cls="tm-status-row",
        ),
        cls="tm-panel mt-6",
    )


def details_block(titles: list[str]) -> Any:
    """Render the source article list for the current response.

    The response section can show both newly inserted titles and the full saved
    corpus. Keeping the source list in a collapsible block preserves a clean
    primary reading experience while still exposing traceability when the user
    wants to inspect the indexed material.
    """
    return Details(
        Summary("Indexed articles"),
        Ul(
            *[
                Li(A(title, href=app_state.state.page_urls.get(title, "#"), target="_blank"))
                for title in titles
            ]
        ),
        open=True,
    )


def response_panel(
    heading: str = "Awaiting Input",
    message: str = "Seed the corpus to begin. Once indexing is complete, this page will clearly show that the index is ready.",
    answer: str = "",
    titles: list[str] | None = None,
) -> Any:
    """Render the bottom response surface for success, idle, and error states.

    This single component is used for the initial empty state, seed results,
    follow-up answers, and exception messages. Consolidating those states keeps
    HTMX updates simple and ensures all outcomes share the same visual treatment
    and source-link affordances.
    """
    return Card(
        H2(heading),
        P(message),
        Pre(answer) if answer else "",
        details_block(titles or []) if titles else "",
        id="response-area",
        cls="tm-panel tm-response",
    )


@rt("/")
def get(request: Request):
    if request.headers.get("HX-Request") == "true":
        return page_main()
    return layout()


@rt("/seed", methods=["POST"])
def post(request: Request, seed_query: str = ""):
    try:
        seed_result = app_state.seed_corpus(seed_query)
        answer = app_state.query_corpus(seed_query, mode=DEFAULT_QUERY_MODE)
        content = response_panel(
            "Index Complete",
            seed_result["message"] + " The corpus is now ready for follow-up questions without reindexing.",
            answer=answer,
            titles=seed_result["inserted_titles"] or app_state.state.indexed_titles,
        )
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)
    except Exception as exc:
        content = response_panel("Seed Corpus", str(exc))
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)


@rt("/query", methods=["POST"])
def post_query(request: Request, query: str = "", mode: str = DEFAULT_QUERY_MODE):
    try:
        answer = app_state.query_corpus(query, mode=mode)
        content = response_panel(
            "Follow-up Answer",
            f"Answered from the existing corpus with mode '{mode}'. No reindex was performed.",
            answer=answer,
            titles=app_state.state.indexed_titles,
        )
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)
    except Exception as exc:
        content = response_panel("Query Existing Corpus", str(exc))
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)


if __name__ == "__main__":
    serve()
