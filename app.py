"""Streamlit chat UI for ResearchCrew.

- API keys are read from .env only (never shown in the UI).
- Multiple chats; each chat has its OWN history and its OWN PDF knowledge base.
  A new chat starts with no PDFs; PDFs uploaded in one chat are invisible to every other chat.
"""
from __future__ import annotations

import json
from dataclasses import replace

import streamlit as st

from researchcrew.agents import AgentError, CrewAgentRunner
from researchcrew.chat import answer_followup, answer_from_pdf, route
from researchcrew.chats import ChatStore
from researchcrew.config import ConfigError, Settings
from researchcrew.pipeline import ResearchPipeline
from researchcrew.retrieval import (BGEEmbedder, Retriever, SerperClient, VectorStore,
                                    chunk_pages, load_pdf)

st.set_page_config(page_title="ResearchCrew", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%); color: white;}
.hero h1 {margin: 0; font-size: 1.9rem;} .hero p {margin: .3rem 0 0; opacity: .92;}
</style>
<div class="hero"><h1>🔬 ResearchCrew</h1>
<p>Plans · retrieves (web + PDF RAG) · critiques its own evidence · re-researches gaps · writes a cited report</p></div>
""", unsafe_allow_html=True)

EXAMPLES = [
    "What are the trade-offs between RAG and fine-tuning for domain-specific question answering?",
    "How do multi-agent LLM frameworks like CrewAI, LangGraph and AutoGen differ in design?",
    "What are current best practices for evaluating retrieval-augmented generation systems?",
]
ICON = {"plan": "🧭", "retrieve": "🔎", "research": "🧪", "critique": "🧐", "write": "✍️", "done": "✅"}

base = Settings.from_env()
chats = ChatStore(base.chats_dir)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder(model: str) -> BGEEmbedder:
    return BGEEmbedder(model)


@st.cache_resource(show_spinner=False)
def get_store(model: str, chroma_dir: str, chat_id: str) -> VectorStore:
    """One private ChromaDB collection per chat => PDFs never leak between chats."""
    return VectorStore(get_embedder(model), chroma_dir, collection=f"chat_{chat_id}")


# ------------------------------------------------------------------ current chat
if "chat_id" not in st.session_state:
    existing = chats.list()
    st.session_state.chat_id = existing[0]["id"] if existing else chats.create()["id"]
chat = chats.load(st.session_state.chat_id) or chats.create()
st.session_state.chat_id = chat["id"]
messages: list[dict] = chat["messages"]
store = get_store(base.embedding_model, base.chroma_dir, chat["id"])
if "pending" not in st.session_state:
    st.session_state.pending = None
if "upload_n" not in st.session_state:
    st.session_state.upload_n = 0


def switch_to(chat_id: str) -> None:
    st.session_state.chat_id = chat_id
    st.session_state.upload_n += 1       # reset the uploader widget
    st.rerun()


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        if messages or store.count():
            switch_to(chats.create()["id"])

    st.subheader("💬 Chats")
    for c in chats.list()[:15]:
        if c["n"] == 0 and c["id"] != chat["id"]:
            continue
        current = c["id"] == chat["id"]
        if st.button(("▶ " if current else "") + c["title"][:34], key=f"chat_{c['id']}",
                     use_container_width=True) and not current:
            switch_to(c["id"])

    st.divider()
    st.subheader("📄 PDFs in this chat")
    st.caption("Uploaded PDFs belong to this chat only. A new chat starts with none.")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                             key=f"up_{chat['id']}_{st.session_state.upload_n}")
    if files and st.button("Index uploaded PDFs", use_container_width=True):
        with st.spinner("Parsing, chunking and embedding… (first run downloads the BGE model)"):
            for f in files:
                doc_id, name, pages = load_pdf(f.getvalue(), f.name)
                n = store.add_chunks(chunk_pages(doc_id, name, pages, base.chunk_size, base.chunk_overlap))
                messages.append({"role": "assistant", "type": "notice",
                                 "content": f"📎 Indexed **{name}** — {len(pages)} pages, {n} chunks. "
                                            f"You can now ask questions about it."})
        chats.save(chat)
        st.session_state.upload_n += 1
        st.rerun()
    sources = store.sources()
    if sources:
        for name, n in sources.items():
            st.caption(f"• {name} — {n} chunks")
        if st.button("Remove PDFs from this chat", use_container_width=True):
            store.clear()
            messages.append({"role": "assistant", "type": "notice", "content": "🗑️ PDFs removed from this chat."})
            chats.save(chat)
            st.rerun()
    else:
        st.caption("No PDFs indexed in this chat.")

    st.divider()
    st.subheader("⚙️ Settings")
    st.markdown(("✅" if base.groq_api_key else "❌") + " Groq key  ·  "
                + ("✅" if base.serper_api_key else "⚪") + " Serper key")
    st.caption(f"Models: `{base.model_smart.split('/', 1)[-1]}` · `{base.model_fast.split('/', 1)[-1]}`")
    max_iters = st.slider("Max critic re-research rounds", 0, 3, base.max_iterations)
    k_web = st.slider("Web results per query", 1, 8, base.k_web)
    k_pdf = st.slider("PDF chunks per query", 1, 8, base.k_pdf)
    use_web = st.toggle("Use live web search", value=bool(base.serper_api_key),
                        disabled=not base.serper_api_key)
    settings = replace(base, max_iterations=max_iters, k_web=k_web, k_pdf=k_pdf)

    st.divider()
    if st.button("🗑️ Delete this chat", use_container_width=True):
        store.drop()
        chats.delete(chat["id"])
        rest = chats.list()
        st.session_state.chat_id = rest[0]["id"] if rest else chats.create()["id"]
        st.session_state.upload_n += 1
        st.rerun()


# ------------------------------------------------------------------- result view
def render_result(trace: dict, idx: int, warnings: list[str] | None = None) -> None:
    for w in warnings or []:
        st.warning(w)
    s = trace["stats"]
    m = st.columns(6)
    m[0].metric("Sub-questions", s["sub_questions"])
    m[1].metric("Critic rounds", s["iterations"])
    m[2].metric("Evidence", s["evidence_retrieved"])
    m[3].metric("Sources cited", s["unique_sources"])
    m[4].metric("Citations", s["citations"])
    m[5].metric("Critic score", f"{s['final_score']:g}/10")

    t_report, t_plan, t_loop, t_ev = st.tabs(["📄 Report", "🧭 Findings", "🔁 Critic loop", "🗂️ Evidence"])
    with t_report:
        st.markdown(trace["report"])
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Markdown", trace["report"], "research_report.md", "text/markdown",
                           key=f"md_{chat['id']}_{idx}")
        c2.download_button("⬇️ Full trace (JSON)", json.dumps(trace, indent=2), "research_trace.json",
                           "application/json", key=f"js_{chat['id']}_{idx}")
    with t_plan:
        for f in trace["findings"]:
            with st.expander(f"Q{f['sub_id']}: {f['question']}"):
                st.markdown(f["answer"])
                st.caption(f"Confidence: **{f['confidence']}** · revisions: {f['revision']} · "
                           f"sources: {', '.join(f['evidence_ids']) or 'none'}")
                if f["unresolved"] and f["unresolved"].lower() != "none":
                    st.info(f"Unresolved: {f['unresolved']}")
    with t_loop:
        st.caption("Each round the critic scores the evidence; gaps trigger targeted follow-up searches.")
        for r in trace["reviews"]:
            st.markdown(f"**Review #{r['iteration']}** — score `{r['score']:g}/10` · verdict `{r['verdict']}`")
            if r["summary"]:
                st.write(r["summary"])
            for g in r["gaps"]:
                tag = "🤖 guardrail" if g["auto"] else "🧐 critic"
                st.markdown(f"- {tag} · Q{g['sub_id']}: {g['issue']}  \n  follow-up: `{g['follow_up_queries']}`")
            st.divider()
    with t_ev:
        for e in trace["evidence"]:
            icon = "🌐" if e["source_type"] == "web" else "📄"
            with st.expander(f"{icon} [{e['id']}] {e['title'][:90]} (score {e['score']})"):
                st.caption(e["location"])
                st.write(e["text"])


def render_passages(passages: list[dict]) -> None:
    with st.expander(f"📄 Passages used ({len(passages)})"):
        for p in passages:
            st.caption(f"{p['location']}")
            st.write(p["text"])


# --------------------------------------------------------------------- chat history
if not messages:
    st.caption("Ask a research question, upload a PDF in the sidebar and ask about it, or try one of these:")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state.pending = ex
            st.rerun()

for i, m in enumerate(messages):
    kind = m.get("type")
    with st.chat_message(m["role"], avatar="📎" if kind == "notice" else None):
        if kind == "research":
            if m.get("interpreted"):
                st.caption(f"Interpreted as: *{m['interpreted']}*")
            render_result(m["trace"], i, m.get("warnings"))
        elif kind == "notice":
            st.markdown(m["content"])
        else:
            st.markdown(m["content"])
            if kind == "pdf" and m.get("passages"):
                render_passages(m["passages"])

# ------------------------------------------------------------------------ new turn
prompt = st.chat_input("Ask a research question, or ask about your PDF…")
if not prompt and st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        web = SerperClient(base.serper_api_key, settings.fetch_pages) if (use_web and base.serper_api_key) else None
        retriever = Retriever(store, web, k_web, k_pdf, settings.pdf_min_score)
        try:
            runner = CrewAgentRunner(settings)
            pdf_names = list(store.sources().keys())
            pdf_score = None
            if pdf_names:
                top = store.search(prompt, k=1)
                pdf_score = top[0].score if top else 0.0
            with st.spinner("Understanding your message…"):
                mode, standalone = route(runner, prompt, messages, pdf_names, pdf_score)

            if mode == "pdf":
                with st.spinner("Reading your PDF…"):
                    answer, passages = answer_from_pdf(runner, store, prompt, standalone)
                st.markdown(answer)
                render_passages(passages)
                new = [{"role": "user", "content": prompt},
                       {"role": "assistant", "type": "pdf", "content": answer, "passages": passages}]
            elif mode == "followup":
                with st.spinner("Answering from the conversation…"):
                    answer = answer_followup(runner, prompt, messages)
                st.markdown(answer)
                new = [{"role": "user", "content": prompt},
                       {"role": "assistant", "type": "chat", "content": answer}]
            else:
                if not retriever.has_sources:
                    st.error("No sources available. Add SERPER_API_KEY to .env and/or upload a PDF in this chat.")
                    st.stop()
                interpreted = standalone if standalone.strip() != prompt.strip() else ""
                if interpreted:
                    st.caption(f"Interpreted as: *{interpreted}*")
                with st.status("Running the crew…", expanded=True) as status:
                    pipe = ResearchPipeline(
                        runner, retriever, settings,
                        on_event=lambda stg, msg: status.write(f"{ICON.get(stg, '•')} **{stg}** — {msg}"))
                    result = pipe.run(standalone)
                    status.update(label=f"Done in {result.elapsed_s}s", state="complete", expanded=False)
                trace = result.to_dict()
                render_result(trace, len(messages) + 1, retriever.warnings)
                new = [{"role": "user", "content": prompt},
                       {"role": "assistant", "type": "research", "content": result.report,
                        "trace": trace, "warnings": retriever.warnings, "interpreted": interpreted}]
            messages.extend(new)
            chats.save(chat)
        except ConfigError as exc:
            st.error(str(exc))
        except AgentError as exc:
            st.error(f"{exc}\n\nIf this is a rate limit, wait a minute and send the message again.")