import re
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src.qa import answer_question

st.set_page_config(page_title="AtlasAI", page_icon="🧭", layout="wide")
st.title("AtlasAI")
st.caption("Ask about AI topics. Answers come from a Neo4j knowledge graph built from web sources.")

with st.sidebar:
    st.header("About")
    st.write(
        "GraphRAG over an AI knowledge graph: vector search finds relevant text, "
        "then graph traversal adds connected facts. Only AI-related questions are answered."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def graph_html(facts):
    net = Network(height="420px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222222", cdn_resources="remote")
    for s, r, t in facts:
        net.add_node(s, label=s)
        net.add_node(t, label=t)
        net.add_edge(s, t, label=r, title=r)
    return net.generate_html()


def to_streamlit_math(text):
    """Convert \\( ... \\) to $...$ and \\[ ... \\] to $$...$$ so Streamlit renders the math."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "$$" + m.group(1).strip() + "$$", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    return text


def render_assistant(msg):
    st.markdown(to_streamlit_math(msg["answer"]))
    if msg["sources"]:
        st.markdown("**Sources**")
        for s in msg["sources"]:
            if s["url"].startswith("local/"):
                st.markdown(f"[{s['n']}] 📄 {s['title']} (paper)")
            else:
                st.markdown(f"[{s['n']}] [{s['title']}]({s['url']})")
    if msg["facts"]:
        with st.expander(f"Graph facts used ({len(msg['facts'])})"):
            components.html(graph_html(msg["facts"]), height=440)
            for s, r, t in msg["facts"]:
                st.write(f"{s} → {r} → {t}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_assistant(m)

if question := st.chat_input("Ask an AI question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the graph..."):
            out = answer_question(question)
        msg = {
            "role": "assistant",
            "answer": out["answer"],
            "sources": out["sources"],
            "facts": out["facts"],
        }
        render_assistant(msg)
    st.session_state.messages.append(msg)