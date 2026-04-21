"""
LLM/chatbot_page.py
------------------------
Streamlit chat page — plugs into the DashboardBI dashboard as an extra page.

Add this to DashboardBI/app.py navigation:
    from LLM.chatbot_page import render_chatbot
    elif page == "Assistant BI":
        render_chatbot()
"""

import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from LLM.mcp.mcp_client import get_mcp_client

EXAMPLE_QUESTIONS = [
    "Quels sont les 5 meilleurs produits cette semaine ?",
    "Génère un rapport de tendances du marché.",
    "Crée un profil client basé sur les Top-K produits.",
    "Quelle stratégie marketing recommandes-tu ?",
    "Quelles sont les règles d'association avec le lift le plus élevé ?",
    "Donne-moi les produits Electronics les mieux notés.",
]


def _route_question(question: str, client) -> str:
    """Route a user question to the right MCP tool."""
    q = question.lower()
    resp: dict = {}

    if any(w in q for w in ["tendance","rapport","trend","semaine","marché"]):
        resp = client.call("generate_trend_report", {"top_k": 10})
    elif any(w in q for w in ["profil","client","persona","cible"]):
        resp = client.call("build_client_profile", {"top_k": 20})
    elif any(w in q for w in ["stratégie","marketing","recommand","action"]):
        resp = client.call("recommend_strategy", {})
    elif any(w in q for w in ["règle","association","lift","panier"]):
        resp = client.call("get_association_rules", {"min_lift": 1.5, "top_n": 5})
        if "result" in resp:
            rules = resp["result"]
            if rules:
                lines = [f"  {r['antecedents']} → {r['consequents']} (lift={r['lift']:.2f})"
                         for r in rules]
                return "Règles d'association (top lift) :\n" + "\n".join(lines)
    elif any(w in q for w in ["top","meilleur","classement","produit"]):
        resp = client.call("get_top_products", {"k": 5})
        if "result" in resp:
            prods = resp["result"]
            lines = [f"  {i+1}. {p.get('title','?')} — score={p.get('score',0):.3f}, "
                     f"prix={p.get('price',0):.2f}$, ★{p.get('rating','?')}"
                     for i, p in enumerate(prods)]
            return "Top produits :\n" + "\n".join(lines)
    else:
        resp = client.call("answer_question", {"question": question})

    if "error" in resp:
        return f"Erreur : {resp['error']}"
    return resp.get("result", "Pas de réponse disponible.")


def render_chatbot():
    """Render the chatbot page inside the Streamlit app."""
    st.title("Assistant BI")
    st.caption("Posez vos questions sur les données produits — propulsé par LLM via MCP")

    client = get_mcp_client()

    # Quick-access example buttons
    st.markdown("**Questions rapides :**")
    cols = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.setdefault("messages", [])
            st.session_state["messages"].append({"role": "user", "content": q})
            with st.spinner("Analyse en cours..."):
                answer = _route_question(q, client)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Posez votre question..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                answer = _route_question(user_input, client)
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Audit log sidebar
    with st.expander("Journal d'audit MCP"):
        logs = client.audit.get_recent_logs(n=10)
        if logs:
            import pandas as pd
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs[["ts","server","tool","allowed","latency_ms"]],
                         use_container_width=True)
        else:
            st.info("Aucun appel enregistré encore.")