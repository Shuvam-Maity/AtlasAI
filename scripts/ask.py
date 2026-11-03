from src.qa import answer_question
from src.graph_db import close


def main():
    print("AtlasAI - ask about AI topics. Type 'exit' to quit.")
    print("Add '--facts' to a question to also see the graph facts used.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break

        show_facts = "--facts" in q
        q = q.replace("--facts", "").strip()

        out = answer_question(q)
        print(f"\nAtlasAI: {out['answer']}\n")

        if out["sources"]:
            print("Sources:")
            for s in out["sources"]:
                print(f"  [{s['n']}] {s['title']} - {s['url']}")
            print()

        if show_facts and out["facts"]:
            print("Graph facts used:")
            for s, r, t in out["facts"][:15]:
                print(f"  {s} -{r}-> {t}")
            print()

    close()


if __name__ == "__main__":
    main()