"""
Application streamlit pour rechercher dans la base de données vectorielle.
"""

import rag_core as core
import inference
from langchain_core.messages import HumanMessage


def main():
    while True:
        inp = input("Requête (q pour quitter) : ")
        if inp.lower() == "q":
            break

        docs = core.query(inp, 3)
        print(docs, "\n\n")

        for chunk in inference.chat(
            [HumanMessage(content="Je cherche des infos sur la linguistique")]
        ):
            if isinstance(chunk, str):
                print(
                    f'\nAfficher un spinner, recherche avec la query "{chunk}"',
                    end="\n",
                    flush=True,
                )
            else:
                print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    main()
