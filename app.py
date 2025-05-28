"""
Application streamlit pour rechercher dans la base de données vectorielle.
"""
from inference import query, llm_summary

def main():
    while True:
        inp = input("Requête (q pour quitter) : ")
        if inp.lower() == "q":
            break
        
        docs = query(inp, 3)
        print(docs)
        
        for tok in llm_summary("Infos linguistique", docs):
            print(tok, end="", flush=True)

if __name__ == "__main__":
    main()
