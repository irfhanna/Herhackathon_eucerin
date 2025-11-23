import json
import logging
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
import ollama


# -----------------------------
# 1. TRIPLE EXTRACTION (OLLAMA)
# -----------------------------
def extract_triples(text_block):
    """
    Extract knowledge graph triples using a local Ollama model.
    Returns: list of dicts: [{"subject": "", "relation": "", "object": ""}]
    """
    prompt = f"""
    Extract knowledge graph triples from the text.
    Return ONLY valid JSON in this structure:

    [
      {{"subject": "...", "relation": "...", "object": "..."}}
    ]

    Text:
    {text_block}
    """

    try:
        response = ollama.chat(
            model="llama2:7b",    
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        content = response["message"]["content"]
        triples = json.loads(content)

    except Exception as e:
        print("⚠️ Could not parse triples:", e)
        triples = []

    return triples


# --------------------------------------
# 2. CONVERT TRIPLES → NETWORKX GRAPH
# --------------------------------------
def triples_to_graph(triples, graph=None, source="csv"):
    if graph is None:
        graph = nx.DiGraph()

    for t in triples:
        subj = t.get("subject", "").strip()
        rel = t.get("relation", "").strip()
        obj = t.get("object", "").strip()

        if not subj or not obj or not rel:
            continue

        graph.add_node(subj, label="Entity", source=source)
        graph.add_node(obj, label="Entity", source=source)
        graph.add_edge(subj, obj, relation=rel, source=source)

    return graph


# ----------------------------------------
# 3. BUILD GRAPH BY PROCESSING CSV COLUMN
# ----------------------------------------
def build_graph_from_csv(csv_path, text_column="clean_text"):
    df = pd.read_csv(csv_path)

    graph = nx.DiGraph()

    for idx, row in df.iterrows():
        text = row.get(text_column, "")

        if not isinstance(text, str) or len(text.strip()) < 5:
            continue

        # chunk text in 400-character segments
        for i in range(0, len(text), 400):
            chunk = text[i:i+400]
            triples = extract_triples(chunk)
            graph = triples_to_graph(triples, graph, source="csv")

    return graph


# -----------------------------
# 4. EXPORT GRAPH TO NEO4J
# -----------------------------
def export_graph_to_neo4j(graph, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        print("Clearing Neo4j database...")
        session.run("MATCH (n) DETACH DELETE n")

        print("Uploading nodes...")
        for node, data in graph.nodes(data=True):
            session.run("""
                MERGE (n:Entity {name: $name})
                SET n.label = $label
            """, name=node, label=data.get("label", "Entity"))

        print("Uploading edges...")
        for u, v, data in graph.edges(data=True):
            session.run("""
                MATCH (a:Entity {name: $start})
                MATCH (b:Entity {name: $end})
                MERGE (a)-[r:RELATION {type: $type}]->(b)
            """, start=u, end=v, type=data.get("relation", "related_to"))

    print("Graph export complete!")


# -----------------------------
# 5. MAIN SCRIPT
# -----------------------------
def main():
    logging.basicConfig(level=logging.INFO)

    CSV_PATH = "./skin_social_media_data.csv"   # ← your CSV file
    TEXT_COLUMN = "clean_text"


    print("Building knowledge graph from CSV...")
    graph = build_graph_from_csv(CSV_PATH, TEXT_COLUMN)

    print("Graph stats:")
    print("  Nodes:", graph.number_of_nodes())
    print("  Edges:", graph.number_of_edges())

    export_graph_to_neo4j(graph, NEO4J_URI, NEO4J_USER, NEO4J_PASS)


if __name__ == "__main__":
    main()
