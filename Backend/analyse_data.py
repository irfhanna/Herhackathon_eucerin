import ollama
from neo4j import GraphDatabase


# --- Neo4j Driver ---
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def retrieve_graph_info(concern, limit=15):
    """
    Retrieve entity relationships from Neo4j matching the skincare concern.
    Returns list of strings: 'A -[REL]-> B'
    """
    query = """
    MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
    WHERE toLower(e1.name) CONTAINS toLower($kw)
       OR toLower(e2.name) CONTAINS toLower($kw)
    RETURN e1.name AS source, type(r) AS relation, e2.name AS target
    LIMIT $limit
    """

    with driver.session() as session:
        results = session.run(query, kw=concern, limit=limit)
        return [
            f"{record['source']} -[{record['relation']}]-> {record['target']}"
            for record in results
        ]


def call_ollama(prompt, model="llama2:7b"):
    """
    Simple wrapper for Ollama text generation.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def analyse_skin_concern(concern):
    """
    Input: skin concern string
    Output:      
      - Summary
      - Most asked question
      - Skincare product question
    """

    graph_data = retrieve_graph_info(concern)

    graph_text = "\n".join(graph_data) if graph_data else " "

    prompt = f"""
You are an expert skincare analyst.

You will receive:
- A skincare concern
- Knowledge graph relationships extracted from user discussions

Task:
1. Summarise the main insights about the concern.
2. Identify the *most asked question* based on the relationships.
3. Create *one skincare-product-related question* commonly asked for this concern.

Skincare Concern: {concern}

Graph Data:
{graph_text}

Output format:

Summary:
<summary>

Most Asked Question:
<question>

Product-Related Question:
<question>
"""

    result = call_ollama(prompt)

    return {
        "concern": concern,
        "graph_data": graph_data,
        "analysis": result
    }


# --- Example Usage ---
if __name__ == "__main__":
    concern = "acne"
    result = analyse_skin_concern(concern)

   

    driver.close()
