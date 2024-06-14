from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase

app = FastAPI()

# Neo4j driver initialization
uri = "bolt://localhost:7687"  # Change to your Neo4j URI
user = "neo4j"  # Change to your Neo4j username
password = "12345678"  # Change to your Neo4j password
driver = GraphDatabase.driver(uri, auth=(user, password))

class Node(BaseModel):
    label: str
    properties: dict

def create_node_in_neo4j(label: str, properties: dict):
    query = f"CREATE (n:{label}) set n= $properties RETURN n"
    with driver.session() as session:
        x={"properties":properties}
        result = session.run(query, x)
        return result.single()

@app.post("/nodes/")
async def create_node(node: Node):
    try:
        result = create_node_in_neo4j(node.label, node.properties)
        if result:
            return {"message": "Node created successfully", "node": result["n"]}
        else:
            raise HTTPException(status_code=400, detail="Node creation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
