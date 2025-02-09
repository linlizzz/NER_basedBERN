from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def visualize_subgraph(self, nodes):
        with self.driver.session() as session:
            # Create Cypher query to fetch subgraph
            node_list = "', '".join(nodes)
            query = f"""
                MATCH (n1)
                WHERE n1.id IN ['{node_list}']
                MATCH (n1)-[r]-(n2)
                WHERE n2.id IN ['{node_list}']
                RETURN n1, r, n2
            """
            result = session.run(query)
            return result.data()

def visualization_neo4j(nodes_list, neo4j_url, username, password):
    try:
        # Initialize Neo4j connection
        neo4j_conn = Neo4jConnection(neo4j_url, username, password)
        
        # Get subgraph visualization
        vis_result = neo4j_conn.visualize_subgraph(nodes_list)
        
        # Close connection
        neo4j_conn.close()
        
        
        return vis_result
    except Exception as e:
        return f"Error visualizing in Neo4j: {str(e)}"
'''
def agent(kg_nodes_embedding, user_input, option = "combined"):
    
    vis_res = visualization_neo4j(nodes_list_answer)
    print("Neo4j Visualization Results:")
    print(vis_res)
    return vis_res  # Return visualization results
'''
