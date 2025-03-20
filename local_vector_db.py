#!/usr/bin/env python
# coding: utf-8

from chromadb.config import Settings
from chromadb import Client
import os
from glob import glob
import chromadb
from flask import Flask, request, jsonify
from tqdm import tqdm
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores.base import VectorStore
from typing import Dict, List
import atexit
import gc

def increase_file_limit(target_limit=65535):
    """Set the file descriptor limit to the target_limit if possible."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard < target_limit:
        print(f"Hard limit ({hard}) is less than the target ({target_limit}). Adjust your system configuration.")
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard))
        print(f"Successfully increased file descriptor limit to {target_limit}.")
    except ValueError as e:
        print(f"Failed to set file descriptor limit to {target_limit}: {e}")
    except Exception as ex:
        print(f"Unexpected error when setting file descriptor limit: {ex}")

class ChromaDBManager:
    """Manages ChromaDB connections with proper cleanup."""
    
    def __init__(self):
        self.dbs = {}
        self.embeddings = None
        atexit.register(self.cleanup_all)

    def initialize(self, db_paths: List[str], embedding_model: str, collection_name: str):
        """Initialize ChromaDB instances."""
        print("Initializing ChromaDB embeddings...")
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        for idx, db_path in enumerate(tqdm(db_paths, desc="Processing ChromaDBs", unit="db")):
            try:
                db_name = f"chroma_db{idx + 1}"
                db, stats = self._initialize_single_db(
                    db_path,
                    collection_name
                )
                self.dbs[db_name] = db
                print(f"Initialized {db_name} with {stats['document_count']} documents")
            except Exception as e:
                print(f"Error processing DB at {db_path}: {e}")
                continue

    def _initialize_single_db(self, persist_directory: str, collection_name: str) -> tuple[Chroma, dict]:
        """Initialize a single ChromaDB instance."""
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        try:
            collection = db._collection
            count = collection.count() if collection else 0
            print(f"Collection found with {count} documents")
        except Exception as e:
            print(f"Warning: Error accessing collection: {e}")
            count = 0

        return db, {"document_count": count}

    def cleanup_db(self, db_name: str):
        """Cleanup specific database connection."""
        if db_name in self.dbs:
            try:
                # Simply remove the reference to the database object
                # Let Python's garbage collector handle the cleanup
                print(f"Removing {db_name} from active databases")
                del self.dbs[db_name]
                print(f"Successfully cleaned up {db_name}")
            except Exception as e:
                print(f"Error cleaning up DB {db_name}: {e}")

    def cleanup_all(self):
        """Cleanup all database connections."""
        print("\nCleaning up all ChromaDB connections...")
        db_names = list(self.dbs.keys())
        for db_name in db_names:
            self.cleanup_db(db_name)
        self.dbs.clear()
        if self.embeddings:
            self.embeddings = None
        gc.collect()
        print("All ChromaDB connections cleaned up")

# Initialize Flask app
app = Flask(__name__)

# Configure OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-tQi6bjiemwTkb4HL9aBVT3BlbkFJ8CXmkRGGXYI60a1VF8An"

# Initialize ChromaDB manager
db_manager = ChromaDBManager()

# Global configuration
increase_file_limit()
db_paths = glob(f"..{os.sep}knowledge*{os.sep}knowledge_base*")
embedding_model = 'text-embedding-ada-002'
collection_name = 'publications'

print(f"Found {len(db_paths)} DB paths")
print(f"DB Paths: {db_paths}")
print(f"Embedding Model: {embedding_model}")
print(f"Collection Name: {collection_name}")

# Initialize ChromaDB instances
db_manager.initialize(db_paths, embedding_model, collection_name)

@app.route('/query', methods=['POST'])
def query_documents():
    """Handle POST requests to query the vector databases."""
    try:
        payload = request.json
        query = payload.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        top_k = payload.get('top_k', 5)
        similarity_threshold = payload.get('similarity_threshold', 0.75)
        
        normalized_query = query.lower()
        all_results = []
        
        # Query each database
        for db_name, db in db_manager.dbs.items():
            try:
                db_results = db.similarity_search_with_relevance_scores(
                    normalized_query,
                    k=top_k
                )
                print(f"Found {len(db_results)} results in {db_name}")
                for doc, score in db_results:
                    all_results.append((doc, score, db_name))
            except Exception as e:
                print(f"Error querying {db_name}: {str(e)}")
                continue
        
        filtered_results = [
            res for res in all_results if res[1] >= similarity_threshold
        ]
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        if not sorted_results:
            return jsonify({
                "message": "No relevant context information found.",
                "results": [],
                "context": ""
            }), 200
        
        results_with_similarity = [
            {
                "db_name": db_name,
                "source": doc.metadata.get("source", "N/A"),
                "start_index": doc.metadata.get("start_index", "N/A"),
                "content": doc.page_content,
                "relevance_score": float(score)
            }
            for doc, score, db_name in sorted_results
        ]
        
        context_text = "\n\n---\n\n".join([
            f"Database: {db_name}\n"
            f"Source: {doc.metadata.get('source', 'N/A')}\n"
            f"Content: {doc.page_content}\n"
            f"Relevance Score: {score:.2f}" 
            for doc, score, db_name in sorted_results
        ])
        
        return jsonify({
            "message": f"Found {len(sorted_results)} relevant context entries",
            "results": results_with_similarity,
            "context": context_text
        }), 200
        
    except Exception as e:
        print(f"Error in query_documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "db_count": len(db_manager.dbs),
        "db_names": list(db_manager.dbs.keys())
    }), 200

def main():
    """Main entry point for the application."""
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        return 1
    finally:
        # Ensure cleanup on exit
        db_manager.cleanup_all()
    return 0

if __name__ == '__main__':
    exit(main())