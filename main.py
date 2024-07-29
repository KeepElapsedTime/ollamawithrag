import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Demo knowledgebase
knowledge_base = [
    "Ollama是一個強大的本地語言模型工具。",
    "RAG代表檢索增強生成。",
    "Python是一種流行的編程語言。",
    "向量數據庫用於高效地存儲和檢索向量。"
]

model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base_embeddings = model.encode(knowledge_base)

def query_ollama(prompt, api_url="YOUR_API_URL", model="YOUR_MODEL_NAME"):
    """Search Ollama API"""
    try:
        response = requests.post(f"{api_url}/api/generate", json={"model":model,"prompt": prompt,"stream":False})
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response Content: {response.content}")
        
        response.raise_for_status()
        
        try:
            return response.json()['response']
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Response Text: {response.text}")
            return f"Error decoding JSON: {e}"
    except requests.RequestException as e:
        print(f"Request Error: {e}")
        return f"Error making request: {e}"

def retrieve_relevant_context(query, top_k=2):
    """檢索相關上下文"""
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], knowledge_base_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    return [knowledge_base[i] for i in top_indices]

def rag_with_ollama(query):
    """使用RAG和Ollama生成回答"""
    context = retrieve_relevant_context(query)
    prompt = f"基於以下信息：\n{' '.join(context)}\n\n回答問題：{query}"
    return query_ollama(prompt)

# 測試
def main():
    user_query = "什麼是Ollama和RAG？"
    print(f"問題: {user_query}")
    answer = rag_with_ollama(user_query)
    print(f"回答: {answer}")

if __name__ == "__main__":
    main()

