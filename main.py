import requests
import json
import numpy as np

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print(f"Available models: {[model['name'] for model in models['models']]}")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Make sure it's running on port 11434")
        return False

def generate_embedding(text, model="mxbai-embed-large"):
    """Generate embedding for given text using the specified model"""
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["embedding"]
            print(f"✅ Successfully generated embedding for: '{text}'")
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
            return embedding
        else:
            print(f"❌ Error generating embedding: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def test_embedding_similarity(text1, text2, model="mxbai-embed-large"):
    """Test similarity between two texts using embeddings"""
    print(f"\n🔍 Testing similarity between:")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    
    embedding1 = generate_embedding(text1, model)
    embedding2 = generate_embedding(text2, model)
    
    if embedding1 and embedding2:
        # Calculate cosine similarity
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"📊 Cosine similarity: {cosine_sim:.4f}")
        
        return cosine_sim
    return None

if __name__ == "__main__":
    print("🚀 Testing Ollama and mxbai-embed-large model\n")
    
    # Test 1: Check Ollama connection
    if not test_ollama_connection():
        exit(1)
    
    print("\n" + "="*50)
    
    # Test 2: Generate a simple embedding
    test_text = "The quick brown fox jumps over the lazy dog"
    embedding = generate_embedding(test_text)
    
    print("\n" + "="*50)
    
    # Test 3: Test similarity between related texts
    test_embedding_similarity(
        "The cat is sleeping on the couch",
        "A feline is resting on the sofa"
    )
    
    print("\n" + "="*50)
    
    # Test 4: Test similarity between unrelated texts
    test_embedding_similarity(
        "The weather is sunny today",
        "I love programming in Python"
    )
    
    print("\n✅ All tests completed!")