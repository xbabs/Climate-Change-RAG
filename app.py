import streamlit as st
import chromadb
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Function to load the API key securely
def load_api_key():
    # In a real deployment scenario, you would load this from environment variables
    # For local testing and Colab, you can use userdata, but for deployment
    # on platforms like Streamlit Cloud, environment variables are recommended.
    try:
        # Attempt to load from environment variable first (recommended for deployment)
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key:
            print("Google Generative AI API key loaded from environment variable.")
            return api_key
        else:
            # Fallback for Colab environment (won't work in standalone deployment)
            from google.colab import userdata
            api_key = userdata.get('GOOGLE_API_KEY')
            if api_key:
                print("Google Generative AI API key loaded from Colab Secrets Manager.")
                return api_key
            else:
                st.error("GOOGLE_API_KEY not found. Please set it as an environment variable or in Colab Secrets.")
                return None
    except Exception as e:
        st.error(f"An error occurred while loading GOOGLE_API_KEY: {e}")
        return None

# Set up the Google Generative AI API key
GOOGLE_API_KEY = load_api_key()
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.stop() # Stop the app if API key is not loaded

# Initialize ChromaDB client (in-memory for this example)
# For persistent storage in a deployed app, you might need a different setup
@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client()
    collection_name = "climate_change_papers"
    try:
        # Try to get the collection if it exists
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection '{collection_name}'.")
    except:
        # If collection doesn't exist, create it
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")
        # Note: In a real app, you would load and process your data here
        # and add it to the collection if it's newly created.
        # For this example, we assume the collection is populated elsewhere or is empty initially.
    return collection

collection = get_chroma_collection()

# Load a pre-trained sentence transformer model
@st.cache_resource
def get_sentence_transformer_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence Transformer model loaded.")
    return model

model = get_sentence_transformer_model()


# --- Start of retrieve_documents function definition ---
def retrieve_documents(query, collection, model, n_results=5):
    """
    Retrieves relevant documents from the ChromaDB collection based on a query.
    """
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'distances'] # Include document text and similarity distances
    )

    # Format the results
    retrieved_docs = []
    if results and results['ids'] and results['documents']:
        for i in range(len(results['ids'][0])):
            retrieved_docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i]
            })
    return retrieved_docs
# --- End of retrieve_documents function definition ---


# --- Start of generate_answer_gemini function definition ---
def generate_answer_gemini(query, retrieved_documents):
    """
    Generates an answer to the query based on the retrieved documents using Google's Gemini model.
    """
    if not retrieved_documents:
        return "I could not find any relevant information to answer your question."

    # Combine the retrieved document texts into a single context
    context = "\n---\n".join([doc['text'] for doc in retrieved_documents])

    try:
        # Initialize the Gemini model
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Craft the prompt for the language model
        prompt = f"""Answer the following question based on the context below:

Question: {query}

Context:
{context}

Answer:
"""
        # Call the Gemini API to generate the answer
        response = gemini_model.generate_content(prompt)

        return response.text

    except Exception as e:
        print(f"An error occurred during text generation with Gemini: {e}")
        return "Sorry, I could not generate an answer at this time."
# --- End of generate_answer_gemini function definition ---


# --- Start of rag_answer_query function definition ---
def rag_answer_query(query, collection, model):
    """
    Answers a user query using the RAG approach.
    """
    # 1. Retrieve relevant documents
    retrieved_documents = retrieve_documents(query, collection, model)

    # 2. Generate answer based on retrieved documents
    generated_answer = generate_answer_gemini(query, retrieved_documents)

    return generated_answer
# --- End of rag_answer_query function definition ---


# Streamlit App Interface
st.title("Climate Change RAG Assistant")

st.write("Ask a question about climate change based on the loaded documents.")

user_query = st.text_input("Enter your query:")

if user_query:
    # Add a spinner while processing
    with st.spinner("Searching and generating answer..."):
        # Assume the collection is already populated from a previous run or a separate script
        # In a real application, you would need to handle data loading and embedding
        # population more robustly, potentially outside the Streamlit app or on first run.

        # For this example, we'll use the existing 'collection' and 'model' loaded
        # If the collection is empty, the retrieval will return nothing, and the
        # generation function will handle it.

        if collection.count() == 0:
             st.warning("The ChromaDB collection is empty. Please run the data loading and embedding steps first.")
        else:
            answer = rag_answer_query(user_query, collection, model)
            st.subheader("RAG Assistant's Answer:")
            st.write(answer)
