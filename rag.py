import os
import requests
import tempfile
import uvicorn
import threading
from fastapi import FastAPI, Depends, HTTPException, Security, APIRouter
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Import LangChain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the universal loader
from loaders import DocumentLoader, load_file, download_and_load
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings

# Import for TF-IDF embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Try to import FAISS, fallback to simple vector store if not available
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available, using simple vector store")

# Helper functions for cleaning LLM responses
def clean_llm_response(response: str) -> str:
    """
    Clean LLM response by removing excessive newlines and formatting artifacts.
    Preserves list structure but makes it more readable.
    """
    if not response:
        return response
    
    import re
    
    # Replace multiple newlines with a single space
    cleaned = response.replace('\n\n', ' ').replace('\n', ' ')
    
    # Remove excessive spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove common verbose preambles (optional - uncomment if you want more concise responses)
    # preambles = [
    #     r'^Based on the information provided in the context,?\s*',
    #     r'^According to the provided context:?\s*[-\s]*',
    #     r'^The context provides\s+',
    #     r'^From the context,?\s*'
    # ]
    # for pattern in preambles:
    #     cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Strip leading/trailing whitespace and quotes
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    
    return cleaned

def clean_response_list(responses: List[str]) -> List[str]:
    """
    Clean a list of LLM responses.
    """
    return [clean_llm_response(resp) for resp in responses]

# Simple TF-IDF Embeddings class - optimized for speed
class TfidfEmbeddings(Embeddings):
    """Simple TF-IDF based embeddings using scikit-learn - optimized for performance"""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = None
        self.fitted = False
        self._corpus = []
    
    def _ensure_fitted(self, texts: List[str]):
        """Ensure the vectorizer is fitted on a corpus"""
        if not self.fitted:
            # Combine with any existing corpus
            all_texts = self._corpus + texts
            
            # Dynamic parameters based on corpus size
            num_docs = len(all_texts)
            if num_docs == 1:
                # For single document, use very permissive settings
                min_df = 1
                max_df = 1.0
                max_features = min(self.max_features, 100)  # Limit features for small corpus
            elif num_docs <= 5:
                # For very small corpus
                min_df = 1
                max_df = 1.0
                max_features = min(self.max_features, 500)
            else:
                # For larger corpus, use original settings
                min_df = 1
                max_df = 0.95
                max_features = self.max_features
            
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 1),  # Reduced from (1, 2) for speed
                min_df=min_df,
                max_df=max_df,
                norm='l2',  # Explicit normalization
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True  # Use sublinear tf scaling for better performance
            )
            self.vectorizer.fit(all_texts)
            self.fitted = True
            self._corpus = all_texts
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using TF-IDF"""
        self._ensure_fitted(texts)
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert sparse matrix to dense more efficiently
        return tfidf_matrix.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using TF-IDF"""
        if not self.fitted:
            # If not fitted yet, return a zero vector
            return [0.0] * self.max_features
        
        tfidf_vector = self.vectorizer.transform([text])
        return tfidf_vector.toarray().flatten().tolist()

# Simple Vector Store as fallback when FAISS is not available
class SimpleVectorStore:
    """Simple vector store using cosine similarity with TF-IDF embeddings"""
    
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.doc_texts = [doc.page_content for doc in documents]
        self.doc_vectors = embeddings.embed_documents(self.doc_texts)
    
    def similarity_search(self, query: str, k: int = 4):
        """Search for similar documents using cosine similarity"""
        query_vector = self.embeddings.embed_query(query)
        similarities = []
        
        for i, doc_vector in enumerate(self.doc_vectors):
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            
            if norm_query > 0 and norm_doc > 0:
                similarity = dot_product / (norm_query * norm_doc)
            else:
                similarity = 0
            
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]
    
    def as_retriever(self, search_kwargs=None):
        """Return a retriever interface"""
        k = search_kwargs.get('k', 4) if search_kwargs else 4
        return SimpleRetriever(self, k)

class SimpleRetriever:
    """Simple retriever interface for SimpleVectorStore"""
    
    def __init__(self, vectorstore, k=4):
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query: str):
        return self.vectorstore.similarity_search(query, k=self.k)
    
    def __call__(self, query: str):
        return self.get_relevant_documents(query)

# --- Initial Setup & Global Objects ---
load_dotenv()

print("Loading model with maximum accuracy parameters...")
llm_model = ChatOpenAI(
    model=os.getenv("GENERATIVE_MODEL", "anthropic/claude-3-haiku"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,  # Reduced to 0 for maximum determinism and accuracy
    max_tokens=350,   # Increased for more detailed answers
    top_p=0.95,      # Slightly increased for better token selection
    frequency_penalty=0.0,  # No frequency penalty for factual responses
    presence_penalty=0.0,   # No presence penalty for factual responses
    default_headers={"HTTP-Referer": "http://localhost"}
)
print("✅ Advanced model loaded successfully.")

embeddings = TfidfEmbeddings(max_features=500)  # Reduced from 1000
print("✅ TF-IDF embeddings initialized (no external dependencies).")

def download_and_parse_document(doc_url: str) -> List[Document]:
    """
    Download and parse document from URL using universal loader.
    Supports multiple formats: PDF, DOCX, PPTX, images, ZIP, email, XML, etc.
    """
    try:
        print(f"📥 Downloading document from: {doc_url}")
        
        # Use universal loader to download and process
        text_content, detected_format = download_and_load(doc_url)
        
        print(f"✅ Document loaded - Format: {detected_format}")
        print(f"✅ Content length: {len(text_content)} characters")
        
        # Check if document is too large
        if len(text_content) > 2000000:  # 2MB text limit
            print(f"⚠️ Large document detected ({len(text_content)} chars), truncating...")
            text_content = text_content[:2000000]  # 2MB limit
            print(f"✅ Truncated to {len(text_content)} chars")
        
        # Create Document objects for compatibility with existing RAG pipeline
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split into chunks
        chunks = text_splitter.split_text(text_content)
        
        # Convert to Document objects with metadata
        docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": doc_url,
                    "format": detected_format,
                    "chunk": i,
                    "total_chunks": len(chunks)
                }
            )
            docs.append(doc)
        
        print(f"✅ Created {len(docs)} document chunks")
        return docs
        
    except Exception as e:
        raise Exception(f"Document download and parsing failed: {str(e)}")

def process_multi_format_document_and_questions(docs: List[Document], questions: List[str]) -> List[str]:
    """
    Core RAG processing function that works with pre-processed documents.
    Supports the existing Blaze & Deep Dive strategy.
    """
    start_time = datetime.now()
    
    try:
        print(f"🎯 Processing {len(docs)} chunks with Blaze & Deep Dive strategy")
        
        print("🔪 Optimizing document chunking for accuracy...")
        # Enhanced chunking strategy for better context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),  # Increased from 600 for better context
            chunk_overlap=int(int(os.getenv("CHUNK_SIZE", 800)) * 0.25),  # Increased overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # More comprehensive separators
        )
        
        try:
            chunks = text_splitter.split_documents(docs)
            
            # Limit number of chunks to prevent memory issues and timeouts
            max_chunks = 7500
            if len(chunks) > max_chunks:
                print(f"⚠️ Too many chunks ({len(chunks)}), limiting to {max_chunks}")
                chunks = chunks[:max_chunks]
            
            print(f"✅ Created {len(chunks)} chunks")
        except Exception as e:
            raise Exception(f"Document chunking failed: {str(e)}")
        
        if not chunks:
            raise Exception("Document processing resulted in zero chunks - document may be empty or corrupted")

        # Create vector store with progress monitoring
        print("🔍 Creating embeddings...")
        try:
            import threading
            
            # Use threading-based timeout for embeddings
            embedding_complete = threading.Event()
            embedding_result = {"vectorstore": None, "error": None}
            
            def create_embeddings():
                try:
                    if FAISS_AVAILABLE:
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        print("✅ Using FAISS vector store")
                    else:
                        vectorstore = SimpleVectorStore(chunks, embeddings)
                        print("✅ Using simple vector store (FAISS fallback)")
                    embedding_result["vectorstore"] = vectorstore
                except Exception as e:
                    embedding_result["error"] = e
                finally:
                    embedding_complete.set()
            
            # Start embedding creation in a separate thread
            embed_thread = threading.Thread(target=create_embeddings)
            embed_thread.daemon = True
            embed_thread.start()
            
            # Wait for completion or timeout (60 seconds)
            if embedding_complete.wait(timeout=60):
                if embedding_result["error"]:
                    raise embedding_result["error"]
                vectorstore = embedding_result["vectorstore"]
            else:
                raise TimeoutError("Embedding creation timeout")
            
            print(f"✅ Vector store created with {len(chunks)} embeddings")
            
        except TimeoutError:
            raise Exception("Embedding creation timeout - too many chunks")
        except Exception as e:
            raise Exception(f"Vector store creation failed: {str(e)}")
        
        # Continue with the existing RAG processing logic...
        return process_questions_with_rag(vectorstore, chunks, questions)
        
    except Exception as e:
        print(f"❌ Error in multi-format processing: {e}")
        raise e

def process_questions_with_rag(vectorstore, chunks: List[Document], questions: List[str]) -> List[str]:
    """Extract the RAG processing logic into a separate function for reusability."""
    
    # Enhanced RAG prompt for maximum accuracy
    RAG_PROMPT = PromptTemplate.from_template("""You are a meticulous expert analyst with a PhD in document analysis. Your task is to provide highly accurate, fact-based answers using ONLY the provided context.

CRITICAL ACCURACY GUIDELINES:
- ONLY use information explicitly stated in the context - never infer or assume
- Quote exact numbers, percentages, dates, and specific terms when available
- If multiple pieces of information relate to the question, organize them clearly with bullet points
- Distinguish between facts, requirements, and conditional statements
- If information is partial or unclear, state exactly what is unclear
- If the context doesn't contain the specific information requested, state: "The provided context does not contain [specific information requested]"

RESPONSE STRUCTURE:
1. Direct answer to the question (if available in context)
2. Supporting details and specifics from the context
3. Any relevant conditions, limitations, or qualifications mentioned
4. Clear statement if information is incomplete or missing

Context:
{context}

Question: {question}

Accurate Answer:""")

    # --- Enhanced Triage Pipeline ---
    triage_prompt = PromptTemplate.from_template("""You are a strict quality control specialist evaluating answer accuracy and completeness.

EVALUATION CRITERIA:
- GOOD: Answer contains specific, factual information directly from the context (numbers, names, dates, processes, etc.)
- GOOD: Answer addresses the question with concrete details
- GOOD: Answer acknowledges limitations when context is insufficient
- BAD: Answer is vague, generic, or lacks specific details
- BAD: Answer contains obvious refusals like "I cannot answer" without trying to extract available information
- BAD: Answer appears to contain information not present in the context (hallucination)

Respond with EXACTLY one word: "GOOD" or "BAD"

Question: {question}
Answer: {answer}

Evaluation:""")
    triage_chain = triage_prompt | llm_model | StrOutputParser()

    # --- Enhanced Deep Dive Re-ranker ---
    json_parser = JsonOutputParser(pydantic_object=RerankScore)
    rerank_prompt = PromptTemplate(
        template="""You are a document relevance expert. Evaluate how well this document chunk answers the specific question.

SCORING CRITERIA (0-10):
- 10: Document directly and completely answers the question with specific details
- 8-9: Document contains most of the information needed with good specifics
- 6-7: Document contains relevant information but may be incomplete or partially relevant
- 4-5: Document contains some related information but doesn't directly address the question
- 1-3: Document mentions the topic but provides little useful information for the question
- 0: Document is completely irrelevant to the question

Consider:
- Exact keyword matches and semantic relevance
- Completeness of information for answering the question
- Specificity and detail level in the document
- How directly the document addresses the question

Respond with ONLY a JSON object containing a single "score" key with an integer value 0-10.

{format_instructions}

Question: {question}
Document: {context}

Relevance Score:""", 
        input_variables=["question", "context"], 
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )
    rerank_chain = rerank_prompt | llm_model | json_parser

    # --- Create Chains with Enhanced Retrieval for Accuracy ---
    # Blaze retriever with more chunks for better context
    blaze_retriever = vectorstore.as_retriever(search_kwargs={'k': 5})  # Increased from 3
    blaze_chain = ({"context": blaze_retriever, "question": RunnablePassthrough()}) | RAG_PROMPT | llm_model | StrOutputParser()

    # Enhanced BM25 retriever for better keyword matching
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20  # Increased from 15 for better recall
    
    # Create ensemble retriever with optimized weights for accuracy
    if FAISS_AVAILABLE:
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # Increased from 15
        # Adjusted weights to favor BM25 for exact matches
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.75, 0.25])
    else:
        # Use simple vector store retriever
        simple_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, simple_retriever], weights=[0.75, 0.25])
    
    # --- EXECUTION LOGIC ---
    final_answers = {}
    questions_to_process = questions

    print("📊 PASS 1: Running 'Blaze' answers...")
    blaze_answers = blaze_chain.batch(questions_to_process, {"max_concurrency": 15})
    
    triage_inputs = [{"question": q, "answer": a} for q, a in zip(questions_to_process, blaze_answers)]
    triage_results = triage_chain.batch(triage_inputs, {"max_concurrency": 15})
    
    questions_for_deep_dive = []
    for i, result in enumerate(triage_results):
        question = questions_to_process[i]
        if "GOOD" in result.upper():
            final_answers[question] = blaze_answers[i]
            print(f"✅ Blaze success for Q{i+1}")
        else:
            questions_for_deep_dive.append(question)
            print(f"🔄 Deep dive needed for Q{i+1}")

    if questions_for_deep_dive:
        print(f"🚀 PASS 2: Running 'Deep Dive' on {len(questions_for_deep_dive)} difficult questions...")
        
        # Step 1: Retrieve documents for all difficult questions
        retrieved_docs = ensemble_retriever.batch(questions_for_deep_dive, {"max_concurrency": 8})

        # Step 2: Re-rank documents for each question
        final_contexts = []
        for i, docs_list in enumerate(retrieved_docs):
            question = questions_for_deep_dive[i]
            if not docs_list:
                final_contexts.append("")
                continue
            
            rerank_inputs = [{"question": question, "context": doc.page_content} for doc in docs_list]
            scores = rerank_chain.batch(rerank_inputs, {"max_concurrency": 15})
            
            scored_docs = []
            for j, score_dict in enumerate(scores):
                try:
                    score = score_dict.get('score')
                    # Lowered threshold from 6 to 5 for better recall, but still filtering low relevance
                    if score is not None and score >= 5:
                        scored_docs.append((docs_list[j], score))
                except (AttributeError, ValueError):
                    continue
            
            if not scored_docs:
                # Enhanced Safety Net: use top 3 original docs and log the issue
                print(f"⚠️ Re-ranking found no relevant docs for question: {question[:50]}...")
                top_docs = docs_list[:3]  # Increased from 2 to 3
            else:
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                # Take top 4 documents instead of 3 for more comprehensive context
                top_docs = [doc for doc, score in scored_docs[:4]]
                print(f"✅ Selected {len(top_docs)} highly relevant docs (scores: {[score for _, score in scored_docs[:4]]})")

            # Enhanced context combination with clear separators
            contexts = []
            for i, doc in enumerate(top_docs):
                contexts.append(f"--- Document {i+1} ---\n{doc.page_content}")
            final_contexts.append("\n\n".join(contexts))

        # Step 3: Generate final answers using the curated contexts
        deep_dive_rag_chain = RAG_PROMPT | llm_model | StrOutputParser()
        deep_dive_inputs = [{"context": c, "question": q} for c, q in zip(final_contexts, questions_for_deep_dive)]
        deep_dive_answers = deep_dive_rag_chain.batch(deep_dive_inputs, {"max_concurrency": 8})

        for question, answer in zip(questions_for_deep_dive, deep_dive_answers):
            final_answers[question] = answer

    # Re-order the answers to match the original question order
    ordered_answers = [final_answers.get(q, "An error occurred processing this question.") for q in questions]
    
    # Clean the responses before returning
    cleaned_answers = clean_response_list(ordered_answers)
    
    return cleaned_answers

# Core RAG processing function that can be used by both FastAPI and Gradio
def process_document_and_questions(doc_url: str, questions: List[str]) -> List[str]:
    """
    Core RAG processing function - uses multi-format document loader.
    Supports PDF, DOCX, PPTX, images (with OCR), ZIP, email, XML, and more.
    """
    try:
        print(f"🎯 Processing document with Blaze & Deep Dive strategy: {doc_url}")
        
        # Use multi-format processing directly
        docs = download_and_parse_document(doc_url)
        return process_multi_format_document_and_questions(docs, questions)
            
    except Exception as e:
        print(f"❌ Error in document processing: {e}")
        raise e

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str] = Field(..., min_items=1, max_items=20)

class HackRxResponse(BaseModel):
    answers: List[str]

class RerankScore(BaseModel):
    score: int = Field(..., description="The relevance score from 0 to 10.")

app = FastAPI(
    title="Advanced RAG API - Blaze & Deep Dive Strategy with Multi-Format Support",
    description="Enterprise-grade RAG system with intelligent triage, deep dive processing, and comprehensive document format support (PDF, DOCX, PPTX, images with OCR, ZIP, email, XML, and more)",
    version="2.1.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
router = APIRouter(prefix="/api/v1")
security = HTTPBearer()
API_KEY = os.getenv("BEARER_TOKEN", "7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post("/hackrx/run", response_model=HackRxResponse)
def run_hackrx_job(
    request: HackRxRequest,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Advanced RAG endpoint with Blaze & Deep Dive strategy and Multi-Format Support:
    - Fast "Blaze" processing for simple questions
    - Intelligent triage to identify questions needing deep dive
    - Deep dive with ensemble retrieval and re-ranking for complex questions
    - Supports PDF, DOCX, PPTX, images (with OCR), ZIP, email, XML, and more
    """
    try:
        doc_url = str(request.documents)
        questions = request.questions
        
        # Use the multi-format processing by default
        docs = download_and_parse_document(doc_url)
        answers = process_multi_format_document_and_questions(docs, questions)
        
        return HackRxResponse(answers=answers)

    except Exception as e:
        import traceback
        print(f"❌ An error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "strategy": "blaze_and_deep_dive",
        "version": "2.1.0"
    }

@router.get("/stats")
def get_system_stats():
    """Get system statistics"""
    return {
        "system_info": {
            "version": "2.1.0",
            "strategy": "blaze_and_deep_dive",
            "features": [
                "fast_blaze_processing",
                "intelligent_triage",
                "ensemble_retrieval",
                "document_reranking",
                "deep_dive_processing",
                "multi_format_support"  # Added new feature
            ]
        },
        "timestamp": datetime.now().isoformat()
    }

app.include_router(router)

# Export the app for deployment
app = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting Advanced RAG API - Blaze & Deep Dive Strategy with Multi-Format Support on port {port}")
    print("🎯 Features: Fast Blaze processing, Intelligent triage, Deep dive with ensemble retrieval")
    print("📊 Focus: Speed for simple questions, accuracy for complex ones")
    print("📋 Supported formats: PDF, DOCX, PPTX, images (OCR), ZIP, email, XML, and more")
    print("🌐 Main endpoint: /api/v1/hackrx/run")
    
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=port,
        reload=False
    )
