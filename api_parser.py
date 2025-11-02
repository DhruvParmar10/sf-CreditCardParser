"""
FastAPI endpoint for Credit Card Statement Parser
Provides REST API for extracting structured data from credit card statements.
Enhanced with RAG capabilities for answering questions about the statements.
"""

import os
import tempfile
import requests
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv
import uvicorn

from parser import CreditCardParser
from rag import download_and_parse_document, process_multi_format_document_and_questions, clean_response_list

load_dotenv()

app = FastAPI(
    title="Credit Card Statement Parser API with RAG",
    description="Extract structured data from credit card statements (5 key data points from 5+ issuers) and answer questions using RAG",
    version="2.0.0"
)

security = HTTPBearer()
API_KEY = os.getenv("BEARER_TOKEN", "7c695e780a6ab6eacffab7c9326e5d8e472a634870a6365979c5671ad28f003c")

# Initialize parser
parser = CreditCardParser()

# Request/Response models
class ParseURLRequest(BaseModel):
    """Request model for parsing from URL"""
    url: HttpUrl = Field(..., description="URL of the credit card statement PDF")

class ParseResponse(BaseModel):
    """Response model for parsed data"""
    card_issuer: str
    card_last_four_digits: str
    billing_cycle: str
    payment_due_date: str
    total_amount_due: str
    source_file: str
    processing_timestamp: str

class ParseWithQuestionsRequest(BaseModel):
    """Request model for parsing with questions"""
    url: HttpUrl = Field(..., description="URL of the credit card statement PDF")
    questions: Optional[List[str]] = Field(default=None, max_length=20, description="Optional questions about the statement")

class ParseWithQuestionsResponse(BaseModel):
    """Response model for parsed data with Q&A"""
    card_issuer: str
    card_last_four_digits: str
    billing_cycle: str
    payment_due_date: str
    total_amount_due: str
    source_file: str
    processing_timestamp: str
    answers: Optional[List[str]] = Field(default=None, description="Answers to questions (if questions were provided)")

class BatchParseURLRequest(BaseModel):
    """Request model for batch parsing from URLs"""
    urls: List[HttpUrl] = Field(..., min_length=1, max_length=10, description="List of PDF URLs (max 10)")

class BatchParseResponse(BaseModel):
    """Response model for batch parsing"""
    results: List[ParseResponse]
    total_processed: int
    successful: int
    failed: int

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key"""
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def download_pdf(url: str) -> str:
    """Download PDF from URL to temporary file"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            return temp_file.name
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

@app.post("/parse/upload", response_model=ParseResponse)
async def parse_upload(
    file: UploadFile = File(..., description="Credit card statement PDF file"),
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Parse a credit card statement from uploaded PDF file.
    
    Extracts 5 key data points:
    1. Card issuer name
    2. Last 4 digits of card
    3. Billing cycle
    4. Payment due date
    5. Total amount due
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Parse the statement
        result = parser.parse_statement(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

@app.post("/parse/url", response_model=ParseResponse)
async def parse_url(
    request: ParseURLRequest,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Parse a credit card statement from a URL.
    
    Extracts 5 key data points:
    1. Card issuer name
    2. Last 4 digits of card
    3. Billing cycle
    4. Payment due date
    5. Total amount due
    """
    try:
        # Download PDF
        temp_path = download_pdf(str(request.url))
        
        # Parse the statement
        result = parser.parse_statement(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

@app.post("/parse/batch", response_model=BatchParseResponse)
async def parse_batch(
    request: BatchParseURLRequest,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Parse multiple credit card statements from URLs (batch processing).
    
    Maximum 10 statements per request.
    """
    results = []
    successful = 0
    failed = 0
    
    for url in request.urls:
        try:
            # Download PDF
            temp_path = download_pdf(str(url))
            
            # Parse the statement
            result = parser.parse_statement(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            if "error" not in result:
                successful += 1
            else:
                failed += 1
            
            results.append(result)
            
        except Exception as e:
            failed += 1
            results.append({
                "error": str(e),
                "source_file": str(url),
                "processing_timestamp": ""
            })
    
    return {
        "results": results,
        "total_processed": len(request.urls),
        "successful": successful,
        "failed": failed
    }

@app.post("/parse/upload-with-questions", response_model=ParseWithQuestionsResponse)
async def parse_upload_with_questions(
    file: UploadFile = File(..., description="Credit card statement PDF file"),
    questions: Optional[str] = None,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Upload and parse a credit card statement, optionally answer questions using RAG.
    
    Extracts 5 key data points:
    1. Card issuer name
    2. Last 4 digits of card
    3. Billing cycle
    4. Payment due date
    5. Total amount due
    
    Additionally, if questions are provided (as JSON array string), uses RAG to answer them.
    Example questions parameter: '["What are the top 3 transactions?", "What is my credit limit?"]'
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Parse the statement for key data points
        result = parser.parse_statement(temp_path)
        
        if "error" in result:
            os.unlink(temp_path)
            raise HTTPException(status_code=500, detail=result["error"])
        
        # If questions are provided, use RAG to answer them
        answers = None
        if questions:
            try:
                import json
                # Parse questions from JSON string
                question_list = json.loads(questions)
                
                if not isinstance(question_list, list):
                    raise ValueError("Questions must be a JSON array")
                
                if len(question_list) > 20:
                    raise ValueError("Maximum 20 questions allowed")
                
                print(f"🔍 Processing {len(question_list)} questions using RAG...")
                
                # Load document for RAG from file path
                from loaders import load_file
                from langchain_core.documents import Document
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                text, _ = load_file(temp_path)
                
                # Convert text to Document objects for RAG processing
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk, metadata={"source": temp_path}) for chunk in chunks]
                
                # Process questions
                answers = process_multi_format_document_and_questions(docs, question_list)
                # Clean responses (already cleaned in rag.py, but double-check for safety)
                answers = clean_response_list(answers)
                
                print(f"✅ Answered {len(answers)} questions")
                
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Questions must be valid JSON array")
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                print(f"⚠️ RAG processing failed: {e}")
                # Don't fail the entire request if RAG fails
                answers = [f"Error answering question: {str(e)}" for _ in question_list]
        
        # Clean up
        os.unlink(temp_path)
        
        # Combine results
        response = {**result, "answers": answers}
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "credit_card_parser",
        "version": "1.0.0"
    }

@app.post("/parse/with-questions", response_model=ParseWithQuestionsResponse)
async def parse_with_questions(
    request: ParseWithQuestionsRequest,
    is_authenticated: bool = Depends(verify_api_key)
):
    """
    Parse a credit card statement and optionally answer questions using RAG.
    
    Extracts 5 key data points:
    1. Card issuer name
    2. Last 4 digits of card
    3. Billing cycle
    4. Payment due date
    5. Total amount due
    
    Additionally, if questions are provided, uses RAG to answer them based on the statement content.
    """
    try:
        # Download PDF
        temp_path = download_pdf(str(request.url))
        
        # Parse the statement for key data points
        result = parser.parse_statement(temp_path)
        
        if "error" in result:
            os.unlink(temp_path)
            raise HTTPException(status_code=500, detail=result["error"])
        
        # If questions are provided, use RAG to answer them
        answers = None
        if request.questions and len(request.questions) > 0:
            try:
                print(f"🔍 Processing {len(request.questions)} questions using RAG...")
                
                # Load document for RAG
                docs = download_and_parse_document(str(request.url))
                
                # Process questions
                answers = process_multi_format_document_and_questions(docs, request.questions)
                # Clean responses (already cleaned in rag.py, but double-check for safety)
                answers = clean_response_list(answers)
                
                print(f"✅ Answered {len(answers)} questions")
                
            except Exception as e:
                print(f"⚠️ RAG processing failed: {e}")
                # Don't fail the entire request if RAG fails
                answers = [f"Error answering question: {str(e)}" for _ in request.questions]
        
        # Clean up
        os.unlink(temp_path)
        
        # Combine results
        response = {**result, "answers": answers}
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Credit Card Statement Parser API with RAG",
        "version": "2.0.0",
        "endpoints": {
            "parse_upload": "POST /parse/upload - Upload and parse PDF (key data only)",
            "parse_upload_with_questions": "POST /parse/upload-with-questions - Upload PDF and answer questions using RAG",
            "parse_url": "POST /parse/url - Parse PDF from URL (key data only)",
            "parse_with_questions": "POST /parse/with-questions - Parse PDF and answer questions using RAG",
            "parse_batch": "POST /parse/batch - Batch parse from URLs",
            "health": "GET /health - Health check"
        },
        "data_points_extracted": [
            "card_issuer",
            "card_last_four_digits",
            "billing_cycle",
            "payment_due_date",
            "total_amount_due"
        ],
        "features": [
            "structured_data_extraction",
            "rag_question_answering",
            "blaze_and_deep_dive_strategy"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Credit Card Parser API with RAG on port {port}")
    print("📊 Extracting 5 key data points from credit card statements")
    print("🏦 Supporting 5+ major credit card issuers")
    print("💬 RAG-powered Q&A enabled - ask questions about statements!")
    print("🎯 Features: Structured extraction + Blaze & Deep Dive RAG strategy")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
