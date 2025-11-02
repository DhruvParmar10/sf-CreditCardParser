# Credit Card Statement Parser & RAG System

An enterprise-grade FastAPI solution for automated extraction of structured data from credit card statements, coupled with an advanced Retrieval-Augmented Generation (RAG) system utilizing the proprietary Blaze & Deep Dive strategy for intelligent document analysis and question answering.

## Overview

This system provides financial institutions and enterprises with robust capabilities for processing credit card statements and extracting actionable insights through natural language queries. The architecture combines state-of-the-art document parsing with intelligent retrieval mechanisms to deliver accurate, reliable results at scale.

## Core Capabilities

### 1. Structured Data Extraction Engine

Automated extraction of critical financial data points from credit card statements:

- Issuer identification
- Card number (last 4 digits)
- Billing cycle period
- Payment due date
- Total amount due

**Supported Institutions:** HDFC Bank, ICICI Bank, State Bank of India, Axis Bank, American Express, and additional major issuers.

**Output Format:** Structured JSON for seamless integration with downstream systems.

### 2. Intelligent RAG System with Dual-Mode Processing

**Blaze Mode**

- Optimized for straightforward queries requiring rapid response times
- Employs streamlined retrieval with focused context window (5 chunks)
- Suitable for high-throughput scenarios

**Intelligent Triage Layer**

- Automatic quality assessment of initial responses
- Dynamically routes complex queries to deep dive processing
- Ensures optimal balance between speed and accuracy

**Deep Dive Mode**

- Activated for queries requiring comprehensive analysis
- Ensemble retrieval strategy combining dual search methods:
  - **BM25 Keyword Search**: Statistical relevance ranking excels at exact term matching
  - **TF-IDF Vector Search**: Term frequency analysis captures document similarity patterns
  - Weighted fusion (75% BM25, 25% TF-IDF) optimized for precision
- Advanced document re-ranking with LLM-based relevance scoring (0-10 scale)
- Multi-document context assembly from top-4 performing segments
- Optimized for maximum accuracy in complex analytical queries

**Response Processing**

- Automated sanitization of LLM output
- Removal of formatting artifacts and excessive whitespace
- Ensures consistent, clean responses for production environments

### 3. Universal Document Processing

Enterprise-ready document ingestion supporting multiple formats:

- Portable Document Format (PDF) via PyMuPDF
- Microsoft Word documents (.docx)
- Microsoft PowerPoint presentations (.pptx)
- Microsoft Excel spreadsheets (.xlsx)
- Image files with OCR capabilities (JPEG, PNG, TIFF)
- Compressed archives with recursive processing (.zip)
- Email messages (.eml, .msg)
- XML structured documents
- Plain text files (.txt)

## System Architecture

```
sureF/
├── api_parser.py          # FastAPI endpoints for credit card parsing + RAG
├── parser.py              # Credit card statement parser logic
├── rag.py                 # Advanced RAG system with Blaze & Deep Dive
├── loaders.py             # Universal document loader for multiple formats
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── data/                  # Data directory
└── README.md             # This file
```

## Installation and Configuration

### System Requirements

- Python 3.8 or higher
- Tesseract OCR engine (for optical character recognition)
- Minimum 4GB RAM recommended
- Network connectivity for LLM API access

### Deployment Steps

1. **Clone or download the project:**

```bash
cd /path/to/sureF
```

2. **Create virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR (for image support):

   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Configure environment variables:**

Create a `.env` file in the project root:

```env
# Language Model Configuration
OPENROUTER_API_KEY=<your_api_key>
GENERATIVE_MODEL=anthropic/claude-3-haiku

# API Security
BEARER_TOKEN=<secure_token_string>

# Processing Parameters
CHUNK_SIZE=800
```

## API Reference

### Service Endpoints

The system exposes two primary API services:

1. **Credit Card Parser Service** (Port 8000)
2. **RAG Processing Service** (Port 7860)

### Starting Services

#### Credit Card Parser API

```bash
python api_parser.py
```

#### RAG Processing API

```bash
python rag.py
```

### REST API Endpoints

#### Parser Service Endpoints

**Parse Document from URL**

````
**Endpoint:** `POST /parse/url`

**Request Body:**
```json
{
  "url": "https://example.com/statement.pdf"
}
````

**Parse Uploaded Document**

```
**Endpoint:** `POST /parse/upload`

**Content-Type:** `multipart/form-data`

**Parse with Question Answering**
```

**Endpoint:** `POST /parse/with-questions`

```json
{
  "url": "https://example.com/statement.pdf",
  "questions": [
    "What are the top 3 transactions?",
    "What is my credit limit?",
    "Where did I spend the most?"
  ]
}
```

**Batch Processing**

````
**Endpoint:** `POST /parse/batch`

**Request Body:**
```json
{
  "urls": [
    "https://example.com/statement1.pdf",
    "https://example.com/statement2.pdf"
  ]
}
````

#### RAG Service Endpoints

**Document Analysis with Natural Language Queries**

````
**Endpoint:** `POST /api/v1/hackrx/run`

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "What are the key findings?",
    "What is the conclusion?"
  ]
}
````

**Service Health Monitoring**

```
**Endpoints:**
- `GET /health`
- `GET /api/v1/health`

**System Metrics**
```

**Endpoint:** `GET /api/v1/stats`

### Response Schemas

#### Structured Parsing Response

```json
{
  "card_issuer": "HDFC Bank",
  "card_last_four_digits": "5541",
  "billing_cycle": "01 Oct 2019 - 31 Oct 2019",
  "payment_due_date": "20 Nov 2019",
  "total_amount_due": "Rs. 225,000.00",
  "source_file": "statement.pdf",
  "processing_timestamp": "2024-11-02T10:30:00"
}
```

#### RAG Analysis Response

```json
{
  "answers": [
    "The top 3 transactions are: 1. BILLOCARD.COM, BARCELONA ES - 52,219.36 Dr on 24/10/2019 2. BILLOCARD.COM, BARCELONA ES - 51,974.40 Dr on 22/10/2019 3. BILLOCARD.COM, BARCELONA ES - 31,220.31 Dr on 25/10/2019",
    "The credit limit for the credit card number 45145700****5541 is 225,000.00.",
    "The highest spending appears to be on the BILLOARD.COM merchant category in Barcelona, ES."
  ]
}
```

## Technical Implementation

### Component Architecture

**parser.py - Extraction Engine**

- LLM-based structured output generation
- Multi-issuer compatibility layer
- Regex-based fallback mechanisms for robustness

**rag.py - Retrieval System**

- Dual-mode processing architecture (Blaze/Deep Dive)
- Quality assurance through automated triage
- Hybrid retrieval mechanisms:
  - BM25 keyword matching (20-chunk retrieval) - 75% weight
  - TF-IDF vector search (20-chunk retrieval) - 25% weight
  - Ensemble fusion with optimized weighting
  - LLM-based document re-ranking (0-10 relevance scoring)
  - Top-4 document selection for context assembly
- Production-grade response sanitization

**loaders.py - Document Ingestion Pipeline**

- Format detection using python-magic
- Recursive archive processing
- Optical character recognition for images
- Email parsing capabilities
- XML document handling

**api_parser.py - API Gateway**

- RESTful endpoint exposure
- Token-based authentication
- Batch processing optimization
- Integrated parser and RAG functionality

## Security

### Authentication Mechanism

All API endpoints require Bearer token authentication:

**Header Format:**

```
Authorization: Bearer <token>
```

**Example cURL Request:**

```bash
curl -X POST "http://localhost:8000/parse/url" \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/statement.pdf"}'
```

## Response Processing

### Automated Output Sanitization

The system implements comprehensive response cleaning to ensure production-ready output:

**Processing Steps:**

- Normalization of line breaks and whitespace
- Removal of redundant spacing patterns
- Trimming of leading/trailing whitespace
- Elimination of quotation mark artifacts
- Optional removal of verbose preambles (configurable)

**Example Transformation:**
**Input (Raw LLM Output):**

```
"Based on the information provided...\n\n1. Transaction...\n2. Transaction..."
```

**Output (Processed):**

```
"Based on the information provided... 1. Transaction... 2. Transaction..."
```

## Performance Characteristics

### Optimization Features

**Retrieval Strategy**

- TF-IDF statistical vectorization (no neural embeddings required)
- BM25 keyword-based relevance ranking
- Hybrid ensemble with weighted fusion (75% BM25, 25% TF-IDF)
- Optional FAISS integration for faster similarity search
- Zero-dependency fallback using pure Python cosine similarity

**Concurrency Management**

- Batch processing with configurable concurrency limits
- Asynchronous processing for I/O-bound operations

**Resource Management**

- Optimized chunking: 800 characters with 25% overlap
- Timeout protection: 60-second embeddings timeout
- Memory controls: 7500-chunk processing limit

**Scalability Metrics**

- Sub-second response times for Blaze mode queries
- Efficient handling of multi-document batch processing
- Graceful degradation under high load

## Quality Assurance

### Testing Procedures

**Validation Script:**

```python
from rag import clean_llm_response, clean_response_list

# Test response sanitization
test_input = "Based on...\n\n1. Item\n2. Item"
sanitized_output = clean_llm_response(test_input)
assert sanitized_output == "Based on... 1. Item 2. Item"
```

## Configuration Reference

### Environment Configuration

| Parameter            | Description              | Default Value              | Required |
| -------------------- | ------------------------ | -------------------------- | -------- |
| `OPENROUTER_API_KEY` | API key for LLM provider | -                          | Yes      |
| `GENERATIVE_MODEL`   | LLM model identifier     | `anthropic/claude-3-haiku` | No       |
| `BEARER_TOKEN`       | API authentication token | -                          | Yes      |
| `CHUNK_SIZE`         | Document chunking size   | `800`                      | No       |
| `PORT`               | Service port number      | `8000`/`7860`              | No       |

## Troubleshooting and Support

### Common Issues and Resolutions

**FAISS Unavailable**

The system includes automatic fallback to an internal vector store. For optimal performance in production environments, install FAISS:

```bash
pip install faiss-cpu
```

**Tesseract OCR Configuration**

Verify Tesseract installation and PATH configuration:

```bash
tesseract --version
```

**LLM API Connectivity**

Ensure valid API credentials are configured in `.env` file. Verify network connectivity to the LLM provider endpoint.

## Technology Stack

### Core Dependencies

**Framework and Infrastructure**

- FastAPI - High-performance web framework
- Uvicorn - ASGI server implementation
- Pydantic - Data validation and settings management

**Document Processing**

- PyMuPDF - PDF document handling
- python-docx - Microsoft Word processing
- python-pptx - PowerPoint document parsing
- pytesseract - OCR engine interface
- python-magic - File type detection

**Machine Learning and NLP**

- LangChain - RAG framework and orchestration
- scikit-learn - TF-IDF statistical vectorization (no deep learning required)
- rank_bm25 - BM25 keyword-based retrieval algorithm
- faiss-cpu - Optimized vector similarity search (optional, fallback available)

**Complete dependency manifest available in `requirements.txt`**

## Advanced Configuration

### Response Preamble Filtering

To enable removal of verbose preambles from LLM responses, modify `rag.py` (lines 61-67):

```python
# Uncomment to enable preamble removal
preambles = [
    r'^Based on the information provided in the context,?\s*',
    r'^According to the provided context:?\s*[-\s]*',
    r'^The context provides\s+',
    r'^From the context,?\s*'
]
for pattern in preambles:
    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
```

### Document Chunking Optimization

Adjust chunking parameters in `.env`:

```env
CHUNK_SIZE=1000  # Larger chunks for more context
```

### Alternative LLM Models

Configure different models via environment variables:

```env
GENERATIVE_MODEL=openai/gpt-4-turbo-preview
# or
GENERATIVE_MODEL=anthropic/claude-3-opus
```

## Production Deployment Considerations

### Best Practices

1. **Security**

   - Rotate bearer tokens regularly
   - Use environment-specific configurations
   - Implement rate limiting at the infrastructure level
   - Enable HTTPS/TLS in production

2. **Monitoring**

   - Implement application performance monitoring (APM)
   - Track API response times and error rates
   - Monitor LLM API usage and costs
   - Set up alerting for system health metrics

3. **Scalability**

   - Deploy behind a load balancer for horizontal scaling
   - Use containerization (Docker) for consistent deployments
   - Consider implementing caching for frequently accessed documents
   - Optimize database connections if adding persistence layer

4. **Reliability**
   - Implement circuit breakers for external API calls
   - Add request retry logic with exponential backoff
   - Configure appropriate timeout values
   - Maintain comprehensive logging for troubleshooting

## License

[Add your license here]

## Maintainers and Contributors

[Add contributor information here]

## Support and Contact

For technical support, bug reports, or feature requests:

- Submit issues via GitHub issue tracker
- Contact: [Add contact information]

For enterprise support and custom implementations, please reach out to the development team.

---

**System Version**: 2.1.0  
**Release Date**: November 2, 2024  
**Status**: Production Ready  
**API Stability**: Stable
