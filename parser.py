"""
Credit Card Statement Parser
Extracts 5 key data points from credit card statements across 5 major issuers.
Returns structured JSON format.
"""

import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Define the structured output schema
class CreditCardStatement(BaseModel):
    """Schema for credit card statement data extraction"""
    card_issuer: str = Field(description="Name of the credit card issuer (e.g., HDFC, ICICI, SBI, Axis, American Express)")
    card_last_four_digits: str = Field(description="Last 4 digits of the credit card number")
    billing_cycle: str = Field(description="Billing cycle period (e.g., '01 Sep 2024 - 30 Sep 2024')")
    payment_due_date: str = Field(description="Payment due date (e.g., '20 Oct 2024')")
    total_amount_due: str = Field(description="Total amount due/outstanding balance (e.g., 'Rs. 25,432.50' or '$1,234.56')")

class CreditCardParser:
    """Parser for extracting structured data from credit card statements"""
    
    def __init__(self):
        """Initialize the parser with LLM"""
        self.llm = ChatOpenAI(
            model=os.getenv("GENERATIVE_MODEL", "anthropic/claude-3-haiku"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=500,
            default_headers={"HTTP-Referer": "http://localhost"}
        )
        
        # JSON parser
        self.json_parser = JsonOutputParser(pydantic_object=CreditCardStatement)
        
        # Extraction prompt
        self.extraction_prompt = PromptTemplate(
            template="""You are an expert at extracting structured data from credit card statements.

Analyze the following credit card statement text and extract EXACTLY these 5 data points:

1. **card_issuer**: The name of the bank/credit card company (e.g., HDFC Bank, ICICI Bank, SBI Card, Axis Bank, American Express, etc.)
2. **card_last_four_digits**: The last 4 digits of the credit card number (format: "XXXX")
3. **billing_cycle**: The billing period (format: "DD MMM YYYY - DD MMM YYYY" or similar)
4. **payment_due_date**: The payment due date (format: "DD MMM YYYY" or similar)
5. **total_amount_due**: The total amount due/outstanding balance with currency symbol (format: "Rs. X,XXX.XX" or "$X,XXX.XX")

IMPORTANT RULES:
- Extract information EXACTLY as it appears in the statement
- For amounts, include the currency symbol and formatting
- If a field is not found, use "Not Found" as the value
- Be precise with dates and numbers
- Look for common labels like "Statement Period", "Payment Due", "Total Amount Due", "Card Number", etc.
- For Indian cards, look for amounts in Rupees (Rs. or ₹)
- For international cards, look for amounts in dollars ($) or other currencies

{format_instructions}

Credit Card Statement Text:
{text}

Extracted Data (JSON):""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()}
        )
        
        # Create extraction chain
        self.extraction_chain = self.extraction_prompt | self.llm | self.json_parser
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Limit text length to avoid token limits (first 5000 chars usually contain key info)
            if len(full_text) > 5000:
                full_text = full_text[:5000]
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def parse_statement(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse a credit card statement PDF and extract structured data.
        
        Args:
            pdf_path: Path to the credit card statement PDF
            
        Returns:
            Dictionary containing extracted data points
        """
        try:
            print(f"📄 Processing: {pdf_path}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            print(f"✅ Extracted {len(text)} characters from PDF")
            
            # Use LLM to extract structured data
            print("🔍 Extracting structured data...")
            result = self.extraction_chain.invoke({"text": text})
            
            # Add metadata
            result["source_file"] = Path(pdf_path).name
            result["processing_timestamp"] = datetime.now().isoformat()
            
            print("✅ Extraction complete")
            return result
            
        except Exception as e:
            print(f"❌ Error parsing statement: {e}")
            return {
                "error": str(e),
                "source_file": Path(pdf_path).name,
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def parse_multiple_statements(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple credit card statements.
        
        Args:
            pdf_paths: List of paths to credit card statement PDFs
            
        Returns:
            List of dictionaries containing extracted data for each statement
        """
        results = []
        
        for pdf_path in pdf_paths:
            result = self.parse_statement(pdf_path)
            results.append(result)
        
        return results
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF statements
            
        Returns:
            List of dictionaries containing extracted data for each statement
        """
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
        
        print(f"📁 Found {len(pdf_files)} PDF files in {directory_path}")
        
        return self.parse_multiple_statements([str(f) for f in pdf_files])


def main():
    """CLI interface for the parser"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python parser.py <pdf_file>              # Parse single file")
        print("  python parser.py <pdf_file1> <pdf_file2> # Parse multiple files")
        print("  python parser.py --dir <directory>       # Parse all PDFs in directory")
        return
    
    parser = CreditCardParser()
    
    # Check if directory mode
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("Error: Please provide directory path")
            return
        
        results = parser.parse_directory(sys.argv[2])
    else:
        # Parse individual files
        pdf_paths = sys.argv[1:]
        results = parser.parse_multiple_statements(pdf_paths)
    
    # Print results as formatted JSON
    print("\n" + "="*80)
    print("EXTRACTION RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Save to output file
    output_file = "credit_card_extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
