# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# https://cookbook.openai.com/examples/parse_pdf_docs_for_rag
# tools/adaptive_pdf_extraction.py

import os
import re
import base64
import asyncio
import json
import tempfile
import subprocess
import concurrent.futures
from typing import Dict, Union, Any, List, Tuple, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Core libraries
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

# LLM clients
import anthropic
from openai import OpenAI

# Your existing imports
from kdcube_ai_app.tools.processing import DataSourceExtractionResult

import logging
logger = logging.getLogger("AdaptivePDFExtractor")

# ===========================================
# Complexity Assessment Results
# ===========================================

@dataclass
class ComplexityAssessment:
    """Results from PDF complexity analysis"""
    complexity_score: float  # 0-100
    document_type: str  # research_paper, manual, report, simple_document
    math_density: str  # heavy, medium, light, none
    table_complexity: str  # complex, medium, simple, none
    layout_complexity: str  # complex, medium, simple
    page_count: int
    text_density: float  # 0-100
    image_ratio: float  # 0-100
    recommended_strategy: str  # cheap, balanced, premium
    metadata: Dict[str, Any]
    processing_time: float

    @property
    def needs_premium_extraction(self) -> bool:
        return self.complexity_score > 70 or self.math_density == "heavy"

    @property
    def can_use_cheap_extraction(self) -> bool:
        return self.complexity_score < 40 and self.layout_complexity == "simple"

@dataclass
class ExtractorInstance:
    """Wrapper for extractor instances to enable reuse"""
    extractor: 'PDFExtractorBase'
    extractor_type: str
    created_at: float
    last_used: float
    use_count: int

    def mark_used(self):
        self.last_used = time.time()
        self.use_count += 1

# ===========================================
# Abstract Base Classes
# ===========================================

class ComplexityAssessorProtocol(Protocol):
    """Protocol for complexity assessors"""

    def assess_complexity(self, content: Union[str, bytes], file_path: str) -> ComplexityAssessment:
        """Assess PDF complexity and return assessment"""
        ...

class PDFExtractorBase(ABC):
    """Base class for all PDF extractors"""

    def __init__(self, name: str):
        self.name = name
        self.supported_formats = ["pdf"]
        self.output_format = "original_text"  # Override in subclasses

    @abstractmethod
    def extract(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Extract content from PDF"""
        pass

    def can_handle_complexity(self, assessment: ComplexityAssessment) -> bool:
        """Check if this extractor can handle the given complexity"""
        return True  # Override in subclasses

    def get_processing_cost_estimate(self, assessment: ComplexityAssessment) -> float:
        """Get estimated processing cost (0-100 scale)"""
        return 50.0  # Override in subclasses

# ===========================================
# Complexity Assessors
# ===========================================

class CheapComplexityAssessor:
    """Fast, library-based complexity assessment"""

    def __init__(self):
        self.name = "cheap_assessor"

    def assess_complexity(self, content: Union[str, bytes], file_path: str) -> ComplexityAssessment:
        """Quick complexity assessment using PDF libraries"""
        start_time = time.time()

        try:
            # Write content to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                # Quick analysis with PyPDF2
                with open(tmp_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    page_count = len(reader.pages)

                    # Sample first few pages for analysis
                    sample_pages = min(3, page_count)
                    sample_text = ""

                    for i in range(sample_pages):
                        try:
                            page_text = reader.pages[i].extract_text()
                            sample_text += page_text + "\n"
                        except:
                            continue

                # Analyze text characteristics
                analysis = self._analyze_text_characteristics(sample_text, page_count)

                # Quick PDF structure analysis with pdfplumber
                structure_analysis = self._analyze_pdf_structure(tmp_path)

                # Combine analyses
                complexity_score = self._calculate_complexity_score(analysis, structure_analysis)

                processing_time = time.time() - start_time

                return ComplexityAssessment(
                    complexity_score=complexity_score,
                    document_type=analysis['document_type'],
                    math_density=analysis['math_density'],
                    table_complexity=structure_analysis['table_complexity'],
                    layout_complexity=structure_analysis['layout_complexity'],
                    page_count=page_count,
                    text_density=analysis['text_density'],
                    image_ratio=structure_analysis['image_ratio'],
                    recommended_strategy=self._recommend_strategy(complexity_score),
                    metadata={
                        "assessor": "cheap",
                        "sample_pages": sample_pages,
                        "analysis_details": {**analysis, **structure_analysis}
                    },
                    processing_time=processing_time
                )

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Cheap complexity assessment failed: {str(e)}")
            # Return default assessment
            return ComplexityAssessment(
                complexity_score=50.0,
                document_type="unknown",
                math_density="medium",
                table_complexity="medium",
                layout_complexity="medium",
                page_count=1,
                text_density=50.0,
                image_ratio=10.0,
                recommended_strategy="balanced",
                metadata={"assessor": "cheap", "error": str(e)},
                processing_time=time.time() - start_time
            )

    def _analyze_text_characteristics(self, text: str, page_count: int) -> Dict[str, Any]:
        """Analyze text content characteristics"""

        if not text.strip():
            return {
                'document_type': 'unknown',
                'math_density': 'none',
                'text_density': 0.0,
                'has_formulas': False,
                'has_citations': False
            }

        # Math detection patterns
        math_patterns = [
            r'[∂∆∇∑∏∫]',  # Mathematical symbols
            r'\$[^$]+\$',   # LaTeX inline math
            r'\$\$[^$]+\$\$',  # LaTeX display math
            r'[α-ωΑ-Ω]',   # Greek letters
            r'\b[f|g|h]\([x|y|z|t]\)',  # Function notation
            r'[±≤≥≠≈∞]',   # Mathematical operators
            r'\b\d+[.,]\d+[.,]\d+',  # Complex numbers/equations
        ]

        import re
        math_matches = sum(len(re.findall(pattern, text)) for pattern in math_patterns)
        text_length = len(text)

        # Determine math density
        if text_length > 0:
            math_ratio = math_matches / text_length * 1000  # Scale for readability
        else:
            math_ratio = 0

        if math_ratio > 15:
            math_density = "heavy"
        elif math_ratio > 5:
            math_density = "medium"
        elif math_ratio > 1:
            math_density = "light"
        else:
            math_density = "none"

        # Document type detection
        academic_indicators = [
            'abstract', 'introduction', 'methodology', 'results',
            'conclusion', 'references', 'theorem', 'lemma', 'proof'
        ]

        text_lower = text.lower()
        academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)

        if academic_score >= 4:
            document_type = "research_paper"
        elif any(word in text_lower for word in ['manual', 'guide', 'instruction']):
            document_type = "manual"
        elif any(word in text_lower for word in ['report', 'analysis', 'summary']):
            document_type = "report"
        else:
            document_type = "simple_document"

        # Text density (chars per page)
        text_density = min(100.0, (text_length / page_count) / 20)  # Normalize to 0-100

        return {
            'document_type': document_type,
            'math_density': math_density,
            'text_density': text_density,
            'has_formulas': math_matches > 0,
            'has_citations': 'et al.' in text or 'citation' in text_lower,
            'math_matches': math_matches,
            'academic_score': academic_score
        }

    def _analyze_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF structure using pdfplumber"""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_tables = 0
                total_images = 0
                complex_tables = 0
                multi_column_pages = 0

                # Analyze first few pages for structure
                sample_pages = min(3, len(pdf.pages))

                for i in range(sample_pages):
                    page = pdf.pages[i]

                    # Count tables
                    tables = page.extract_tables()
                    page_tables = len(tables) if tables else 0
                    total_tables += page_tables

                    # Check table complexity
                    if tables:
                        for table in tables:
                            if table and len(table) > 5 and len(table[0]) > 4:
                                complex_tables += 1

                    # Detect multi-column layout (heuristic)
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        short_lines = sum(1 for line in lines if len(line.strip()) < 50)
                        if short_lines / len(lines) > 0.6:  # Many short lines suggest columns
                            multi_column_pages += 1

                    # Count images (approximate)
                    try:
                        page_images = len(page.images) if hasattr(page, 'images') else 0
                        total_images += page_images
                    except:
                        pass

                # Determine complexities
                if total_tables > 5 or complex_tables > 2:
                    table_complexity = "complex"
                elif total_tables > 2:
                    table_complexity = "medium"
                elif total_tables > 0:
                    table_complexity = "simple"
                else:
                    table_complexity = "none"

                if multi_column_pages > 1:
                    layout_complexity = "complex"
                elif multi_column_pages > 0:
                    layout_complexity = "medium"
                else:
                    layout_complexity = "simple"

                # Image ratio
                pages_analyzed = max(1, sample_pages)
                image_ratio = min(100.0, (total_images / pages_analyzed) * 20)

                return {
                    'table_complexity': table_complexity,
                    'layout_complexity': layout_complexity,
                    'image_ratio': image_ratio,
                    'total_tables': total_tables,
                    'total_images': total_images,
                    'multi_column_pages': multi_column_pages
                }

        except Exception as e:
            logger.warning(f"PDF structure analysis failed: {str(e)}")
            return {
                'table_complexity': 'medium',
                'layout_complexity': 'medium',
                'image_ratio': 10.0,
                'total_tables': 0,
                'total_images': 0,
                'multi_column_pages': 0
            }

    def _calculate_complexity_score(self, text_analysis: Dict, structure_analysis: Dict) -> float:
        """Calculate overall complexity score"""

        score = 0.0

        # Math density contribution (0-30 points)
        math_scores = {"none": 0, "light": 10, "medium": 20, "heavy": 30}
        score += math_scores.get(text_analysis['math_density'], 15)

        # Table complexity (0-25 points)
        table_scores = {"none": 0, "simple": 8, "medium": 15, "complex": 25}
        score += table_scores.get(structure_analysis['table_complexity'], 10)

        # Layout complexity (0-25 points)
        layout_scores = {"simple": 0, "medium": 12, "complex": 25}
        score += layout_scores.get(structure_analysis['layout_complexity'], 10)

        # Document type (0-15 points)
        doc_scores = {"simple_document": 0, "report": 5, "manual": 8, "research_paper": 15}
        score += doc_scores.get(text_analysis['document_type'], 7)

        # Image ratio (0-5 points)
        score += min(5.0, structure_analysis['image_ratio'] / 20)

        return min(100.0, score)

    def _recommend_strategy(self, complexity_score: float) -> str:
        """Recommend extraction strategy based on complexity"""
        if complexity_score < 30:
            return "cheap"
        elif complexity_score < 70:
            return "balanced"
        else:
            return "premium"

class LLMComplexityAssessor:
    """LLM-based complexity assessment (quick metadata only)"""

    def __init__(self, provider: str = "anthropic", model: str = "claude-3-5-sonnet-20241022"):
        self.name = "llm_assessor"
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            self.client = OpenAI()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def assess_complexity(self, content: Union[str, bytes], file_path: str) -> ComplexityAssessment:
        """LLM-based complexity assessment using first page"""
        start_time = time.time()

        try:
            # Convert first page to image
            first_page_image = self._get_first_page_image(content)

            if not first_page_image:
                # Fallback to cheap assessment
                logger.warning("Could not extract first page image, falling back to cheap assessment")
                fallback = CheapComplexityAssessor()
                return fallback.assess_complexity(content, file_path)

            # Quick assessment prompt
            assessment_prompt = """Analyze this first page and provide ONLY complexity metadata as JSON.

Be very concise - just the key metrics needed for extraction strategy selection.

Respond with this exact JSON structure:
{
    "complexity_score": 75,
    "document_type": "research_paper|manual|report|simple_document",
    "math_density": "heavy|medium|light|none",
    "table_complexity": "complex|medium|simple|none", 
    "layout_complexity": "complex|medium|simple",
    "text_density": 85,
    "image_ratio": 15,
    "page_estimate": 10,
    "extraction_challenges": ["challenge1", "challenge2"],
    "recommended_strategy": "cheap|balanced|premium"
}"""

            # Call LLM
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=800,  # Keep it short
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": first_page_image
                                }
                            },
                            {"type": "text", "text": assessment_prompt}
                        ]
                    }]
                )
                response_text = response.content[0].text
            else:
                # OpenAI implementation would go here
                raise NotImplementedError("OpenAI LLM assessment not implemented yet")

            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            assessment_data = json.loads(response_text.strip())

            processing_time = time.time() - start_time

            return ComplexityAssessment(
                complexity_score=assessment_data.get("complexity_score", 50.0),
                document_type=assessment_data.get("document_type", "unknown"),
                math_density=assessment_data.get("math_density", "medium"),
                table_complexity=assessment_data.get("table_complexity", "medium"),
                layout_complexity=assessment_data.get("layout_complexity", "medium"),
                page_count=assessment_data.get("page_estimate", 1),
                text_density=assessment_data.get("text_density", 50.0),
                image_ratio=assessment_data.get("image_ratio", 10.0),
                recommended_strategy=assessment_data.get("recommended_strategy", "balanced"),
                metadata={
                    "assessor": "llm",
                    "provider": self.provider,
                    "model": self.model,
                    "extraction_challenges": assessment_data.get("extraction_challenges", []),
                    "llm_response": assessment_data
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"LLM complexity assessment failed: {str(e)}")
            # Fallback to cheap assessment
            fallback = CheapComplexityAssessor()
            result = fallback.assess_complexity(content, file_path)
            result.metadata["llm_assessment_error"] = str(e)
            result.metadata["fallback_used"] = True
            return result

    def _get_first_page_image(self, content: Union[str, bytes]) -> Optional[str]:
        """Convert first page of PDF to base64 image"""

        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                # Convert first page to image using PyMuPDF
                doc = fitz.open(tmp_path)
                if len(doc) == 0:
                    return None

                page = doc.load_page(0)  # First page

                # High DPI for better quality
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)

                # Convert to bytes
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode()

                doc.close()
                return base64_image

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to convert PDF first page to image: {str(e)}")
            return None

# ===========================================
# PDF Extractors
# ===========================================

class CheapPDFExtractor(PDFExtractorBase):
    """Fast, cheap extraction using basic libraries"""

    def __init__(self):
        super().__init__("cheap_extractor")
        self.output_format = "markdown"

    def extract(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Extract using PyPDF2 with rule-based markdown conversion"""

        try:
            # Write content to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                # Extract text with PyPDF2
                with open(tmp_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)

                    text_parts = []
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                text_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n")
                                text_parts.append(page_text)
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                            continue

                raw_text = "".join(text_parts)

                # Convert to markdown using rule-based approach
                markdown_content = self._text_to_markdown(raw_text)

                # Create metadata
                filename = os.path.basename(file_path) if file_path else "unknown.pdf"
                metadata = {
                    "source_file": file_path,
                    "filename": filename,
                    "extractor": self.name,
                    "text_format": "markdown",
                    "extraction_method": "pypdf2_with_markdown_rules",
                    "page_count": len(reader.pages),
                    "processing_time": 0,  # Could add timing
                    "quality_estimate": "basic"
                }

                return [DataSourceExtractionResult(
                    content=markdown_content,
                    metadata=metadata
                )]

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Cheap PDF extraction failed: {str(e)}")
            return []

    def _text_to_markdown(self, text: str) -> str:
        """Convert raw text to markdown using heuristic rules"""

        lines = text.split('\n')
        markdown_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                markdown_lines.append('')
                continue

            # Detect headings
            if self._is_heading(stripped):
                level = self._get_heading_level(stripped)
                markdown_lines.append(f"{'#' * level} {stripped}")

            # Detect lists
            elif self._is_list_item(stripped):
                markdown_lines.append(f"- {stripped}")

            # Regular paragraph
            else:
                markdown_lines.append(stripped)

        return '\n'.join(markdown_lines)

    def _is_heading(self, line: str) -> bool:
        """Detect if line is likely a heading"""
        return (
                (line.isupper() and len(line) < 80) or
                line.endswith(':') or
                (len(line) < 60 and not line.endswith('.')) or
                any(pattern in line.upper() for pattern in ['CHAPTER', 'SECTION', 'ABSTRACT', 'INTRODUCTION', 'CONCLUSION'])
        )

    def _get_heading_level(self, line: str) -> int:
        """Determine heading level"""
        if any(pattern in line.upper() for pattern in ['CHAPTER']):
            return 1
        elif any(pattern in line.upper() for pattern in ['SECTION', 'ABSTRACT', 'INTRODUCTION']):
            return 2
        else:
            return 3

    def _is_list_item(self, line: str) -> bool:
        """Detect list items"""
        import re
        return bool(re.match(r'^[\-\*\•]\s+', line) or re.match(r'^\d+[\.\)]\s+', line))

    def can_handle_complexity(self, assessment: ComplexityAssessment) -> bool:
        """Can handle simple documents"""
        return assessment.complexity_score < 50

    def get_processing_cost_estimate(self, assessment: ComplexityAssessment) -> float:
        """Very low cost"""
        return 5.0

class BalancedPDFExtractor(PDFExtractorBase):
    """Balanced extraction using pdfplumber with enhanced processing"""

    def __init__(self):
        super().__init__("balanced_extractor")
        self.output_format = "markdown"

    def extract(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Extract using pdfplumber with table detection and structure preservation"""

        try:
            # Write content to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                markdown_parts = []

                with pdfplumber.open(tmp_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        markdown_parts.append(f"\n\n--- Page {page_num + 1} ---\n\n")

                        # Extract tables first
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                if table:
                                    markdown_table = self._table_to_markdown(table)
                                    markdown_parts.append(markdown_table + "\n")

                        # Extract regular text
                        page_text = page.extract_text()
                        if page_text:
                            # Clean up text and convert to markdown
                            cleaned_text = self._clean_text(page_text)
                            markdown_text = self._text_to_markdown(cleaned_text)
                            markdown_parts.append(markdown_text)

                final_markdown = "\n".join(markdown_parts)

                # Create metadata
                filename = os.path.basename(file_path) if file_path else "unknown.pdf"
                metadata = {
                    "source_file": file_path,
                    "filename": filename,
                    "extractor": self.name,
                    "text_format": "markdown",
                    "extraction_method": "pdfplumber_with_tables",
                    "page_count": len(pdf.pages),
                    "processing_time": 0,
                    "quality_estimate": "good"
                }

                return [DataSourceExtractionResult(
                    content=final_markdown,
                    metadata=metadata
                )]

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Balanced PDF extraction failed: {str(e)}")
            return []

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert table to markdown format"""
        if not table or not table[0]:
            return ""

        markdown_rows = []

        # Header row
        header_row = [cell.strip() if cell else "" for cell in table[0]]
        markdown_rows.append("| " + " | ".join(header_row) + " |")

        # Separator row
        markdown_rows.append("| " + " | ".join(["---"] * len(header_row)) + " |")

        # Data rows
        for row in table[1:]:
            if row:
                clean_row = [cell.strip() if cell else "" for cell in row]
                # Pad row to match header length
                while len(clean_row) < len(header_row):
                    clean_row.append("")
                markdown_rows.append("| " + " | ".join(clean_row[:len(header_row)]) + " |")

        return "\n".join(markdown_rows)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Replace tab characters and clean up spacing
        text = text.replace('\t', ' ')
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text

    def _text_to_markdown(self, text: str) -> str:
        """Enhanced text to markdown conversion"""
        lines = text.split('\n')
        markdown_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                markdown_lines.append('')
                continue

            # Enhanced heading detection
            if self._is_heading(stripped):
                level = self._get_heading_level(stripped)
                markdown_lines.append(f"{'#' * level} {stripped}")

            # Enhanced list detection
            elif self._is_list_item(stripped):
                markdown_lines.append(f"- {stripped}")

            # Quote detection
            elif stripped.startswith('"') and stripped.endswith('"'):
                markdown_lines.append(f"> {stripped}")

            # Regular paragraph
            else:
                markdown_lines.append(stripped)

        return '\n'.join(markdown_lines)

    def _is_heading(self, line: str) -> bool:
        """Enhanced heading detection"""
        import re
        return (
                (line.isupper() and len(line) < 80) or
                re.match(r'^\d+\.?\s+[A-Z]', line) or
                (re.match(r'^[A-Z][A-Za-z\s]+$', line) and len(line) < 60) or
                any(pattern in line.upper() for pattern in [
                    'CHAPTER', 'SECTION', 'ABSTRACT', 'INTRODUCTION',
                    'METHODOLOGY', 'RESULTS', 'CONCLUSION', 'REFERENCES'
                ])
        )

    def _get_heading_level(self, line: str) -> int:
        """Enhanced heading level detection"""
        import re
        if re.match(r'^CHAPTER\s+', line.upper()):
            return 1
        elif any(pattern in line.upper() for pattern in ['ABSTRACT', 'INTRODUCTION', 'CONCLUSION']):
            return 2
        elif re.match(r'^\d+\.?\s+', line):
            return 3
        else:
            return 4

    def _is_list_item(self, line: str) -> bool:
        """Enhanced list item detection"""
        import re
        return bool(
            re.match(r'^[\-\*\•]\s+', line) or
            re.match(r'^\d+[\.\)]\s+', line) or
            re.match(r'^[a-z][\.\)]\s+', line) or
            re.match(r'^[A-Z][\.\)]\s+', line)
        )

    def can_handle_complexity(self, assessment: ComplexityAssessment) -> bool:
        """Can handle medium complexity documents"""
        return assessment.complexity_score < 80

    def get_processing_cost_estimate(self, assessment: ComplexityAssessment) -> float:
        """Medium cost"""
        return 25.0

class AnthropicPDFExtractor(PDFExtractorBase):
    """High-quality extraction using Anthropic Claude"""

    def __init__(self, model: str = "claude-3-7-sonnet-latest"):
        super().__init__("anthropic_extractor")
        self.output_format = "markdown"
        self.model = model
        self.client = anthropic.Anthropic()

    async def extract_async(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Async extraction method"""

        try:
            # Convert PDF to images
            images = self._pdf_to_images(content)

            if not images:
                logger.error("Could not convert PDF to images")
                return []

            # Process each page
            markdown_parts = []

            for i, image_data in enumerate(images):
                page_markdown = await self._extract_page_with_claude(image_data, i + 1)
                if page_markdown:
                    markdown_parts.append(f"\n\n--- Page {i + 1} ---\n\n")
                    markdown_parts.append(page_markdown)

            final_markdown = "\n".join(markdown_parts)

            # Create metadata
            filename = os.path.basename(file_path) if file_path else "unknown.pdf"
            metadata = {
                "source_file": file_path,
                "filename": filename,
                "extractor": self.name,
                "text_format": "markdown",
                "extraction_method": "anthropic_claude_multimodal",
                "model": self.model,
                "page_count": len(images),
                "processing_time": 0,
                "quality_estimate": "high"
            }

            return [DataSourceExtractionResult(
                content=final_markdown,
                metadata=metadata
            )]

        except Exception as e:
            logger.error(f"Anthropic PDF extraction failed: {str(e)}")
            return []

    def extract(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Sync wrapper for async extraction"""
        return asyncio.run(self.extract_async(content, file_path))

    def _pdf_to_images(self, content: Union[str, bytes]) -> List[str]:
        """Convert PDF to base64 images"""

        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                doc = fitz.open(tmp_path)
                images = []

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)

                    # High DPI for mathematical content
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to base64
                    img_data = pix.tobytes("png")
                    base64_image = base64.b64encode(img_data).decode()
                    images.append(base64_image)

                doc.close()
                return images

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            return []

    async def _extract_page_with_claude(self, image_data: str, page_num: int) -> str:
        """Extract single page content using Claude"""

        extraction_prompt = f"""Extract all content from this page {page_num} with perfect structure preservation.

**CRITICAL REQUIREMENTS**:
- Preserve ALL mathematical formulas exactly as they appear
- Maintain proper document structure (headings, lists, tables)
- Convert to clean markdown format
- Use LaTeX notation for mathematical content: $inline$ and $$display$$
- Preserve table structure with markdown tables
- Keep all text content, don't summarize

Output clean markdown with:
- Proper heading hierarchy (# ## ###)
- Lists with proper formatting
- Tables in markdown format
- Mathematical formulas in LaTeX
- Preserved text flow and structure

Extract everything visible on this page:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": extraction_prompt}
                    ]
                }]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude page extraction failed for page {page_num}: {str(e)}")
            return f"[Error extracting page {page_num}: {str(e)}]"

    def can_handle_complexity(self, assessment: ComplexityAssessment) -> bool:
        """Can handle any complexity"""
        return True

    def get_processing_cost_estimate(self, assessment: ComplexityAssessment) -> float:
        """High cost due to LLM usage"""
        return 80.0 + (assessment.page_count * 2.0)  # Base cost + per page

class OpenAIPDFExtractor(PDFExtractorBase):
    """High-quality extraction using OpenAI GPT-4V following cookbook pattern"""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__("openai_extractor")
        self.output_format = "markdown"
        self.model = model
        self.client = OpenAI()

        # Rate limiting parameters
        self.max_retries = 3
        self.retry_delay = 1.0

    async def extract_async(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Async extraction method following OpenAI cookbook pattern"""

        try:
            start_time = time.time()

            # Convert PDF to images
            images = self._pdf_to_images(content)

            if not images:
                logger.error("Could not convert PDF to images for OpenAI extraction")
                return []

            logger.info(f"Processing {len(images)} pages with OpenAI GPT-4V")

            # Process pages with rate limiting and error handling
            markdown_parts = []
            successful_pages = 0

            for i, image_data in enumerate(images):
                try:
                    page_markdown = await self._extract_page_with_openai(image_data, i + 1)
                    if page_markdown and page_markdown.strip():
                        markdown_parts.append(f"\n\n--- Page {i + 1} ---\n\n")
                        markdown_parts.append(page_markdown)
                        successful_pages += 1

                        # Add small delay to respect rate limits
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Failed to extract page {i + 1}: {str(e)}")
                    markdown_parts.append(f"\n\n--- Page {i + 1} (Error) ---\n\n")
                    markdown_parts.append(f"[Error extracting page {i + 1}: {str(e)}]\n")

            final_markdown = "\n".join(markdown_parts)
            processing_time = time.time() - start_time

            # Create metadata following the pattern
            filename = os.path.basename(file_path) if file_path else "unknown.pdf"
            metadata = {
                "source_file": file_path,
                "filename": filename,
                "extractor": self.name,
                "text_format": "markdown",
                "extraction_method": "openai_gpt4v_vision",
                "model": self.model,
                "page_count": len(images),
                "successful_pages": successful_pages,
                "processing_time": processing_time,
                "quality_estimate": "high",
                "extraction_stats": {
                    "total_pages": len(images),
                    "successful_pages": successful_pages,
                    "failed_pages": len(images) - successful_pages,
                    "success_rate": successful_pages / len(images) if images else 0
                }
            }

            return [DataSourceExtractionResult(
                content=final_markdown,
                metadata=metadata
            )]

        except Exception as e:
            logger.error(f"OpenAI PDF extraction failed: {str(e)}")
            return self._create_error_result(file_path, str(e))

    def extract(self, content: Union[str, bytes], file_path: str) -> List[DataSourceExtractionResult]:
        """Sync wrapper for async extraction"""
        try:
            return asyncio.run(self.extract_async(content, file_path))
        except Exception as e:
            logger.error(f"OpenAI sync extraction failed: {str(e)}")
            return self._create_error_result(file_path, str(e))

    def _pdf_to_images(self, content: Union[str, bytes]) -> List[str]:
        """Convert PDF to base64 images optimized for OpenAI API"""

        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                doc = fitz.open(tmp_path)
                images = []

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)

                    # Optimize image quality for OpenAI (balance quality vs token usage)
                    # OpenAI recommends reasonable resolution to avoid excessive token consumption
                    mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom (less than Anthropic's 2x for cost efficiency)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PNG bytes
                    img_data = pix.tobytes("png")

                    # Check image size (OpenAI has limits)
                    if len(img_data) > 20 * 1024 * 1024:  # 20MB limit
                        logger.warning(f"Page {page_num + 1} image too large, reducing quality")
                        # Reduce quality if too large
                        mat = fitz.Matrix(1.0, 1.0)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")

                    # Convert to base64
                    base64_image = base64.b64encode(img_data).decode()
                    images.append(base64_image)

                doc.close()
                return images

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to convert PDF to images for OpenAI: {str(e)}")
            return []

    async def _extract_page_with_openai(self, image_data: str, page_num: int) -> str:
        """Extract single page content using OpenAI GPT-4V with retry logic"""

        # Optimized prompt for OpenAI following cookbook pattern
        extraction_prompt = f"""You are an expert document analyst. Extract ALL content from this PDF page {page_num} with perfect accuracy and structure.

CRITICAL REQUIREMENTS:
- Extract ALL visible text exactly as it appears
- Preserve document structure: headings, paragraphs, lists, tables
- Convert mathematical formulas to LaTeX notation: $inline$ and $display$
- Format output as clean markdown
- Maintain proper heading hierarchy (# ## ###)
- Convert tables to markdown table format
- Preserve all technical content and citations

EXTRACTION GUIDELINES:
- Headers: Use appropriate markdown heading levels
- Lists: Use proper bullet points (-) or numbered lists (1.)
- Tables: Convert to markdown table format with | separators
- Math: Use LaTeX notation for formulas and equations
- Quotes: Use > for quotations
- Code: Use ``` for code blocks if any
- Citations: Preserve reference formatting

Output ONLY the markdown content, no commentary or explanations.

Extract everything visible on this page:"""

        for attempt in range(self.max_retries):
            try:
                # Prepare the message in OpenAI format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": extraction_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high"  # Use high detail for better extraction
                                }
                            }
                        ]
                    }
                ]

                # Make API call with optimized parameters
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4000,  # Sufficient for most pages
                    temperature=0,    # Deterministic extraction
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                # Extract content
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content and content.strip():
                        return content.strip()
                    else:
                        logger.warning(f"Empty response for page {page_num}")
                        return f"[Empty response for page {page_num}]"
                else:
                    logger.warning(f"No valid response for page {page_num}")
                    return f"[No valid response for page {page_num}]"

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for page {page_num}: {str(e)}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All attempts failed for page {page_num}: {str(e)}")
                    return f"[Error extracting page {page_num} after {self.max_retries} attempts: {str(e)}]"

        return f"[Failed to extract page {page_num}]"

    def _create_error_result(self, file_path: str, error_msg: str) -> List[DataSourceExtractionResult]:
        """Create error result when extraction fails"""

        filename = os.path.basename(file_path) if file_path else "unknown.pdf"
        metadata = {
            "source_file": file_path,
            "filename": filename,
            "extractor": self.name,
            "text_format": "markdown",
            "extraction_method": "openai_gpt4v_error",
            "model": self.model,
            "page_count": 0,
            "processing_time": 0,
            "quality_estimate": "failed",
            "error": error_msg
        }

        return [DataSourceExtractionResult(
            content=f"# Extraction Error\n\nOpenAI PDF extraction failed: {error_msg}",
            metadata=metadata
        )]

    def can_handle_complexity(self, assessment: ComplexityAssessment) -> bool:
        """Can handle any complexity"""
        return True

    def get_processing_cost_estimate(self, assessment: ComplexityAssessment) -> float:
        """High cost due to LLM usage - OpenAI pricing"""
        base_cost = 85.0

        # OpenAI GPT-4V pricing is higher for vision
        page_cost = 3.0  # Higher than Anthropic due to vision token costs

        # Additional cost for high-detail images
        image_detail_cost = assessment.page_count * 1.0

        return base_cost + (assessment.page_count * page_cost) + image_detail_cost

# ===========================================
# Main Adaptive Pipeline
# ===========================================

class AdaptivePDFExtractionPipeline:
    """Main pipeline orchestrating complexity assessment and extraction"""

    def __init__(self,
                 max_concurrent_cheap: int = 3,
                 enable_llm_assessment: bool = True,
                 preferred_llm_provider: str = "anthropic"):

        self.max_concurrent_cheap = max_concurrent_cheap
        self.enable_llm_assessment = enable_llm_assessment

        # Initialize assessors
        self.cheap_assessor = CheapComplexityAssessor()
        if enable_llm_assessment:
            self.llm_assessor = LLMComplexityAssessor(provider=preferred_llm_provider)
        else:
            self.llm_assessor = None

        # Initialize extractors
        self.extractors = {
            "cheap": CheapPDFExtractor(),
            "balanced": BalancedPDFExtractor(),
            "anthropic": AnthropicPDFExtractor(),
            "openai": OpenAIPDFExtractor()
        }

        # Instance cache for reuse
        self.extractor_instances: Dict[str, ExtractorInstance] = {}
        self.instance_lock = threading.Lock()

        # Thread pool for cheap processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_cheap)

    def extract_pdf(self,
                    content: Union[str, bytes],
                    file_path: str,
                    force_strategy: Optional[str] = None,
                    use_llm_assessment: Optional[bool] = None) -> Tuple[List[DataSourceExtractionResult], ComplexityAssessment]:
        """
        Main extraction method with adaptive strategy selection

        Args:
            content: PDF content as bytes or string
            file_path: Path to the PDF file
            force_strategy: Force specific strategy ("cheap", "balanced", "premium")
            use_llm_assessment: Override default LLM assessment setting

        Returns:
            Tuple of (extraction_results, complexity_assessment)
        """

        # Step 1: Assess complexity
        logger.info(f"Starting adaptive PDF extraction for: {file_path}")

        assessment = self._assess_complexity(content, file_path, use_llm_assessment)

        logger.info(f"Complexity assessment: score={assessment.complexity_score:.1f}, "
                    f"strategy={assessment.recommended_strategy}, type={assessment.document_type}")

        # Step 2: Select extraction strategy
        if force_strategy:
            strategy = force_strategy
            logger.info(f"Using forced strategy: {strategy}")
        else:
            strategy = self._select_extraction_strategy(assessment)
            logger.info(f"Selected strategy: {strategy}")

        # Step 3: Get or create extractor instance
        extractor = self._get_extractor_instance(strategy)

        if not extractor:
            logger.error(f"Could not get extractor for strategy: {strategy}")
            return [], assessment

        # Step 4: Extract
        try:
            logger.info(f"Extracting with {extractor.name}")
            extraction_results = extractor.extract(content, file_path)

            # Add assessment info to metadata
            for result in extraction_results:
                result.metadata["complexity_assessment"] = {
                    "score": assessment.complexity_score,
                    "strategy_used": strategy,
                    "recommended_strategy": assessment.recommended_strategy,
                    "document_type": assessment.document_type,
                    "math_density": assessment.math_density
                }

            logger.info(f"Extraction completed: {len(extraction_results)} results")
            return extraction_results, assessment

        except Exception as e:
            logger.error(f"Extraction failed with {strategy}: {str(e)}")
            return [], assessment

    def _assess_complexity(self,
                           content: Union[str, bytes],
                           file_path: str,
                           use_llm_assessment: Optional[bool] = None) -> ComplexityAssessment:
        """Assess PDF complexity using appropriate method"""

        should_use_llm = (
            use_llm_assessment if use_llm_assessment is not None
            else self.enable_llm_assessment and self.llm_assessor is not None
        )

        if should_use_llm:
            logger.info("Using LLM complexity assessment")
            return self.llm_assessor.assess_complexity(content, file_path)
        else:
            logger.info("Using cheap complexity assessment")
            return self.cheap_assessor.assess_complexity(content, file_path)

    def _select_extraction_strategy(self, assessment: ComplexityAssessment) -> str:
        """Select best extraction strategy based on assessment"""

        # Use assessment recommendation as starting point
        recommended = assessment.recommended_strategy

        # Apply business logic
        if assessment.needs_premium_extraction:
            return "anthropic"  # Use premium for complex documents
        elif assessment.can_use_cheap_extraction:
            return "cheap"      # Use cheap for simple documents
        else:
            return "balanced"   # Use balanced for medium complexity

    def _get_extractor_instance(self, strategy: str) -> Optional[PDFExtractorBase]:
        """Get or create extractor instance for reuse"""

        with self.instance_lock:
            # Map strategy to extractor type
            extractor_mapping = {
                "cheap": "cheap",
                "balanced": "balanced",
                "premium": "anthropic",
                "anthropic": "anthropic",
                "openai": "openai"
            }

            extractor_type = extractor_mapping.get(strategy, "balanced")

            if extractor_type not in self.extractors:
                logger.error(f"Unknown extractor type: {extractor_type}")
                return None

            # Check if we have a cached instance
            if extractor_type in self.extractor_instances:
                instance = self.extractor_instances[extractor_type]
                instance.mark_used()
                return instance.extractor

            # Create new instance
            extractor = self.extractors[extractor_type]
            instance = ExtractorInstance(
                extractor=extractor,
                extractor_type=extractor_type,
                created_at=time.time(),
                last_used=time.time(),
                use_count=1
            )

            self.extractor_instances[extractor_type] = instance
            return extractor

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""

        with self.instance_lock:
            stats = {
                "extractors_cached": len(self.extractor_instances),
                "extractor_usage": {}
            }

            for extractor_type, instance in self.extractor_instances.items():
                stats["extractor_usage"][extractor_type] = {
                    "use_count": instance.use_count,
                    "created_at": instance.created_at,
                    "last_used": instance.last_used
                }

        return stats

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)

# ===========================================
# Batch Processing Functions (OpenAI Cookbook Pattern)
# ===========================================

class BatchPDFProcessor:
    """Batch processor for multiple PDFs following OpenAI cookbook patterns"""

    def __init__(self,
                 extractor_type: str = "openai",
                 max_concurrent: int = 5,
                 rate_limit_delay: float = 0.2):
        self.extractor_type = extractor_type
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay

        # Initialize the appropriate extractor
        if extractor_type == "openai":
            self.extractor = OpenAIPDFExtractor()
        elif extractor_type == "anthropic":
            self.extractor = AnthropicPDFExtractor()
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")

    async def process_pdfs_batch(self,
                               pdf_files: List[str],
                               progress_callback: Optional[callable] = None) -> List[Tuple[str, List[DataSourceExtractionResult], Optional[Exception]]]:
        """
        Process multiple PDFs in batch with rate limiting and error handling
        Following OpenAI cookbook pattern for batch processing
        """

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_single_pdf(pdf_path: str) -> Tuple[str, List[DataSourceExtractionResult], Optional[Exception]]:
            async with semaphore:
                try:
                    # Add rate limiting delay
                    await asyncio.sleep(self.rate_limit_delay)

                    # Read PDF content
                    with open(pdf_path, 'rb') as f:
                        content = f.read()

                    # Extract using the specified extractor
                    if hasattr(self.extractor, 'extract_async'):
                        extraction_results = await self.extractor.extract_async(content, pdf_path)
                    else:
                        # Fallback to sync method
                        extraction_results = self.extractor.extract(content, pdf_path)

                    return pdf_path, extraction_results, None

                except Exception as e:
                    logger.error(f"Batch processing failed for {pdf_path}: {str(e)}")
                    return pdf_path, [], e

        # Process all PDFs concurrently
        tasks = [process_single_pdf(pdf_path) for pdf_path in pdf_files]

        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, len(pdf_files), result[0])

        return results

class OpenAIDocumentProcessor:
    """
    Document processor following OpenAI cookbook pattern for RAG preparation
    Handles PDF extraction, chunking, and preparation for vector storage
    """

    def __init__(self,
                 model: str = "gpt-4o",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        self.extractor = OpenAIPDFExtractor(model=model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process_document_for_rag(self,
                                     content: Union[str, bytes],
                                     file_path: str,
                                     include_metadata_enhancement: bool = True) -> Dict[str, Any]:
        """
        Process document for RAG following OpenAI cookbook pattern

        Returns:
            Dict with extracted content, chunks, and metadata for RAG
        """

        # Step 1: Extract content
        extraction_results = await self.extractor.extract_async(content, file_path)

        if not extraction_results:
            return {"error": "Extraction failed", "chunks": [], "metadata": {}}

        result = extraction_results[0]
        markdown_content = result.content

        # Step 2: Enhanced metadata extraction if requested
        enhanced_metadata = {}
        if include_metadata_enhancement:
            enhanced_metadata = await self._extract_document_metadata(content, file_path)

        # Step 3: Chunk the document
        chunks = self._chunk_markdown_content(markdown_content)

        # Step 4: Prepare chunks for RAG
        rag_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **result.metadata,
                **enhanced_metadata,
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_type": self._classify_chunk_type(chunk)
            }

            rag_chunks.append({
                "content": chunk,
                "metadata": chunk_metadata,
                "chunk_id": f"{result.metadata['filename']}_chunk_{i}"
            })

        return {
            "document_metadata": {**result.metadata, **enhanced_metadata},
            "full_content": markdown_content,
            "chunks": rag_chunks,
            "extraction_stats": {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                "content_types": self._analyze_content_types(chunks)
            }
        }

    async def _extract_document_metadata(self, content: Union[str, bytes], file_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata using OpenAI for better RAG performance"""

        try:
            # Get first page image for metadata extraction
            first_page_image = self._get_first_page_image(content)

            if not first_page_image:
                return {}

            metadata_prompt = """Analyze this document and extract key metadata for search and retrieval:

Extract:
1. Document title
2. Authors (if any)
3. Document type (research paper, manual, report, etc.)
4. Main topics/subjects
5. Key concepts or terms
6. Publication info (if visible)
7. Abstract or summary (if present)

Return as JSON:
{
    "title": "Document title",
    "authors": ["Author 1", "Author 2"],
    "document_type": "research_paper",
    "main_topics": ["topic1", "topic2"],
    "key_concepts": ["concept1", "concept2"],
    "publication_info": "Any publication details",
    "abstract": "Abstract or summary if present"
}"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": metadata_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{first_page_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            response = self.extractor.client.chat.completions.create(
                model=self.extractor.model,
                messages=messages,
                max_tokens=1000,
                temperature=0
            )

            if response.choices and response.choices[0].message:
                content_text = response.choices[0].message.content

                # Parse JSON response
                if "```json" in content_text:
                    content_text = content_text.split("```json")[1].split("```")[0]
                elif "```" in content_text:
                    content_text = content_text.split("```")[1].split("```")[0]

                metadata = json.loads(content_text.strip())
                return metadata

        except Exception as e:
            logger.warning(f"Enhanced metadata extraction failed: {str(e)}")

        return {}

    def _get_first_page_image(self, content: Union[str, bytes]) -> Optional[str]:
        """Get first page as base64 image"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                if isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(content.encode('utf-8'))
                tmp_path = tmp_file.name

            try:
                doc = fitz.open(tmp_path)
                if len(doc) == 0:
                    return None

                page = doc.load_page(0)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode()

                doc.close()
                return base64_image

            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Failed to get first page image: {str(e)}")
            return None

    def _chunk_markdown_content(self, content: str) -> List[str]:
        """Chunk markdown content intelligently"""

        # Split by major sections first
        sections = self._split_by_sections(content)

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by markdown sections"""
        import re

        # Split by headers
        header_pattern = r'^(#{1,6}\s.+)

# ===========================================
# Enhanced Usage Examples with OpenAI Cookbook Pattern
# ===========================================

async def extract_pdf_with_openai_cookbook_pattern(pdf_path: str) -> Dict[str, Any]:
    """
    Extract PDF following OpenAI cookbook pattern
    Returns structured data ready for RAG
    """

    processor = OpenAIDocumentProcessor(
        model="gpt-4o",
        chunk_size=1000,
        chunk_overlap=200
    )

    # Read PDF
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # Process for RAG
    result = await processor.process_document_for_rag(
        content,
        pdf_path,
        include_metadata_enhancement=True
    )

    return result

async def batch_process_pdfs_openai_pattern(pdf_files: List[str]) -> List[Dict[str, Any]]:
    """
    Batch process multiple PDFs following OpenAI cookbook pattern
    """

    processor = BatchPDFProcessor(
        extractor_type="openai",
        max_concurrent=3,  # Respect rate limits
        rate_limit_delay=0.5
    )

    def progress_callback(completed: int, total: int, current_file: str):
        print(f"Progress: {completed}/{total} - Processing: {current_file}")

    batch_results = await processor.process_pdfs_batch(
        pdf_files,
        progress_callback=progress_callback
    )

    # Convert to structured format
    structured_results = []
    for pdf_path, extraction_results, error in batch_results:
        if error is None and extraction_results:
            result = {
                "pdf_path": pdf_path,
                "success": True,
                "content": extraction_results[0].content,
                "metadata": extraction_results[0].metadata
            }
        else:
            result = {
                "pdf_path": pdf_path,
                "success": False,
                "error": str(error) if error else "Unknown error",
                "content": "",
                "metadata": {}
            }

        structured_results.append(result)

    return structured_results

def create_adaptive_pipeline(complexity_strategy: str = "llm",
                            max_concurrent: int = 3) -> AdaptivePDFExtractionPipeline:
    """Factory function to create adaptive pipeline"""

    enable_llm = complexity_strategy == "llm"

    return AdaptivePDFExtractionPipeline(
        max_concurrent_cheap=max_concurrent,
        enable_llm_assessment=enable_llm,
        preferred_llm_provider="anthropic"
    )

def extract_pdf_adaptive(pdf_path: str,
                        complexity_strategy: str = "llm") -> Tuple[List[DataSourceExtractionResult], ComplexityAssessment]:
    """Convenience function for single PDF extraction"""

    # Read PDF file
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # Create pipeline
    pipeline = create_adaptive_pipeline(complexity_strategy)

    try:
        # Extract
        results, assessment = pipeline.extract_pdf(content, pdf_path)
        return results, assessment
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    # Example usage with OpenAI cookbook pattern

    pdf_path = "example_document.pdf"

    print("🤖 Adaptive PDF Extraction System with OpenAI Integration")
    print("=" * 60)

    # Example 1: Basic adaptive extraction
    print("\n1. Basic Adaptive Extraction:")
    pipeline = create_adaptive_pipeline(complexity_strategy="llm")

    with open(pdf_path, 'rb') as f:
        content = f.read()

    results, assessment = pipeline.extract_pdf(content, pdf_path)

    print(f"   Complexity Score: {assessment.complexity_score}")
    print(f"   Strategy Used: {assessment.recommended_strategy}")
    print(f"   Results: {len(results)} extractions")

    if results:
        print(f"   Output Format: {results[0].metadata['text_format']}")
        print(f"   Content Preview: {results[0].content[:200]}...")

    pipeline.cleanup()

    # Example 2: Force OpenAI extraction
    print("\n2. Force OpenAI Extraction:")
    results_openai, assessment_openai = pipeline.extract_pdf(
        content,
        pdf_path,
        force_strategy="openai"
    )

    if results_openai:
        print(f"   OpenAI Model: {results_openai[0].metadata.get('model', 'unknown')}")
        print(f"   Success Rate: {results_openai[0].metadata.get('extraction_stats', {}).get('success_rate', 0):.2%}")

    # Example 3: OpenAI Cookbook Pattern (Async)
    print("\n3. OpenAI Cookbook Pattern (RAG Preparation):")

    async def demo_openai_cookbook():
        # Process single PDF for RAG
        rag_result = await extract_pdf_with_openai_cookbook_pattern(pdf_path)

        print(f"   Document Title: {rag_result['document_metadata'].get('title', 'Unknown')}")
        print(f"   Total Chunks: {rag_result['extraction_stats']['total_chunks']}")
        print(f"   Content Types: {rag_result['extraction_stats']['content_types']}")

        # Show chunk examples
        chunks = rag_result['chunks'][:3]  # First 3 chunks
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1} ({chunk['metadata']['chunk_type']}): {chunk['content'][:100]}...")

    # Run async example
    # asyncio.run(demo_openai_cookbook())  # Uncomment to run

    # Example 4: Batch Processing
    print("\n4. Batch Processing Example:")

    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]  # Your PDF files

    async def demo_batch_processing():
        batch_results = await batch_process_pdfs_openai_pattern(pdf_files)

        successful = sum(1 for r in batch_results if r['success'])
        print(f"   Processed: {len(batch_results)} files")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(batch_results) - successful}")

        for result in batch_results[:2]:  # Show first 2 results
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['pdf_path']}")

    # asyncio.run(demo_batch_processing())  # Uncomment to run

    # Example 5: Cost Estimation
    print("\n5. Extraction Cost Estimation:")

    for strategy in ["cheap", "balanced", "anthropic", "openai"]:
        extractor = pipeline._get_extractor_instance(strategy)
        if extractor:
            cost = extractor.get_processing_cost_estimate(assessment)
            print(f"   {strategy.capitalize()}: ${cost/100:.2f} per document")

    print("\n✅ Examples complete!")
    print("\n💡 Quick Start:")
    print("   # Simple extraction:")
    print("   results, assessment = extract_pdf_adaptive('your_file.pdf')")
    print()
    print("   # OpenAI RAG preparation:")
    print("   rag_data = await extract_pdf_with_openai_cookbook_pattern('your_file.pdf')")
    print()
    print("   # Batch processing:")
    print("   results = await batch_process_pdfs_openai_pattern(['file1.pdf', 'file2.pdf'])")

    print(f"\n🎯 For your mathematical documents (complexity {assessment.complexity_score:.1f}):")
    if assessment.math_density == "heavy":
        print("   ✅ Recommended: OpenAI or Anthropic extraction for perfect formula handling")
        print("   📐 Both models excel at LaTeX math notation conversion")
    else:
        print("   ✅ Recommended: Balanced extraction for good speed/quality trade-off")
        lines = content.split('\n')

        sections = []
        current_section = []

        for line in lines:
            if re.match(header_pattern, line) and current_section:
                # Start new section
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        return sections

    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections into smaller chunks"""

        words = section.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > self.chunk_size and current_chunk:
                # Create chunk with overlap
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Create overlap for next chunk
                overlap_words = int(len(current_chunk) * (self.chunk_overlap / self.chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)

            current_chunk.append(word)
            current_length += word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _classify_chunk_type(self, chunk: str) -> str:
        """Classify chunk type for better retrieval"""

        chunk_lower = chunk.lower()

        if any(marker in chunk for marker in ['#', '##', '###']):
            return "header_section"
        elif '|' in chunk and '---' in chunk:
            return "table"
        elif any(marker in chunk for marker in ['$', '

def create_adaptive_pipeline(complexity_strategy: str = "llm",
                            max_concurrent: int = 3) -> AdaptivePDFExtractionPipeline:
    """Factory function to create adaptive pipeline"""

    enable_llm = complexity_strategy == "llm"

    return AdaptivePDFExtractionPipeline(
        max_concurrent_cheap=max_concurrent,
        enable_llm_assessment=enable_llm,
        preferred_llm_provider="anthropic"
    )

def extract_pdf_adaptive(pdf_path: str,
                        complexity_strategy: str = "llm") -> Tuple[List[DataSourceExtractionResult], ComplexityAssessment]:
    """Convenience function for single PDF extraction"""

    # Read PDF file
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # Create pipeline
    pipeline = create_adaptive_pipeline(complexity_strategy)

    try:
        # Extract
        results, assessment = pipeline.extract_pdf(content, pdf_path)
        return results, assessment
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    # Example usage

    pdf_path = "example_document.pdf"

    # Create adaptive pipeline
    pipeline = create_adaptive_pipeline(complexity_strategy="llm")

    # Extract PDF
    with open(pdf_path, 'rb') as f:
        content = f.read()

    results, assessment = pipeline.extract_pdf(content, pdf_path)

    print(f"Complexity Score: {assessment.complexity_score}")
    print(f"Strategy Used: {assessment.recommended_strategy}")
    print(f"Results: {len(results)} extractions")

    if results:
        print(f"Output Format: {results[0].metadata['text_format']}")
        print(f"Content Preview: {results[0].content[:200]}...")

    # Cleanup
    pipeline.cleanup()]):
            return "mathematical"
        elif any(word in chunk_lower for word in ['figure', 'fig.', 'table', 'chart']):
            return "figure_reference"
        elif any(word in chunk_lower for word in ['abstract', 'summary', 'conclusion']):
            return "summary"
        elif any(word in chunk_lower for word in ['reference', 'bibliography', 'citation']):
            return "reference"
        else:
            return "content"

    def _analyze_content_types(self, chunks: List[str]) -> Dict[str, int]:
        """Analyze distribution of content types"""

        type_counts = {}
        for chunk in chunks:
            chunk_type = self._classify_chunk_type(chunk)
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        return type_counts



def create_adaptive_pipeline(complexity_strategy: str = "llm",
                            max_concurrent: int = 3) -> AdaptivePDFExtractionPipeline:
    """Factory function to create adaptive pipeline"""

    enable_llm = complexity_strategy == "llm"

    return AdaptivePDFExtractionPipeline(
        max_concurrent_cheap=max_concurrent,
        enable_llm_assessment=enable_llm,
        preferred_llm_provider="anthropic"
    )

def extract_pdf_adaptive(pdf_path: str,
                         complexity_strategy: str = "llm") -> Tuple[List[DataSourceExtractionResult], ComplexityAssessment]:
    """Convenience function for single PDF extraction"""

    # Read PDF file
    with open(pdf_path, 'rb') as f:
        content = f.read()

    # Create pipeline
    pipeline = create_adaptive_pipeline(complexity_strategy)

    try:
        # Extract
        results, assessment = pipeline.extract_pdf(content, pdf_path)
        return results, assessment
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    # Example usage

    pdf_path = "example_document.pdf"

    # Create adaptive pipeline
    pipeline = create_adaptive_pipeline(complexity_strategy="llm")

    # Extract PDF
    with open(pdf_path, 'rb') as f:
        content = f.read()

    results, assessment = pipeline.extract_pdf(content, pdf_path)

    print(f"Complexity Score: {assessment.complexity_score}")
    print(f"Strategy Used: {assessment.recommended_strategy}")
    print(f"Results: {len(results)} extractions")

    if results:
        print(f"Output Format: {results[0].metadata['text_format']}")
        print(f"Content Preview: {results[0].content[:200]}...")

    # Cleanup
    pipeline.cleanup()