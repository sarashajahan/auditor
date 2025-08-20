import streamlit as st
import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
import tempfile
from datetime import datetime
import uuid
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import warnings
warnings.filterwarnings("ignore")
import re
import asyncio

# Import the Financial Logic Agent to validate after extraction
try:
    import financial_logic_agent as logic
except Exception as _e:
    logic = None



# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Invoice Data Extractor",
    page_icon="🧾",
    layout="wide"
)

# Initialize session state
if 'processed_result' not in st.session_state:
    st.session_state.processed_result = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def encode_image(file_path):
    """Encode image to base64"""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def try_simplified_extraction(client, model, image_path, is_preprocessed=False):
    """Simplified extraction for large documents that cause JSON truncation"""
    base64_image = encode_image(image_path)
    
    # Simplified prompt focusing on key data
    prompt = f"""
STRICT INSTRUCTION: Only output valid JSON, no markdown or explanations.

This is a large invoice. Extract key information and SUMMARIZE line items instead of listing all individually:

{{
  "quality_assessment": {{
    "quality_too_poor": false,
    "quality_issues": ["Large document - simplified extraction"],
    "readability_score": "high",
    "can_extract_data": true,
    "preprocessing_recommended": false
  }},
  "invoice_header": {{
    "vendor_name": "",
    "vendor_address": "",
    "invoice_number": "",
    "invoice_date": "",
    "total_amount": ""
  }},
  "customer_details": {{
    "customer_name": "",
    "customer_address": ""
  }},
  "line_items_summary": {{
    "total_line_items": 0,
    "sample_items": ["List first 3-5 items only"],
    "total_value": "",
    "note": "Large document - showing sample items only"
  }},
  "financial_summary": {{
    "total_amount": "",
    "currency": ""
  }},
  "terms_and_conditions": {{
    "payment_terms": "",
    "other_conditions": []
  }}
}}

Extract main invoice details and count/sample the line items instead of listing all items individually.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,  # Lower token limit for simplified response
            temperature=0.1
        )
        
        content = response.choices.message.content
        cleaned_content = clean_json_response(content)
        result = json.loads(cleaned_content)
        
        # Convert simplified result to standard format
        return convert_simplified_to_standard_format(result)
        
    except Exception as e:
        st.error(f"❌ Simplified extraction also failed: {e}")
        return None

def convert_simplified_to_standard_format(simplified_result):
    """Convert simplified extraction result to standard format"""
    standard_result = {
        "quality_assessment": simplified_result.get("quality_assessment", {}),
        "invoice_header": simplified_result.get("invoice_header", {}),
        "customer_details": simplified_result.get("customer_details", {}),
        "line_items": [],  # Empty for large documents
        "financial_summary": simplified_result.get("financial_summary", {}),
        "payment_details": {},
        "terms_and_conditions": simplified_result.get("terms_and_conditions", {}),
        "additional_info": {
            "notes": f"Large document with {simplified_result.get('line_items_summary', {}).get('total_line_items', 'many')} items - simplified extraction used"
        },
        "detection_metadata": {
            "large_document": True,
            "extraction_method": "simplified"
        }
    }
    
    # Add sample items if available
    sample_items = simplified_result.get("line_items_summary", {}).get("sample_items", [])
    for i, item_desc in enumerate(sample_items[:5], 1):
        standard_result["line_items"].append({
            "item_number": f"Sample {i}",
            "description": item_desc,
            "quantity": "N/A",
            "unit_price": "N/A",
            "total_price": "N/A",
            "note": "Sample item from large document"
        })
    
    return standard_result

def enhance_currency_detection(result):
    """Enhance currency detection and set defaults"""
    if not result:
        return result
    
    # Currency patterns to look for
    currency_patterns = {
        '₹': 'INR', 'Rs.': 'INR', 'Rs': 'INR', 'INR': 'INR', 'Rupees': 'INR', 'rupees': 'INR',
        '$': 'USD', 'USD': 'USD', 'Dollars': 'USD',
        '€': 'EUR', 'EUR': 'EUR', 'Euros': 'EUR',
        '£': 'GBP', 'GBP': 'GBP', 'Pounds': 'GBP'
    }
    
    detected_currency = None
    
    # 🔧 PRIORITY 1: Look for symbols in amounts FIRST (most reliable)
    amounts_to_check = []
    
    if result.get('financial_summary'):
        amounts_to_check.extend([
            result['financial_summary'].get('total_amount', ''),
            result['financial_summary'].get('subtotal', ''),
            result['financial_summary'].get('amount_in_words', '')
        ])
    
    if result.get('line_items'):
        for item in result['line_items']:
            amounts_to_check.extend([item.get('unit_price', ''), item.get('total_price', '')])
    
    # Look for currency patterns in amounts
    for amount in amounts_to_check:
        if amount and amount != 'N/A':
            for pattern, currency_code in currency_patterns.items():
                if pattern in str(amount):
                    detected_currency = currency_code
                    print(f"DEBUG: Found '{pattern}' in amount '{amount}' -> Currency: {currency_code}")
                    break
            if detected_currency:
                break
    
    # 🔧 PRIORITY 2: Check explicit currency fields (fallback)
    if not detected_currency:
        header_currency = result.get('invoice_header', {}).get('currency', 'N/A')
        financial_currency = result.get('financial_summary', {}).get('currency', 'N/A')
        
        if header_currency and header_currency != 'N/A':
            detected_currency = header_currency
        elif financial_currency and financial_currency != 'N/A':
            detected_currency = financial_currency
    
    # 🔧 PRIORITY 3: Check for Indian tax indicators (secondary fallback)
    if not detected_currency or detected_currency == 'N/A':
        has_indian_tax = False
        
        # Check for Indian indicators
        if result.get('financial_summary'):
            financial = result['financial_summary']
            if (financial.get('cgst', 'N/A') != 'N/A' or 
                financial.get('sgst', 'N/A') != 'N/A' or 
                financial.get('igst', 'N/A') != 'N/A'):
                has_indian_tax = True
        
        if result.get('invoice_header'):
            header = result['invoice_header']
            if (header.get('vendor_gst_number', 'N/A') != 'N/A' or 
                header.get('vendor_pan', 'N/A') != 'N/A'):
                has_indian_tax = True
        
        # Only default to INR if Indian indicators found
        if has_indian_tax:
            detected_currency = 'INR'
        else:
            # If no Indian indicators and no symbols found, this is suspicious
            print("DEBUG: No currency symbols or Indian indicators found - defaulting to INR")
            detected_currency = 'INR'
    
    # Normalize currency
    currency_map = {
        'INR': 'Rupees (₹)', 'Rupees': 'Rupees (₹)', 'rupees': 'Rupees (₹)', 
        'Rs': 'Rupees (₹)', 'Rs.': 'Rupees (₹)',
        'USD': 'US Dollars ($)', 'Dollars': 'US Dollars ($)',
        'EUR': 'Euros (€)', 'GBP': 'British Pounds (£)'
    }
    
    final_currency = currency_map.get(detected_currency, 'Rupees (₹)')
    
    # Update result with detected currency
    if 'invoice_header' in result:
        result['invoice_header']['currency'] = final_currency
    if 'financial_summary' in result:
        result['financial_summary']['currency'] = final_currency
    
    print(f"DEBUG: Final currency set to: {final_currency}")
    
    return result

def clean_json_response(content):
    """Clean markdown formatting and fix truncated JSON"""
    content = content.strip()
    
    # Remove code block markers
    if content.startswith('```json'):
        content = content[7:]  # Remove ```
    elif content.startswith('```'):
        content = content[3:]   # Remove ```
    
    if content.endswith('```'):
        content = content[:-3]  # Remove trailing ```
    
    content = content.strip()
    
    # 🔧 NEW: Fix common JSON truncation issues for large documents
    if not content.endswith('}'):
        # Remove trailing commas and incomplete entries
        content = content.rstrip(',\n\r\t ')
        
        # Try to close incomplete strings
        if content.count('"') % 2 == 1:
            content += '"'
        
        # Close incomplete objects and arrays
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        if open_brackets > 0:
            content += ']' * open_brackets
        if open_braces > 0:
            content += '}' * open_braces
    
    return content

def create_results_directory():
    """Create results directory if it doesn't exist"""
    results_dir = "resultjson"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_result_to_file(result, filename, results_dir="resultjson"):
    """Save individual result to a separate JSON file"""
    if not result:
        return None
    
    # Create timestamp and unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # Add metadata to result
    enhanced_result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unique_id": unique_id,
            "filename": filename,
            "file_id": f"{timestamp}_{unique_id}"
        },
        "extraction_data": result
    }
    
    # Create individual file
    individual_filename = f"invoice_{timestamp}_{unique_id}.json"
    individual_path = os.path.join(results_dir, individual_filename)
    
    with open(individual_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_result, f, indent=2, ensure_ascii=False)
    
    return individual_path, enhanced_result

def append_to_master_results(result, filename, results_dir="resultjson"):
    """Append result to master results file"""
    master_file = os.path.join(results_dir, "all_results.json")
    
    # Create timestamp and unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    # Prepare new entry
    new_entry = {
        "id": f"{timestamp}_{unique_id}",
        "timestamp": datetime.now().isoformat(),
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_filename": filename,
        "extraction_data": result
    }
    
    # Load existing results or create new list
    if os.path.exists(master_file):
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            if not isinstance(all_results, list):
                all_results = [all_results]  # Convert single object to list
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = []
    else:
        all_results = []
    
    # Append new result
    all_results.append(new_entry)
    
    # Save back to master file
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    return master_file, len(all_results)

def get_results_summary(results_dir="resultjson"):
    """Get summary of all stored results"""
    master_file = os.path.join(results_dir, "all_results.json")
    
    if not os.path.exists(master_file):
        return {"total_files": 0, "latest_processing": None}
    
    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        if not isinstance(all_results, list):
            all_results = [all_results]
        
        total_files = len(all_results)
        latest_processing = all_results[-1]["processing_date"] if all_results else None
        
        return {
            "total_files": total_files,
            "latest_processing": latest_processing,
            "results_directory": results_dir
        }
    except:
        return {"total_files": 0, "latest_processing": None}

# NEW PDF PROCESSING FUNCTIONS
def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to images using PyMuPDF (no poppler needed)"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        output_paths = []
        
        pdf_file = Path(pdf_path)
        temp_dir = pdf_file.parent
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi/72, dpi/72)  # Convert DPI to scale factor
            pix = page.get_pixmap(matrix=mat)
            
            image_path = temp_dir / f"{pdf_file.stem}_page_{page_num+1}.jpg"
            pix.save(str(image_path))
            output_paths.append(str(image_path))
        
        doc.close()
        return output_paths
    except Exception as e:
        st.error(f"❌ PDF conversion failed: {e}")
        return []

def get_pdf_page_count(pdf_path):
    """Get number of pages in PDF"""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        st.error(f"❌ Could not read PDF: {e}")
        return 0

def combine_pdf_page_results(page_results, pdf_filename):
    """Combine results from all PDF pages into a structured format"""
    
    successful_pages = [r for r in page_results if 'error' not in r]
    failed_pages = [r for r in page_results if 'error' in r]
    
    # Find the page with the most complete invoice header (usually page 1)
    main_page = None
    for page in successful_pages:
        if page.get('invoice_header', {}).get('vendor_name', 'N/A') != 'N/A':
            main_page = page
            break
    
    if not main_page and successful_pages:
        main_page = successful_pages[0]
    
    # Combine all line items from all pages
    all_line_items = []
    for page in successful_pages:
        page_line_items = page.get('line_items', [])
        for item in page_line_items:
            if item and any(v != 'N/A' and v != '' for v in item.values()):
                item['source_page'] = page.get('page_info', {}).get('page_number', 'Unknown')
                all_line_items.append(item)
    
    # 🔧 IMPROVED: Smart combination of terms and conditions from ALL pages
    combined_terms = {
        "payment_terms": "",
        "delivery_terms": "",
        "warranty_terms": "",
        "return_policy": "",
        "late_payment_charges": "",
        "jurisdiction": "",
        "other_conditions": []
    }
    
    # Collect the BEST (non-N/A) value for each term from all pages
    for page in successful_pages:
        page_terms = page.get('terms_and_conditions', {})
        page_num = page.get('page_info', {}).get('page_number', 'Unknown')
        
        for key, value in page_terms.items():
            if value and value != 'N/A' and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):
                if key == 'other_conditions' and isinstance(value, list):
                    # Add list items
                    for condition in value:
                        if condition and condition != 'N/A':
                            combined_terms[key].append(condition)
                else:
                    # For single values, take the first meaningful one found
                    if not combined_terms[key] or combined_terms[key] == 'N/A':
                        combined_terms[key] = value
                    elif combined_terms[key] != value:
                        # If different values exist, combine them
                        combined_terms[key] = f"{combined_terms[key]} | {value}"
    
    # 🔧 IMPROVED: Smart combination of payment details from ALL pages
    combined_payment = {}
    
    # Start with main page payment details
    if main_page:
        combined_payment = main_page.get('payment_details', {}).copy()
    
    # Enhance with data from other pages
    for page in successful_pages:
        page_payment = page.get('payment_details', {})
        
        for key, value in page_payment.items():
            if value and value != 'N/A' and ((isinstance(value, str) and value.strip()) or (isinstance(value, list) and len(value) > 0)):

                if not combined_payment.get(key) or combined_payment.get(key) == 'N/A':
                    combined_payment[key] = value
                elif combined_payment[key] != value:
                    # If different values exist, combine them
                    combined_payment[key] = f"{combined_payment[key]} | {value}"
    
    # Create combined result
    combined_result = {
        'pdf_info': {
            'source_pdf': pdf_filename,
            'total_pages': len(page_results),
            'successful_pages': len(successful_pages),
            'failed_pages': len(failed_pages),
            'processing_date': datetime.now().isoformat()
        },
        'combined_data': {
            'invoice_header': main_page.get('invoice_header', {}) if main_page else {},
            'customer_details': main_page.get('customer_details', {}) if main_page else {},
            'line_items': all_line_items,
            'financial_summary': main_page.get('financial_summary', {}) if main_page else {},
            'payment_details': combined_payment,  # 🔧 Smart combined payment details
            'terms_and_conditions': combined_terms,  # 🔧 Smart combined terms
            'additional_info': main_page.get('additional_info', {}) if main_page else {}
        },
        'page_by_page_results': page_results,
        'processing_summary': {
            'total_line_items_found': len(all_line_items),
            'pages_with_line_items': len([p for p in successful_pages if p.get('line_items')]),
            'overall_quality': 'high' if len(successful_pages) == len(page_results) else 'medium' if successful_pages else 'low'
        }
    }
    
    return combined_result


def process_multi_page_pdf(pdf_path, uploaded_filename):
    """Process multi-page PDF with smart retry for each page"""
    # Get page count
    page_count = get_pdf_page_count(pdf_path)
    
    if page_count == 0:
        st.error("❌ Invalid PDF or no pages found")
        return None
    
    st.info(f"📄 **PDF detected with {page_count} pages**")
    
    # Convert PDF to images
    with st.spinner('🔄 Converting PDF pages to images...'):
        image_paths = pdf_to_images(pdf_path)
    
    if not image_paths:
        st.error("❌ Failed to convert PDF pages")
        return None
    
    # ADD DEBUG INFO
    st.info(f"🖼️ **Successfully converted {len(image_paths)} pages to images**")
    
    # Process each page
    all_page_results = []
    
    for i, image_path in enumerate(image_paths, 1):
        st.info(f"🔄 **Processing Page {i}/{page_count}...**")
        
        # Process with smart retry
        page_result = process_invoice_with_retry(image_path)
        
        if page_result:
            # ADD DEBUG: Show what was extracted from this page
            vendor = page_result.get('invoice_header', {}).get('vendor_name', 'N/A')
            line_items_count = len(page_result.get('line_items', []))
            st.success(f"✅ Page {i} processed - Vendor: {vendor}, Line Items: {line_items_count}")
            
            # Add page metadata
            page_result['page_info'] = {
                'page_number': i,
                'total_pages': page_count,
                'source_pdf': uploaded_filename,
                'page_image': Path(image_path).name
            }
            all_page_results.append(page_result)
        else:
            st.warning(f"⚠️ Page {i} processing failed")
            # Add failed page placeholder
            all_page_results.append({
                'page_info': {
                    'page_number': i,
                    'total_pages': page_count,
                    'source_pdf': uploaded_filename,
                    'processing_failed': True
                },
                'error': 'Page processing failed'
            })
    
    # ADD DEBUG: Show combination results
    st.info(f"📊 **Total processed results: {len(all_page_results)}**")
    
    # Clean up temporary image files
    for image_path in image_paths:
        try:
            os.unlink(image_path)
        except:
            pass
    
    # Combine results
    combined_result = combine_pdf_page_results(all_page_results, uploaded_filename)
    
    # ADD DEBUG: Show combined line items count
    combined_line_items = len(combined_result.get('combined_data', {}).get('line_items', []))
    st.info(f"🔗 **Combined line items: {combined_line_items}**")
    
    return combined_result

def display_pdf_results(result):
    """Display PDF results in tabbed format with source page info"""
    if not result:
        return
    
    pdf_info = result.get('pdf_info', {})
    combined_data = result.get('combined_data', {})
    
    # Show PDF processing summary
    st.subheader("📄 PDF Processing Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Pages", pdf_info.get('total_pages', 0))
    with col2:
        st.metric("✅ Successful", pdf_info.get('successful_pages', 0))
    with col3:
        st.metric("❌ Failed", pdf_info.get('failed_pages', 0))
    
    st.divider()
    
    # Display combined data with custom line items handling
    st.subheader("📋 Combined Invoice Data (All Pages)")
    
    if combined_data:
        # Create the same tabs as single images
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Invoice Header", 
            "👤 Customer Details", 
            "📦 Line Items", 
            "💰 Financial Summary",
            "💳 Payment Details",
            "📜 Terms & Conditions",
            "ℹ️ Additional Info"
        ])
        
        with tab1:
            st.subheader("Invoice Header Information")
            if "invoice_header" in combined_data:
                for key, value in combined_data["invoice_header"].items():
                    if value and value != "N/A":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab2:
            st.subheader("Customer Details")
            if "customer_details" in combined_data:
                for key, value in combined_data["customer_details"].items():
                    if value and value != "N/A":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab3:
            st.subheader("Line Items (All Pages)")
            if "line_items" in combined_data and combined_data["line_items"]:
                for i, item in enumerate(combined_data["line_items"], 1):
                    # Show source page info for line items
                    source_page = item.get('source_page', 'Unknown')
                    st.write(f"**Item {i}** *(from page {source_page})*:")
                    
                    for key, value in item.items():
                        if key != 'source_page' and value and value != "N/A":
                            st.write(f"  -  {key.replace('_', ' ').title()}: {value}")
                    st.divider()
        
        with tab4:
            st.subheader("Financial Summary")
            if "financial_summary" in combined_data:
                col1, col2 = st.columns(2)
                items = list(combined_data["financial_summary"].items())
                mid = len(items) // 2
                
                with col1:
                    for key, value in items[:mid]:
                        if value and value != "N/A":
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with col2:
                    for key, value in items[mid:]:
                        if value and value != "N/A":
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab5:
            st.subheader("Payment Details")
            if "payment_details" in combined_data:
                for key, value in combined_data["payment_details"].items():
                    if value and value != "N/A":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab6:
            st.subheader("Terms & Conditions")
            if "terms_and_conditions" in combined_data:
                for key, value in combined_data["terms_and_conditions"].items():
                    if value and value != "N/A":
                        if isinstance(value, list):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                st.write(f"  -  {item}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with tab7:
            st.subheader("Additional Information & Metadata")
            if "additional_info" in combined_data:
                st.write("**Additional Info:**")
                for key, value in combined_data["additional_info"].items():
                    if value and value != "N/A":
                        st.write(f"  -  {key.replace('_', ' ').title()}: {value}")
    else:
        st.warning("⚠️ No combined data available from PDF processing")

# YOUR EXISTING FUNCTIONS (unchanged)
import cv2
import numpy as np
from pathlib import Path

def preprocess_image_enhanced(image_path):
    """
    Enhanced preprocessing using OpenCV: grayscale, noise removal, adaptive binarization, morphological closing, and resize.
    Returns the processed image path (or original if failure).
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or invalid: {image_path}")
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Noise reduction
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Adaptive thresholding for binarization
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # 4. Morphological closing
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Resize if too large (like your Pillow logic)
        max_size = 2048
        height, width = morph.shape
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            morph = cv2.resize(morph, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        # Save processed image
        file_path = Path(image_path)
        processed_path = str(file_path.parent / f"{file_path.stem}_enhanced.jpg")
        cv2.imwrite(processed_path, morph)

        return processed_path
    
    except Exception as e:
        print(f"❌ OpenCV preprocessing failed: {e}")
        return image_path
import os

def download_preprocessed_image(preprocessed_image_path):
    if not os.path.exists(preprocessed_image_path):
        st.error("Preprocessed image not found.")
        return
    with open(preprocessed_image_path, "rb") as file:
        st.download_button(
            label="📥 Download Preprocessed Image",
            data=file,
            file_name=os.path.basename(preprocessed_image_path),
            mime="image/jpeg"
        )

# ===============================
# Bridge: Convert extraction to Financial Logic Agent input
# ===============================

def _to_currency_code(currency_text: str) -> str:
    if not currency_text:
        return "USD"
    text = str(currency_text)
    if any(sym in text for sym in ["₹", "INR", "Rupee", "rupee", "Rs", "Rs."]):
        return "INR"
    if any(sym in text for sym in ["$", "USD", "Dollar"]):
        return "USD"
    if any(sym in text for sym in ["€", "EUR", "Euro"]):
        return "EUR"
    if any(sym in text for sym in ["£", "GBP", "Pound"]):
        return "GBP"
    return "USD"

def _parse_amount(val) -> float:
    if val is None or val == "N/A":
        return 0.0
    s = str(val)
    s = re.sub(r"[₹$€£,\s]", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return 0.0
    try:
        return float(m.group(0))
    except Exception:
        return 0.0

def _parse_percentage(val) -> float:
    if not val or val == "N/A":
        return 0.0
    s = str(val).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    try:
        num = float(m.group(0)) if m else 0.0
        return num / 100.0 if num > 1.0 else num
    except Exception:
        return 0.0

def _safe_iso_date(date_text: str) -> str:
    from datetime import datetime
    if not date_text or date_text == "N/A":
        return datetime.utcnow().isoformat()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(date_text.strip(), fmt).isoformat()
        except Exception:
            continue
    return datetime.utcnow().isoformat()

def build_financial_agent_input(extraction_result: dict) -> dict:
    from datetime import datetime
    import uuid

    is_pdf = isinstance(extraction_result, dict) and "pdf_info" in extraction_result
    data = extraction_result.get("combined_data", {}) if is_pdf else extraction_result

    header = data.get("invoice_header", {}) or {}
    financial = data.get("financial_summary", {}) or {}
    items = data.get("line_items", []) or []

    vendor = header.get("vendor_name", "Unknown Vendor") or "Unknown Vendor"
    invoice_date = header.get("invoice_date", "")
    currency_code = _to_currency_code(header.get("currency") or financial.get("currency") or "")

    if currency_code in ("EUR", "GBP"):
        jurisdiction = "EU"
    else:
        jurisdiction = "US"

    line_items = []
    if items:
        for it in items:
            desc = (it or {}).get("description") or "Item"
            qty_raw = (it or {}).get("quantity")
            qty = _parse_amount(qty_raw) if qty_raw not in (None, "N/A", "") else 1.0
            unit_price = _parse_amount((it or {}).get("unit_price"))
            total_price = _parse_amount((it or {}).get("total_price"))
            if unit_price == 0.0 and qty:
                unit_price = total_price / qty if total_price else 0.0
            tax_rate = _parse_percentage((it or {}).get("tax_rate"))
            line_items.append({
                "description": desc,
                "quantity": qty,
                "unit_price": unit_price,
                "tax_rate": tax_rate,
                "category": "unknown",
                "currency": currency_code,
                "jurisdiction": jurisdiction,
                "vendor": vendor
            })
    else:
        total_amount = _parse_amount(financial.get("total_amount"))
        tax_total = _parse_amount(financial.get("total_tax_amount"))
        est_tax_rate = (tax_total / total_amount) if total_amount > 0 else 0.0
        line_items.append({
            "description": f"Invoice from {vendor}",
            "quantity": 1.0,
            "unit_price": total_amount,
            "tax_rate": est_tax_rate,
            "category": "unknown",
            "currency": currency_code,
            "jurisdiction": jurisdiction,
            "vendor": vendor
        })

    claimed_total = _parse_amount(financial.get("total_amount"))

    payload = {
        "report_id": f"INV-{uuid.uuid4().hex[:8]}",
        "employee": {"user_id": "EMP-AUTO", "role": "employee"},
        "submission_date": datetime.utcnow().isoformat(),
        "expense_date": _safe_iso_date(invoice_date),
        "line_items": line_items,
        "currency": currency_code,
        "jurisdiction": jurisdiction,
        "claimed_total": claimed_total,
        # Strict minimal invoice fields block for the logic agent
        "invoice_metadata": {
            "vendor_name": vendor,
            "vendor_address": header.get("vendor_address"),
            "invoice_number": header.get("invoice_number"),
            "invoice_date": invoice_date,
            "currency": currency_code,
            "subtotal": _parse_amount(financial.get("subtotal")),
            "total_tax_amount": _parse_amount(financial.get("total_tax_amount")),
            "total_amount": _parse_amount(financial.get("total_amount"))
        }
    }

    return payload

def write_financial_agent_input_file(extraction_result: dict, output_path: str = "input.json") -> str:
    try:
        payload = build_financial_agent_input(extraction_result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path
    except Exception as e:
        st.error(f"❌ Failed to create Financial Logic Agent input: {e}")
        return ""

def is_quality_too_poor(result):
    """
    Check if GPT-4o says image quality is too poor,
    OR assigns a 'medium' or 'low' readability score.
    """
    if not result:
        return False

    quality_assessment = result.get("quality_assessment", {})
    quality_too_poor = quality_assessment.get("quality_too_poor", False)
    readability_score = quality_assessment.get("readability_score", "high").lower()

    # Consider both 'quality_too_poor' TRUE or readability_score medium/low as needing preprocessing
    return quality_too_poor or readability_score in ["medium", "low"]

def is_quality_unacceptable(result):
    """
    Returns True if model says image quality is so poor that extraction shouldn't proceed or save.
    Check for quality_too_poor == True and can_extract_data == False, or readability_score == 'low' or similar indicators.
    """
    if not result:
        return True

    quality = result.get("quality_assessment", {})
    # If GPT-4o says cannot extract data, or quality is explicitly too bad
    cannot_extract = not quality.get("can_extract_data", True)
    quality_too_poor = quality.get("quality_too_poor", False)
    readability_score = quality.get("readability_score", "high").lower()
    return cannot_extract or quality_too_poor or readability_score == "low"

def try_process_image(client, model, image_path, is_preprocessed=False):
    """Single attempt to process image with GPT-4o with enhanced error handling"""
    base64_image = encode_image(image_path)
    
    # Enhanced prompt that explicitly asks about quality issues
    prompt = f"""
STRICT INSTRUCTION: Only output valid JSON, no markdown or explanations.

{'RETRY ATTEMPT - This is a preprocessed image.' if is_preprocessed else 'FIRST ATTEMPT - This is the original image.'}

First, assess if you can reliably extract data from this image:
- If the image is too blurry, dark, or distorted to read text clearly, set "quality_too_poor" to true
- If you can read most text despite some quality issues, set "quality_too_poor" to false
- If this image is not a valid invoice, set "can_extract_data" to false and add "not invoice" to "quality_issues".


Extract ALL available information from this invoice and return as JSON:
{{
  "quality_assessment": {{
    "quality_too_poor": true/false,
    "quality_issues": ["list any specific quality problems"],
    "readability_score": "high/medium/low",
    "can_extract_data": true/false,
    "preprocessing_recommended": true/false
  }},
  "invoice_header": {{
    "vendor_name": "",
    "vendor_address": "",
    "vendor_phone": "",
    "vendor_email": "",
    "vendor_website": "",
    "vendor_gst_number": "",
    "vendor_pan": "",
    "invoice_number": "",
    "invoice_date": "",
    "due_date": "",
    "purchase_order_number": "",
    "reference_number": "",
    "currency": ""
  }},
  "customer_details": {{
    "customer_name": "",
    "customer_address": "",
    "customer_phone": "",
    "customer_email": "",
    "customer_gst_number": "",
    "customer_pan": "",
    "billing_address": "",
    "shipping_address": "",
    "customer_contact_person": ""
  }},
  "line_items": [
    {{
      "item_number": "",
      "description": "",
      "hsn_sac_code": "",
      "quantity": "",
      "unit": "",
      "unit_price": "",
      "discount": "",
      "tax_rate": "",
      "tax_amount": "",
      "total_price": ""
    }}
  ],
  "financial_summary": {{
    "subtotal": "",
    "total_discount": "",
    "taxable_amount": "",
    "cgst": "",
    "sgst": "",
    "igst": "",
    "cess": "",
    "other_charges": "",
    "shipping_charges": "",
    "total_tax_amount": "",
    "round_off": "",
    "total_amount": "",
    "amount_in_words": ""
  }},
  "payment_details": {{
    "payment_terms": "",
    "payment_method": "",
    "bank_name": "",
    "account_number": "",
    "ifsc_code": "",
    "branch": "",
    "upi_id": "",
    "advance_paid": "",
    "balance_due": ""
  }},
  "terms_and_conditions": {{
    "payment_terms": "",
    "delivery_terms": "",
    "warranty_terms": "",
    "return_policy": "",
    "late_payment_charges": "",
    "jurisdiction": "",
    "other_conditions": []
  }},
  "additional_info": {{
    "notes": "",
    "special_instructions": "",
    "delivery_date": "",
    "place_of_supply": "",
    "reverse_charge": "",
    "document_type": "",
    "series": "",
    "authorised_signatory": "",
    "stamp_or_seal": "",
    "qr_code_present": ""
  }},
  "detection_metadata": {{
    "tables_detected": true/false,
    "handwriting_detected": true/false,
    "logo_detected": true/false,
    "stamp_detected": true/false,
    "signature_detected": true/false,
    "barcode_qr_detected": true/false,
    "multi_page_document": true/false,
    "document_quality": "high/medium/low",
    "extraction_confidence": "high/medium/low",
    "unclear_fields": []
  }}
}}

INSTRUCTIONS:
- Be honest about image quality in the quality_assessment section
- If quality_too_poor is true, still try to extract what you can see
- For missing/unclear fields, use "N/A"
# In your try_process_image function, enhance the currency instructions:

CURRENCY DETECTION - IMPORTANT:
- ALWAYS preserve currency symbols in amounts: $154.06, ₹10,000, €500, etc.
- Include currency symbols in ALL amount fields: total_amount, subtotal, unit_price, etc.
- Do NOT extract just numbers - include the currency symbol with the number
- Look for currency symbols: ₹, $, €, £, ¥, etc.
- Look for currency codes: INR, USD, EUR, GBP, etc.
- Look for currency words: Rupees, Dollars, Euros, Pounds, etc.
- Extract currency in BOTH invoice_header and financial_summary sections
- If amounts have symbols like $100.00, preserve the $ in the JSON output

- If text is completely unreadable due to quality, mention this in quality_issues
- Extract ALL visible text and data fields
- For terms and conditions, extract the full text even if lengthy
- Include any fine print, disclaimers, or legal text
- Capture payment terms like "Net 30", "Due on receipt", etc.
- Extract tax breakdowns (CGST, SGST, IGST) if present
- Include any special notes, delivery instructions, or remarks
- Identify HSN/SAC codes for items if visible
- Extract complete addresses with pin codes
- Include contact details like phone, email, website
- Capture bank details for payments
- Note any stamps, signatures, or authentication marks
- Return only valid JSON without any explanation.
"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,  # 🔧 INCREASED from 2500 to 4000
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        cleaned_content = clean_json_response(content)
        result = json.loads(cleaned_content)

# 🔧 NEW: Enhance currency detection
        result = enhance_currency_detection(result)

        return result

        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON parsing error: {e}")
        # 🔧 NEW: Try simplified extraction for large documents
        return try_simplified_extraction(client, model, image_path, is_preprocessed)
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        return None



def is_invoice_check(result):
    if result is None:
        return False
    quality = result.get('quality_assessment', {})
    issues = quality.get('quality_issues', [])
    if not isinstance(issues, list):
        return False
    issues_lower = [str(issue).strip().lower() for issue in issues]
    can_extract = quality.get('can_extract_data', True)
    return 'not invoice' in issues_lower or not can_extract

def process_invoice_with_retry(image_path):
    API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    if not API_KEY:
        st.error("❌ OpenAI API key not found. Please check your .env file.")
        return None

    client = OpenAI(api_key=API_KEY)

    st.info("🔄 **Step 1:** Trying with original image...")
    result = try_process_image(client, MODEL, image_path, is_preprocessed=False)
    condition = analyze_invoice_quality(result)

    if condition == 'not_invoice':
        st.error("❌ Uploaded file is not a valid invoice. Please upload a proper invoice document.")
        return None

    if condition == 'blur_too_bad':
        st.error("❌ Uploaded invoice is too blurry or unreadable.")
        return None

    if condition == 'blur_maybe':
        st.warning("⚠️ Uploaded invoice is blurry; attempting enhancement.")
        preprocessed_path = preprocess_image_enhanced(image_path)
        download_preprocessed_image(preprocessed_path)
        result = try_process_image(client, MODEL, preprocessed_path, is_preprocessed=True)
        condition = analyze_invoice_quality(result)

        if condition == 'not_invoice':
            st.error("❌ Uploaded file is not a valid invoice after enhancement.")
            return None

        if condition == 'blur_too_bad':
            st.error("❌ Uploaded invoice is too blurry even after enhancement.")
            return None

    return result
def analyze_invoice_quality(result):
    if not result:
        return 'no_data'
    qa = result.get('quality_assessment', {})
    can_extract = qa.get('can_extract_data', True)
    issues = qa.get('quality_issues', [])
    if not isinstance(issues, list):
        issues = []
    issues_lower = [str(i).strip().lower() for i in issues]
    qtp = qa.get('quality_too_poor', False)
    rs = qa.get('readability_score', '').lower()

    if not can_extract and 'not invoice' in issues_lower:
        return 'not_invoice'
    if not can_extract or qtp or rs == 'low':
        return 'blur_too_bad'
    if qtp or rs == 'medium':
        return 'blur_maybe'
    return 'good'



def display_json_results(result):
    """Display JSON results in organized tabs"""
    if not result:
        return
    
    # Create tabs for different sections
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 Invoice Header", 
    "👤 Customer Details", 
    "📦 Line Items", 
    "💰 Financial Summary",
    "💳 Payment Details",
    "📜 Terms & Conditions",
    "ℹ️ Additional Info",
    "💱 Currency"  # 🔧 NEW TAB
    ])

    
    with tab1:
        st.subheader("Invoice Header Information")
        if "invoice_header" in result:
            for key, value in result["invoice_header"].items():
                if value and value != "N/A":
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    with tab2:
        st.subheader("Customer Details")
        if "customer_details" in result:
            for key, value in result["customer_details"].items():
                if value and value != "N/A":
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    with tab3:
        st.subheader("Line Items")
        if "line_items" in result and result["line_items"]:
            for i, item in enumerate(result["line_items"], 1):
                st.write(f"**Item {i}:**")
                for key, value in item.items():
                    if value and value != "N/A":
                        st.write(f"  -  {key.replace('_', ' ').title()}: {value}")
                st.divider()
    
    with tab4:
        st.subheader("Financial Summary")
        if "financial_summary" in result:
            col1, col2 = st.columns(2)
            items = list(result["financial_summary"].items())
            mid = len(items) // 2
            
            with col1:
                for key, value in items[:mid]:
                    if value and value != "N/A":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                for key, value in items[mid:]:
                    if value and value != "N/A":
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    with tab5:
        st.subheader("Payment Details")
        if "payment_details" in result:
            for key, value in result["payment_details"].items():
                if value and value != "N/A":
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    with tab6:
        st.subheader("Terms & Conditions")
        if "terms_and_conditions" in result:
            for key, value in result["terms_and_conditions"].items():
                if value and value != "N/A":
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}:**")
                        for item in value:
                            st.write(f"  -  {item}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    with tab7:
        st.subheader("Additional Information & Metadata")
        if "additional_info" in result:
            st.write("**Additional Info:**")
            for key, value in result["additional_info"].items():
                if value and value != "N/A":
                    st.write(f"  -  {key.replace('_', ' ').title()}: {value}")
        
        if "detection_metadata" in result:
            st.write("**Detection Metadata:**")
            for key, value in result["detection_metadata"].items():
                if value and value != "N/A":
                    st.write(f"  -  {key.replace('_', ' ').title()}: {value}")

    with tab8:
        st.subheader("Currency Information")
    
    # Get currency from different sections
        header_currency = result.get("invoice_header", {}).get("currency", "N/A")
        financial_currency = result.get("financial_summary", {}).get("currency", "N/A")
    
    # Display main currency
        if header_currency and header_currency != "N/A":
            st.write(f"**💰 Primary Currency:** {header_currency}")
        elif financial_currency and financial_currency != "N/A":
            st.write(f"**💰 Primary Currency:** {financial_currency}")
        else:
            st.write(f"**💰 Primary Currency:** Rupees (₹) *(default)*")
    
    # Show currency detection details if available
        if "additional_info" in result and "currency_detection" in result["additional_info"]:
            currency_info = result["additional_info"]["currency_detection"]
        
            st.divider()
            st.write("**🔍 Detection Details:**")
            st.write(f"- **Detected Currency:** {currency_info.get('detected_currency', 'N/A')}")
            st.write(f"- **Final Currency:** {currency_info.get('final_currency', 'N/A')}")
            st.write(f"- **Detection Method:** {currency_info.get('detection_method', 'N/A')}")
    
    # Show currency in amounts if found
        st.divider()
        st.write("**💲 Currency in Amounts:**")
    
    # Check amounts from financial summary
        if "financial_summary" in result:
            financial = result["financial_summary"]
            currency_amounts = []
        
            for key, value in financial.items():
                if value and value != "N/A" and any(symbol in str(value) for symbol in ['₹', '$', '€', '£', 'Rs', 'INR', 'USD']):
                    currency_amounts.append(f"**{key.replace('_', ' ').title()}:** {value}")
        
            if currency_amounts:
                for amount in currency_amounts[:5]:  # Show first 5 amounts with currency
                    st.write(amount)
            else:
                st.write("- No currency symbols found in amounts")
    
    # Show currency indicators
        st.divider()
        st.write("**🏦 Currency Indicators Found:**")
    
        indicators = []
        if result.get("financial_summary", {}).get("cgst", "N/A") != "N/A":
            indicators.append("CGST (Indian tax)")
        if result.get("financial_summary", {}).get("sgst", "N/A") != "N/A":
            indicators.append("SGST (Indian tax)")
        if result.get("financial_summary", {}).get("igst", "N/A") != "N/A":
            indicators.append("IGST (Indian tax)")
        if result.get("invoice_header", {}).get("vendor_gst_number", "N/A") != "N/A":
            indicators.append("GST Number (Indian)")
        if result.get("invoice_header", {}).get("vendor_pan", "N/A") != "N/A":
            indicators.append("PAN Number (Indian)")
    
        if indicators:
            for indicator in indicators:
                st.write(f"- ✅ {indicator}")
        else:
            st.write("- No specific currency indicators detected")


# Main Streamlit App
def main():
    # Header
    st.title("🧾 Invoice Data Extractor")
    st.markdown("**Extract comprehensive data from invoices using OpenAI GPT-4o Vision**")
    st.divider()
    
    # Sidebar for settings
    with  st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key status
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Add your API key to the .env file")
        
        # Model info
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        st.info(f"🤖 Model: {model}")
        
        st.divider()
        
        # Instructions
        st.markdown("""
        ### 📝 Instructions:
        1. Upload an invoice image or PDF
        2. Click 'Process Invoice/PDF'
        3. View extracted data in tabs
        4. Download JSON if needed
        
        **Supported formats:**
        - JPG, JPEG, PNG, TIFF, PDF
        """)
        
        st.divider()
        
        # Results Management Section
        st.header("📁 Results Management")
        
        # Get results summary
        summary = get_results_summary()
        
        if summary["total_files"] > 0:
            st.success(f"📊 **{summary['total_files']}** documents processed")
            st.info(f"🕐 **Latest:** {summary['latest_processing']}")
            
            # Download master results file
            master_file_path = os.path.join("resultjson", "all_results.json")
            if os.path.exists(master_file_path):
                with open(master_file_path, 'r', encoding='utf-8') as f:
                    master_data = f.read()
                
                st.download_button(
                    label="📥 Download All Results",
                    data=master_data,
                    file_name=f"all_invoice_results_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # View results folder contents
            if st.button("📂 View Results Folder", use_container_width=True):
                st.session_state.show_results = True
        else:
            st.info("📋 No results saved yet")
        
        # Show results folder contents if requested
        if st.session_state.get('show_results', False):
            st.subheader("📂 Saved Files")
            results_dir = "resultjson"
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
                for file in sorted(files, reverse=True)[:5]:  # Show latest 5 files
                    st.write(f"📄 `{file}`")
                if len(files) > 5:
                    st.write(f"... and {len(files) - 5} more files")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Invoice")
        
        # ENHANCED FILE UPLOADER WITH PDF SUPPORT
        uploaded_file = st.file_uploader(
            "Choose an invoice image or PDF...",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'pdf'],  # Added PDF support
            help="Upload an image or PDF of your invoice for data extraction"
        )
        
        if uploaded_file is not None:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                # PDF PROCESSING
                st.info("📄 **PDF file detected**")
                
                # Save PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Show PDF info
                page_count = get_pdf_page_count(tmp_path)
                st.write(f"📊 **Pages:** {page_count}")
                
                # Process button for PDF
                if st.button("🚀 Process PDF Invoice", type="primary", use_container_width=True):
                    # Create results directory
                    results_dir = create_results_directory()
                    
                    # Process multi-page PDF
                    result = process_multi_page_pdf(tmp_path, uploaded_file.name)
                    
                    # Clean up temp PDF file
                    os.unlink(tmp_path)
                    
                    if result:
                        # Store result in session state
                        st.session_state.processed_result = result
                        # Write Financial Logic Agent input
                        out_path = write_financial_agent_input_file(result, "input.json")
                        if out_path:
                            st.success(f"🧩 Financial Logic input saved to {out_path}")
                            # Run Financial Logic Agent validation inline
                            if logic is not None:
                                try:
                                    with open(out_path, "r", encoding="utf-8") as f:
                                        fin_data = json.load(f)
                                    # Kick off validation (agent is async)
                                    async def _run_validation():
                                        await logic.init_sqlite_db()
                                        report = logic.parse_expense_report(fin_data)
                                        user = report.employee
                                        agent = logic.FinancialLogicAgent(user)
                                        return await agent.validate_expense_report(report)
                                    with st.spinner('🔍 Running Financial Logic Agent...'):
                                        validation_result = asyncio.run(_run_validation())
                                    # Display summary in UI
                                    st.subheader("🔍 Financial Logic Agent Result")
                                    st.write(f"Valid: {validation_result.is_valid}")
                                    if validation_result.violations:
                                        st.error("Violations:")
                                        for v in validation_result.violations:
                                            st.write(f"- {v}")
                                    if validation_result.warnings:
                                        st.warning("Warnings:")
                                        for w in validation_result.warnings:
                                            st.write(f"- {w}")
                                    st.write(f"Risk Score: {validation_result.risk_score}")
                                    st.write(f"Recommended Action: {validation_result.recommended_action}")
                                except Exception as e:
                                    st.error(f"❌ Failed to run Financial Logic Agent: {e}")
                        
                        # Show processing summary
                        pdf_info = result.get('pdf_info', {})
                        st.success(f"✅ **PDF processed successfully!**")
                        st.info(f"📊 **Summary:** {pdf_info.get('successful_pages', 0)}/{pdf_info.get('total_pages', 0)} pages processed")
                        
                        # Save results
                        with st.spinner('💾 Saving results...'):
                            try:
                                # Save individual file
                                individual_file, enhanced_result = save_result_to_file(
                                    result, uploaded_file.name, results_dir
                                )
                                
                                # Append to master results
                                master_file, total_count = append_to_master_results(
                                    result, uploaded_file.name, results_dir
                                )
                                
                                st.success(f"💾 **Results automatically saved:**")
                                st.write(f"📁 **Individual file:** `{os.path.basename(individual_file)}`")
                                st.write(f"📋 **Master file:** `all_results.json` (Total: {total_count} documents)")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to save results: {e}")
                    else:
                        st.error("❌ Failed to process PDF")
            
            else:
                # IMAGE PROCESSING (your existing code)
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                
                # Process button for images
                if st.button("🚀 Process Invoice", type="primary", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Create results directory
                    results_dir = create_results_directory()
                    
                    # Process with smart retry
                    result = process_invoice_with_retry(tmp_path)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    if result:
                        # Store result in session state
                        st.session_state.processed_result = result
                        # Write Financial Logic Agent input
                        out_path = write_financial_agent_input_file(result, "input.json")
                        if out_path:
                            st.success(f"🧩 Financial Logic input saved to {out_path}")
                            # Run Financial Logic Agent validation inline
                            if logic is not None:
                                try:
                                    with open(out_path, "r", encoding="utf-8") as f:
                                        fin_data = json.load(f)
                                    async def _run_validation():
                                        await logic.init_sqlite_db()
                                        report = logic.parse_expense_report(fin_data)
                                        user = report.employee
                                        agent = logic.FinancialLogicAgent(user)
                                        return await agent.validate_expense_report(report)
                                    with st.spinner('🔍 Running Financial Logic Agent...'):
                                        validation_result = asyncio.run(_run_validation())
                                    st.subheader("🔍 Financial Logic Agent Result")
                                    st.write(f"Valid: {validation_result.is_valid}")
                                    if validation_result.violations:
                                        st.error("Violations:")
                                        for v in validation_result.violations:
                                            st.write(f"- {v}")
                                    if validation_result.warnings:
                                        st.warning("Warnings:")
                                        for w in validation_result.warnings:
                                            st.write(f"- {w}")
                                    st.write(f"Risk Score: {validation_result.risk_score}")
                                    st.write(f"Recommended Action: {validation_result.recommended_action}")
                                except Exception as e:
                                    st.error(f"❌ Failed to run Financial Logic Agent: {e}")
                        
                        # Show quality assessment
                        if "quality_assessment" in result:
                            quality = result["quality_assessment"]
                            
                            if quality.get("quality_too_poor", False):
                                st.warning("⚠️ **Image quality was initially poor - preprocessing was applied**")
                            
                            st.info(f"📊 **Quality Assessment:**")
                            st.write(f"- **Readability:** {quality.get('readability_score', 'N/A')}")
                            st.write(f"- **Can extract data:** {quality.get('can_extract_data', 'N/A')}")
                            
                            if quality.get("quality_issues"):
                                st.write("**Quality issues detected:**")
                                for issue in quality["quality_issues"]:
                                    st.write(f"  - {issue}")
                        
                        # Automatically save to JSON files
                        with st.spinner('💾 Saving results...'):
                            try:
                                # Save individual file
                                individual_file, enhanced_result = save_result_to_file(
                                    result, uploaded_file.name, results_dir
                                )
                                
                                # Append to master results
                                master_file, total_count = append_to_master_results(
                                    result, uploaded_file.name, results_dir
                                )
                                
                                # Success messages
                                st.success("✅ Invoice processed successfully!")
                                st.success(f"💾 **Results automatically saved:**")
                                st.write(f"📁 **Individual file:** `{os.path.basename(individual_file)}`")
                                st.write(f"📋 **Master file:** `all_results.json` (Total: {total_count} documents)")
                                st.write(f"📂 **Location:** `{results_dir}/`")
                                
                            except Exception as e:
                                st.error(f"❌ Failed to save results: {e}")
                                st.success("✅ Invoice processed successfully (but not saved)")
                    else:
                        st.error("❌ Failed to process invoice even after preprocessing")
        else:
            st.info("👆 Please upload an invoice image or PDF to get started")
    
    with col2:
        # Create tabs for extracted data and results history
        data_tab, history_tab = st.tabs(["📊 Current Results", "📁 Results History"])
        
        with data_tab:
            st.header("📊 Extracted Data")
            
            if st.session_state.processed_result:
                # Check if it's a PDF result or single image result
                # Check if it's a PDF result or single image result
                if 'pdf_info' in st.session_state.processed_result:
    # PDF results
                    display_pdf_results(st.session_state.processed_result)
    
    # 👇 ADD DEBUG CODE RIGHT HERE 👇
                    with st.expander("🐛 DEBUG: Raw Page Results"):
                        page_results = st.session_state.processed_result.get('page_by_page_results', [])
                    for i, page_result in enumerate(page_results, 1):
                        st.write(f"**Page {i} Results:**")
                        st.json(page_result)
                        st.divider()
    # 👆 END DEBUG CODE 👆
    
                else:
    # Single image results
                    display_json_results(st.session_state.processed_result)

                
                # Download current result
                st.divider()
                json_str = json.dumps(st.session_state.processed_result, indent=2, ensure_ascii=False)
                
                # Dynamic filename based on content type
                filename = "current_pdf_data.json" if 'pdf_info' in st.session_state.processed_result else "current_invoice_data.json"
                
                st.download_button(
                    label="📥 Download Current JSON",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("📋 Upload and process an invoice or PDF to see extracted data here")
        
        with history_tab:
            st.header("📁 Processing History")
            
            # Show master results if available
            master_file_path = os.path.join("resultjson", "all_results.json")
            if os.path.exists(master_file_path):
                try:
                    with open(master_file_path, 'r', encoding='utf-8') as f:
                        all_results = json.load(f)
                    
                    if isinstance(all_results, list) and all_results:
                        st.write(f"**📊 Total Processed: {len(all_results)} documents**")
                        
                        # Show recent results
                        for i, entry in enumerate(reversed(all_results[-10:])):  # Show last 10
                            with st.expander(f"📄 {entry.get('source_filename', 'Unknown')} - {entry.get('processing_date', 'Unknown date')}"):
                                if 'extraction_data' in entry:
                                    # Handle both PDF and single image results
                                    extraction_data = entry['extraction_data']
                                    
                                    if 'pdf_info' in extraction_data:
                                        # PDF result
                                        pdf_info = extraction_data['pdf_info']
                                        combined_data = extraction_data.get('combined_data', {})
                                        vendor = combined_data.get('invoice_header', {}).get('vendor_name', 'N/A')
                                        total = combined_data.get('financial_summary', {}).get('total_amount', 'N/A')
                                        st.write(f"**Type:** PDF ({pdf_info.get('total_pages', 0)} pages)")
                                        st.write(f"**Vendor:** {vendor}")
                                        st.write(f"**Total Amount:** {total}")
                                    else:
                                        # Single image result
                                        vendor = extraction_data.get('invoice_header', {}).get('vendor_name', 'N/A')
                                        total = extraction_data.get('financial_summary', {}).get('total_amount', 'N/A')
                                        st.write(f"**Type:** Single Image")
                                        st.write(f"**Vendor:** {vendor}")
                                        st.write(f"**Total Amount:** {total}")
                                    
                                    st.write(f"**Processed:** {entry.get('processing_date', 'N/A')}")
                                    
                                    # Option to view full JSON
                                    st.json(extraction_data)
                    else:
                        st.info("📋 No processing history available")
                except Exception as e:
                    st.error(f"❌ Error loading history: {e}")
            else:
                st.info("📋 No processing history found")
    
    # Raw JSON view (collapsible)
    if st.session_state.processed_result:
        with st.expander("🔍 View Raw JSON Output"):
            st.json(st.session_state.processed_result)

if __name__ == "__main__":
    main()
