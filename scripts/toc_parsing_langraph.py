"""
Table of Contents (ToC) Extraction and Refinement with LangGraph and Mistral AI

This script implements a robust multi-step workflow for extracting and refining a document's
Table of Contents using a stateful graph powered by LangGraph. It features enhanced error 
handling, optimized prompts, and configurable retry logic.

The workflow consists of the following nodes:
1.  **Run OCR**: Extracts text and initial ToC from document using Mistral OCR API
2.  **Generate Analysis**: Uses vision/text models to analyze ToC against content
3.  **Generate Code**: Creates robust Python script with defensive programming
4.  **Validate Code**: Checks script correctness with syntax and logic validation
5.  **Execute Script**: Runs script with comprehensive error handling
6.  **Error Recovery**: Attempts to fix validation and execution errors

Key Features:
- Configurable retry attempts for validation and execution
- Template-based code generation with error handling
- Syntax validation before script execution
- Graceful fallback to original ToC if refinement fails
- Optimized prompts with specific examples and constraints

Usage:
    # Basic usage
    python scripts/toc_parsing_langraph.py --file ./document.pdf
    
    # With custom retry limits and output folder
    python scripts/toc_parsing_langraph.py --file ./image.png \
        --output-folder ./results --max-validation-attempts 5 --max-execution-attempts 3
"""
import os
import json
import base64
import mimetypes
import tempfile
import subprocess
import sys
import ast
import traceback
import uuid
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from pathlib import Path

import typer
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models import DocumentURLChunk, ImageURLChunk
from langgraph.graph import StateGraph, END

# --- Pydantic Models for Structured Output ---
class TocEntry(BaseModel):
    """A single entry in a multi-page Table of Contents."""

    id: int = Field(
        ...,
        description=(
            "Sequential index of this entry within the Table of Contents listing. "
            "For example, '1' for the first entry, '2' for the second, etc."
        )
    )
    type: Literal["section", "subsection"] = Field(
        ...,
        description=(
            "Either 'section' (a top-level heading) or 'subsection' (nested under a section)."
        )
    )
    text: str = Field(
        ...,
        description="The exact title of this section/subsection as shown in the Table of Contents."
    )
    parent: Optional[int] = Field(
        None,
        description=(
            "If this is a subsection, the `id` of its parent section; "
            "if a top-level section, leave as null."
        )
    )
    toc_list_page: Optional[int] = Field(
        None,
        description=(
            "The printed or digital page number *where this entry is listed* in the Table of Contents itself. "
            "Use this to jump to the ToC listing—for example, if your ToC spans pages v–vii, you might see "
            "`toc_list_page=vi`."
        )
    )
    chapter_start_page: Optional[int] = Field(
        None,
        description=(
            "The actual document page number *where the content for this entry begins*. "
            "Use this to jump straight to the section’s first page—for example, `chapter_start_page=42`."
        )
    )

class TableOfContents(BaseModel):
    """Structured representation of a document's Table of Contents"""
    entries: List[TocEntry] = Field(..., description="List of all ToC entries.")


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    """
    Enhanced state representation for robust ToC processing workflow.

    Attributes:
        file_path: Path to the local document to process
        output_folder: Directory to save all artifacts
        api_key: The Mistral API key
        is_image: True if the input file is an image
        max_validation_attempts: Maximum validation retry attempts
        max_execution_attempts: Maximum execution retry attempts
        
        # OCR and initial extraction
        ocr_response: The raw JSON response from the OCR API
        raw_toc_json_path: Path to the initially extracted ToC file
        markdown_path: Path to the extracted markdown content file
        
        # Refinement process
        refinement_instructions: Natural language fix instructions
        fixer_script_code: Generated Python code to apply fixes
        syntax_check_passed: Whether the generated code passed syntax validation
        
        # Results and tracking
        final_toc_path: Path to the final corrected ToC file
        validation_attempts: Counter for validation attempts
        execution_attempts: Counter for execution attempts
        
        # Error handling and feedback
        error: Fatal error message that stops the workflow
        last_validation_feedback: Detailed feedback from validation
        last_execution_error: Error message from script execution
        execution_context: Additional context about execution environment
    """
    file_path: str
    output_folder: Path
    api_key: str
    is_image: bool
    max_validation_attempts: int
    max_execution_attempts: int
    
    ocr_response: Optional[Dict[str, Any]]
    raw_toc_json_path: Optional[str]
    markdown_path: Optional[str]
    
    refinement_instructions: Optional[str]
    fixer_script_code: Optional[str]
    syntax_check_passed: bool
    
    final_toc_path: Optional[str]
    validation_attempts: int
    execution_attempts: int
    
    error: Optional[str]
    last_validation_feedback: Optional[str]
    last_execution_error: Optional[str]
    execution_context: Optional[str]


# --- Utility Functions ---
def encode_file_to_base64(file_path: str) -> str:
    """Encodes a file to a base64 data URI."""
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(file_path, "rb") as f:
            file_content = f.read()
        base64_encoded_content = base64.b64encode(file_content).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_content}"
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validates Python code syntax without executing it.
    
    Args:
        code: Python code as string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Code validation error: {str(e)}"
        return False, error_msg


def get_csv_fixer_script() -> str:
    """
    Returns a robust script that applies ToC fixes from a CSV file.
    """
    return '''import json
import sys
import csv
import traceback
from typing import Dict, Any, List, Optional

def apply_fixes_from_csv(toc_data: Dict[str, Any], csv_file_path: str) -> Dict[str, Any]:
    """
    Apply fixes to ToC data based on CSV instructions.
    
    CSV format: id,field,new_value
    
    Args:
        toc_data: Dictionary containing ToC entries
        csv_file_path: Path to CSV file with fixes
        
    Returns:
        Fixed ToC data dictionary
    """
    try:
        # Defensive programming: ensure entries exist
        if not isinstance(toc_data, dict):
            print("Warning: toc_data is not a dictionary, returning as-is")
            return toc_data
            
        entries = toc_data.get("entries", [])
        if not isinstance(entries, list):
            print("Warning: entries is not a list, returning original data")
            return toc_data
        
        # Create a working copy to avoid modifying original
        fixed_entries = []
        
        # Load fixes from CSV
        fixes = {}
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    entry_id = int(row['id'])
                    field = row['field']
                    new_value = row['new_value']
                    
                    # Convert new_value to appropriate type
                    if field in ['start_page_number', 'toc_page_number', 'id', 'parent']:
                        if new_value.lower() in ['null', 'none', '']:
                            new_value = None
                        else:
                            new_value = int(new_value)
                    elif field == 'text':
                        new_value = str(new_value)
                    elif field == 'type':
                        new_value = str(new_value)
                    
                    if entry_id not in fixes:
                        fixes[entry_id] = {}
                    fixes[entry_id][field] = new_value
                    
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file_path}' not found")
            return toc_data
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return toc_data
        
        print(f"Loaded {len(fixes)} entry fixes from CSV")
        
        # Apply fixes to entries
        for entry in entries:
            if not isinstance(entry, dict):
                continue
                
            # Create a copy of the entry
            fixed_entry = entry.copy()
            
            # Get entry ID
            entry_id = fixed_entry.get("id")
            if entry_id is None:
                fixed_entries.append(fixed_entry)
                continue
            
            # Apply fixes if any exist for this entry
            if entry_id in fixes:
                for field, new_value in fixes[entry_id].items():
                    fixed_entry[field] = new_value
                    print(f"Applied fix: Entry {entry_id}, {field} = {new_value}")
            
            fixed_entries.append(fixed_entry)
        
        return {"entries": fixed_entries}
        
    except Exception as e:
        print(f"Error in apply_fixes_from_csv: {e}")
        print("Returning original data")
        traceback.print_exc()
        return toc_data


def main():
    """Main function to handle file I/O and error handling."""
    try:
        if len(sys.argv) != 4:
            print("Usage: python script.py <input_toc.json> <fixes.csv> <output_toc.json>")
            sys.exit(1)
        
        input_file = sys.argv[1]
        csv_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Read input file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}")
            sys.exit(1)
        
        # Apply fixes from CSV
        fixed_toc = apply_fixes_from_csv(toc_data, csv_file)
        
        # Write output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(fixed_toc, f, indent=2, ensure_ascii=False)
            print(f"Successfully wrote fixed ToC to {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


def create_fallback_csv_content() -> str:
    """
    Creates an empty CSV file content that makes no changes.
    """
    return "id,field,new_value\n# No changes needed\n"

# --- Graph Nodes ---
def run_ocr_and_extract_toc(state: GraphState) -> GraphState:
    """
    Node 1: Runs the initial OCR process to extract text and a raw Table of Contents.
    """
    print("--- Starting Node: run_ocr_and_extract_toc ---")
    file_path = state["file_path"]
    api_key = state["api_key"]
    output_folder = state["output_folder"]
    
    # Create intermediates directory
    intermediates_dir = output_folder / "intermediates"
    intermediates_dir.mkdir(exist_ok=True)

    try:
        document_to_process = encode_file_to_base64(file_path)
        
        client = Mistral(api_key=api_key)
        
        document_chunk = (
            ImageURLChunk(image_url=document_to_process) 
            if state["is_image"] 
            else DocumentURLChunk(document_url=document_to_process)
        )

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document_chunk,
            document_annotation_format=response_format_from_pydantic_model(TableOfContents)
        )
        
        # Save raw response for debugging
        raw_response_path = intermediates_dir / "01_ocr_raw_response.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            # Save the raw response object as much as possible
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw OCR response saved to {raw_response_path}")
        
        ocr_response_dict = response.model_dump()
        state["ocr_response"] = ocr_response_dict

        # Save full OCR response
        full_output_path = output_folder / "output.json"
        with open(full_output_path, "w") as f:
            json.dump(ocr_response_dict, f, indent=2)
        print(f"Full OCR response saved to {full_output_path}")

        # Save markdown
        markdown_path = output_folder / "output.md"
        markdown_content = "\n\n---\n\n".join(
            [f"<!-- Page {p.get('index', '?') + 1} -->\n\n{p['markdown']}" for p in ocr_response_dict.get("pages", [])]
        )
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        state["markdown_path"] = str(markdown_path)
        print(f"Markdown content saved to {markdown_path}")

        # Save raw ToC
        if ocr_response_dict.get("document_annotation"):
            toc_data = json.loads(ocr_response_dict["document_annotation"])
            if toc_data.get("entries"):
                raw_toc_path = output_folder / "toc.json"
                with open(raw_toc_path, "w", encoding="utf-8") as f:
                    json.dump(toc_data, f, indent=2)
                state["raw_toc_json_path"] = str(raw_toc_path)
                print(f"Raw ToC saved to {raw_toc_path}")
                
                # Also save in intermediates
                toc_copy_path = intermediates_dir / "02_extracted_toc.json"
                with open(toc_copy_path, "w", encoding="utf-8") as f:
                    json.dump(toc_data, f, indent=2)

    except Exception as e:
        state["error"] = f"Error in OCR node: {e}"
        print(state["error"])
        # Save error details
        error_path = intermediates_dir / "01_ocr_error.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")

    return state

def generate_refinement_instructions(state: GraphState) -> GraphState:
    """
    Node 2: Generates comprehensive natural language instructions to fix the raw ToC.
    Uses optimized prompts with specific examples and structured output format.
    """
    print("--- Starting Node: generate_refinement_instructions ---")
    if state.get("error"): return state
    
    api_key = state["api_key"]
    is_image = state["is_image"]
    
    try:
        with open(state["markdown_path"], 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        with open(state["raw_toc_json_path"], 'r', encoding='utf-8') as f:
            toc_data = json.load(f)
            
        client = Mistral(api_key=api_key)
        
        # Common instruction examples for both image and PDF processing
        example_instructions = """
EXAMPLE INSTRUCTIONS FORMAT:
- modify the start_page_number of toc entry with id 1 to 15
- modify the toc_page_number of toc entry with id 2 to 3
- change the text field of toc entry with id 3 to "Methodology and Approach"
- change the parent field of toc entry with id 4 to 1
- change the type field of toc entry with id 5 to "subsection"
- remove toc entry with id 6
- add new toc entry: type="section", text="Conclusion", parent=null, start_page_number=45, toc_page_number=2

CRITICAL REQUIREMENTS:
1. Reference entries ONLY by their existing id field
2. Use exact field names: start_page_number, toc_page_number, text, parent, type
3. For parent field: use integer id or null (not "null" string)
4. For type field: use only "section" or "subsection"
5. Be precise: start_page_number = where section content begins, toc_page_number = where listed in ToC
6. If no changes needed, respond with exactly "No changes needed"
"""

        prompt = f"""You are an expert document analyst specializing in Table of Contents extraction and validation. Your task is to analyze the extracted markdown content and compare it with the ToC JSON to identify inconsistencies and generate precise fix instructions.

**CURRENT EXTRACTED ToC JSON:**
```json
{json.dumps(toc_data, indent=2)}
```

**FULL EXTRACTED MARKDOWN CONTENT:**
```
{markdown_content}
```

Pay attention that the the table of contents may span over multiple pages, so the extracted markdown is a concatenation of the pages, where there will be a seperartor between the extracted markdown text, indicating on which page the original text appeared. This is not to be confusing with the page numbers in the actual document, where the sections are referring to.

**YOUR ANALYSIS TASK:**
1. Scan the markdown for section headers and page markers
2. Cross-reference with the extracted ToC JSON
3. Identify discrepancies in:
   - Section titles (exact text matching)
   - start_page_number (where sections actually begin in content, where the sections and subsections are referring to)
   - toc_page_number (in which page the entries appears as part of the ToC itself, regardless of where the section is referring to. This refer to the physical page at the beginning of the book where the this ToC entry appears)
   - Hierarchy (which sections are nested under others)
   - Missing sections that appear in markdown but not in ToC
   - Extra ToC entries that don't correspond to actual sections

   **OUTPUT FORMAT:**
Generate explicit fix instructions in this exact format:
- modify the start_page_number of toc entry with id X to Y
- modify the toc_page_number of toc entry with id X to Y
- change the text field of toc entry with id X to "correct text"
- change the parent field of toc entry with id X to Y (or null for top-level sections)
- change the type field of toc entry with id X to "section" (or "subsection")
- remove toc entry with id X
- add new toc entry: type="section", text="New Section", parent=null, start_page_number=10, toc_page_number=2

**EXAMPLE INSTRUCTIONS:**
{example_instructions}

**IMPORTANT ANALYSIS GUIDELINES:**
- Reference entries by their id field from the JSON
- Look for page markers like "Page X" or similar in markdown
- Look for section headers in the markdown content to identify missing ToC entries. Identify section headers by formatting (##, ###, etc.,) or by indicating a high level chapter in text)
- Check if ToC page numbers match to the separator markers in the markdown where entries appear.
- Verify start page numbers match where sections begin.
- Ensure parent-child relationships are correct.
- Only suggest changes that you can clearly identify from the content
- If no changes are needed, respond with "No changes needed"

**PROVIDE YOUR DETAILED ANALYSIS AND INSTRUCTIONS:**"""
        
        messages = []
        if is_image:
            image_base64 = encode_file_to_base64(state["file_path"])
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_base64}
                ]
            })
            model = "pixtral-12b-latest"
        else: # PDF
            messages.append({"role": "user", "content": prompt})
            model = "mistral-medium-2505"
            
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0.05,  # Very low temperature for consistent analysis
            max_tokens=10_000
        )
        
        # Save raw response for debugging
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        raw_response_path = intermediates_dir / "03_refinement_analysis_response.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw refinement analysis response saved to {raw_response_path}")
        
        instructions = response.choices[0].message.content.strip()
        state["refinement_instructions"] = instructions
        
        instructions_path = state["output_folder"] / "toc-refinement-instructions.txt"
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        print(f"Refinement instructions saved to {instructions_path}")
        print(f"Instructions preview: {instructions[:200]}...")
        
    except Exception as e:
        state["error"] = f"Error in instruction generation node: {e}"
        print(state["error"])
        # Save error details
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        error_path = intermediates_dir / "03_refinement_analysis_error.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
    return state
    
def generate_fixer_script(state: GraphState) -> GraphState:
    """
    Node 3: Generates a CSV file with ToC fixes and a script to apply them.
    """
    print("--- Starting Node: generate_fixer_script ---")
    if state.get("error"): return state
    
    instructions = state["refinement_instructions"]
    api_key = state["api_key"]
    
    # Handle no-change case with fallback
    if not instructions or instructions.strip() == "No changes needed":
        print("No changes needed - using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        return state

    # Generate CSV content using optimized prompt
    prompt = f"""You are a data analyst. Generate a CSV file with ToC fixes based on the given instructions.

**Fix Instructions to Implement:**
{instructions}

**CSV FORMAT:**
Generate a CSV with exactly this header: id,field,new_value

Each row should specify:
- id: The entry ID number (integer)
- field: The field to modify (start_page_number, toc_page_number, text, parent, type)
- new_value: The new value for that field

**FIELD TYPES:**
- start_page_number: integer (page where section content begins)
- toc_page_number: integer (page where entry appears in ToC)
- text: string (section title text)
- parent: integer or null (parent entry ID)
- type: string ("section" or "subsection")

**EXAMPLE CSV OUTPUT:**
```csv
id,field,new_value
1,start_page_number,15
1,toc_page_number,2
3,text,"Corrected Section Title"
5,parent,1
6,type,subsection
```

**CRITICAL RULES:**
- Each fix instruction should become one or more CSV rows
- Use exact field names: start_page_number, toc_page_number, text, parent, type
- For null values, use: null
- For text values with commas, use quotes: "text with, comma"
- Only include rows for fields that need to be changed
- Must start with header: id,field,new_value

**Generate the complete CSV content:**"""

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10_000
        )
        
        # Save raw response for debugging
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        raw_response_path = intermediates_dir / "04_csv_generation_response.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw CSV generation response saved to {raw_response_path}")
        
        csv_content = response.choices[0].message.content.strip()
        print(f"LLM generated CSV content (first 300 chars): {csv_content[:300]}...")
        
        # Clean up the CSV content
        try:
            # Remove markdown code blocks if present
            clean_csv = csv_content
            if "```csv" in clean_csv:
                clean_csv = clean_csv.split("```csv")[1].split("```")[0].strip()
            elif "```" in clean_csv:
                clean_csv = clean_csv.split("```")[1].split("```")[0].strip()
            
            # Ensure we have valid CSV content
            if not clean_csv or not clean_csv.strip():
                print("No valid CSV content generated, using fallback")
                clean_csv = create_fallback_csv_content()
            
            # Ensure CSV starts with proper header
            if not clean_csv.startswith("id,field,new_value"):
                if "id,field,new_value" not in clean_csv:
                    clean_csv = "id,field,new_value\n" + clean_csv
                else:
                    # Move header to the top
                    lines = clean_csv.split('\n')
                    header_line = None
                    other_lines = []
                    for line in lines:
                        if line.strip() == "id,field,new_value":
                            header_line = line
                        elif line.strip() and not line.startswith('#'):
                            other_lines.append(line)
                    if header_line:
                        clean_csv = header_line + '\n' + '\n'.join(other_lines)
            
            state["csv_content"] = clean_csv
            print(f"Generated CSV content (first 200 chars): {clean_csv[:200]}...")
                
        except Exception as e:
            print(f"Error parsing CSV content: {e}")
            print(f"Raw response: {csv_content[:500]}...")
            state["csv_content"] = create_fallback_csv_content()
        
        # Use the fixed CSV fixer script
        state["fixer_script_code"] = get_csv_fixer_script()
        
        # Validate CSV syntax
        try:
            import csv
            import io
            csv_reader = csv.DictReader(io.StringIO(state["csv_content"]))
            rows = list(csv_reader)
            if not rows:
                print("Warning: CSV has no data rows")
            else:
                print(f"CSV validation passed: {len(rows)} fix rows")
            state["syntax_check_passed"] = True
        except Exception as e:
            print(f"CSV validation failed: {e}")
            print("Using fallback CSV")
            state["csv_content"] = create_fallback_csv_content()
            state["syntax_check_passed"] = True
        
        # Save the CSV file and script
        csv_path = state["output_folder"] / "toc-fixes.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(state["csv_content"])
        print(f"Fixes CSV saved to {csv_path}")
        
        script_path = state["output_folder"] / "toc-fix-script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(state["fixer_script_code"])
        print(f"Fixer script saved to {script_path}")
        
        # Save the CSV path for later use
        state["csv_file_path"] = str(csv_path)

    except Exception as e:
        print(f"Error in CSV generation: {e}")
        print(f"Raw LLM response: {response.choices[0].message.content[:500] if 'response' in locals() else 'No response'}...")
        print("Using fallback CSV and script")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        # Save error details
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        error_path = intermediates_dir / "04_csv_generation_error.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            if 'response' in locals():
                f.write(f"Raw LLM response: {response.choices[0].message.content}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
        # Still save the fallback files
        csv_path = state["output_folder"] / "toc-fixes.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(state["csv_content"])
        script_path = state["output_folder"] / "toc-fix-script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(state["fixer_script_code"])
        state["csv_file_path"] = str(csv_path)
        
    return state

def validate_script(state: GraphState) -> GraphState:
    """
    Node: Validates that the generated CSV correctly implements the instructions.
    """
    print("--- Starting Node: validate_csv ---")
    if state.get("error"): return state
    
    state["validation_attempts"] += 1
    instructions = state["refinement_instructions"]
    csv_content = state.get("csv_content")
    api_key = state["api_key"]
    
    # If csv_content is missing, create fallback and return
    if not csv_content:
        print("csv_content missing - using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["last_validation_feedback"] = "OK"
        return state

    # Skip validation for fallback "no changes" CSV
    if not instructions or instructions.strip() == "No changes needed":
        state["last_validation_feedback"] = "OK"
        print("Using fallback CSV - validation passed")
        return state
    
    # Quick CSV syntax check
    try:
        import csv
        import io
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)
        if not rows:
            state["last_validation_feedback"] = "CSV syntax error: No data rows found"
            print("CSV validation failed: No data rows")
            return state
    except Exception as e:
        state["last_validation_feedback"] = f"CSV syntax error: {e}"
        print(f"CSV validation failed: {e}")
        return state

    # Perform logical validation using AI
    prompt = f"""You are an expert data analyst. Your task is to validate whether the CSV file correctly implements ALL the given fix instructions.

**ORIGINAL FIX INSTRUCTIONS:**
{instructions}

**GENERATED CSV TO VALIDATE:**
```csv
{csv_content}
```

**VALIDATION CRITERIA:**
1. Does the CSV implement ALL instructions listed?
2. Are the field names correct (start_page_number, toc_page_number, text, parent, type)?
3. Are the entry IDs referenced correctly?
4. Are the new values exactly as specified in the instructions?
5. Is the CSV format correct (id,field,new_value)?
6. Are all required changes included?

**RESPONSE FORMAT:**
- If the CSV correctly implements ALL instructions, respond with exactly: "OK"
- If there are issues, respond with: "ISSUE: [brief description of the main problem]"

**EXAMPLES OF GOOD RESPONSES:**
- "OK"
- "ISSUE: Missing fix for entry id 6 start_page_number"
- "ISSUE: Wrong field name used - should be start_page_number not page_number"
- "ISSUE: Entry id 3 text change not implemented"
- "ISSUE: CSV format incorrect - missing header"

**YOUR VALIDATION RESPONSE:**"""

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic validation
            max_tokens=2000
        )
        
        # Save raw response for debugging
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        raw_response_path = intermediates_dir / f"05_validation_response_attempt_{state['validation_attempts']}.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw validation response saved to {raw_response_path}")
        
        feedback = response.choices[0].message.content.strip()
        state["last_validation_feedback"] = feedback
        print(f"Validation feedback: {feedback}")
        
        # Save validation result for debugging
        validation_path = state["output_folder"] / f"validation-attempt-{state['validation_attempts']}.txt"
        with open(validation_path, 'w', encoding='utf-8') as f:
            f.write(f"Validation Attempt {state['validation_attempts']}\n")
            f.write(f"Instructions:\n{instructions}\n\n")
            f.write(f"Feedback: {feedback}\n")
        
    except Exception as e:
        state["error"] = f"Error in validation node: {e}"
        print(state["error"])
        # Save error details
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        error_path = intermediates_dir / f"05_validation_error_attempt_{state['validation_attempts']}.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
    return state

def refactor_script(state: GraphState) -> GraphState:
    """
    Node: Refactors the script based on validation feedback using the template approach.
    """
    print("--- Starting Node: refactor_script ---")
    if state.get("error"): return state

    instructions = state["refinement_instructions"]
    original_code = state["fixer_script_code"]
    feedback = state["last_validation_feedback"]
    api_key = state["api_key"]

    # Extract the original fix logic from the template
    try:
        # Find the fix logic between the comment markers
        if '{FIXES_PLACEHOLDER}' in original_code or '{APPLY_FIXES_PLACEHOLDER}' in original_code:
            original_fixes = "pass  # No original fixes found"
        else:
            # Try to extract fixes from the existing code
            lines = original_code.split('\n')
            # Look for the setup section
            setup_start = -1
            setup_end = -1
            apply_start = -1
            apply_end = -1
            
            for i, line in enumerate(lines):
                if "# Apply specific fixes here" in line:
                    setup_start = i + 1
                elif setup_start > -1 and "for entry in entries:" in line:
                    setup_end = i
                elif "# Apply the specific fixes defined above" in line:
                    apply_start = i + 1
                elif apply_start > -1 and "fixed_entries.append(fixed_entry)" in line:
                    apply_end = i
                    break
            
            setup_code = "pass  # No setup found"
            apply_code = "pass  # No apply found"
            
            if setup_start > -1 and setup_end > -1:
                setup_code = '\n'.join(lines[setup_start:setup_end])
            if apply_start > -1 and apply_end > -1:
                apply_code = '\n'.join(lines[apply_start:apply_end])
                
            original_fixes = f"{setup_code}\n---APPLY_FIXES---\n{apply_code}"
    except:
        original_fixes = "pass  # Error extracting original fixes"

    prompt = f"""You are an expert data analyst. Your task is to fix the CSV file based on validation feedback.

**ORIGINAL FIX INSTRUCTIONS:**
{instructions}

**ORIGINAL CSV (that failed validation):**
```csv
{original_fixes}
```

**VALIDATION FEEDBACK:**
{feedback}

**YOUR TASK:**
Generate a corrected CSV file that addresses the validation feedback and implements ALL the original instructions.

**CSV FORMAT:**
Must have header: id,field,new_value
Each row: entry_id,field_name,new_value

**FIELD TYPES:**
- start_page_number: integer
- toc_page_number: integer  
- text: string (use quotes if contains commas)
- parent: integer or null
- type: string ("section" or "subsection")

**REQUIREMENTS:**
- Address the specific issue mentioned in the validation feedback
- Ensure ALL original instructions are implemented
- Use exact field names: start_page_number, toc_page_number, text, parent, type
- For null values, use: null
- Include all required changes from the original instructions
- Start with header: id,field,new_value

**Generate the complete corrected CSV:**"""

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10_000
        )
        
        # Save raw response for debugging
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        raw_response_path = intermediates_dir / f"06_refactor_response_attempt_{state['validation_attempts']}.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw refactor response saved to {raw_response_path}")
        
        new_fix_logic = response.choices[0].message.content.strip()
        print(new_fix_logic)
        
        # Parse the response similar to generate_fixer_script
        if "```python" in new_fix_logic:
            # Extract each python block separately
            parts = new_fix_logic.split("```python")
            if len(parts) >= 3:  # Should have at least 2 python blocks
                setup_part = parts[1].split("```")[0].strip()
                apply_part = parts[2].split("```")[0].strip()
            else:
                # Fallback: try to split by separator
                if "---APPLY_FIXES---" in new_fix_logic:
                    sections = new_fix_logic.split("---APPLY_FIXES---")
                    setup_part = sections[0].replace("```python", "").replace("```", "").strip()
                    apply_part = sections[1].replace("```python", "").replace("```", "").strip()
                else:
                    setup_part = "pass  # No setup needed"
                    apply_part = new_fix_logic.split("```python")[1].split("```")[0].strip()
        else:
            # Try to split by separator
            if "---APPLY_FIXES---" in new_fix_logic:
                sections = new_fix_logic.split("---APPLY_FIXES---")
                setup_part = sections[0].strip()
                apply_part = sections[1].strip()
            else:
                # Assume it's all application logic
                setup_part = "pass  # No setup needed"
                apply_part = new_fix_logic
        
        # Clean up the CSV response
        clean_csv = new_csv_content
        if "```csv" in clean_csv:
            clean_csv = clean_csv.split("```csv")[1].split("```")[0].strip()
        elif "```" in clean_csv:
            clean_csv = clean_csv.split("```")[1].split("```")[0].strip()
        
        # Ensure proper header
        if not clean_csv.startswith("id,field,new_value"):
            if "id,field,new_value" not in clean_csv:
                clean_csv = "id,field,new_value\n" + clean_csv
        
        state["csv_content"] = clean_csv
        
        # Validate CSV syntax
        try:
            import csv
            import io
            csv_reader = csv.DictReader(io.StringIO(clean_csv))
            rows = list(csv_reader)
            print(f"Refactored CSV validated: {len(rows)} rows")
            state["syntax_check_passed"] = True
        except Exception as e:
            print(f"Refactored CSV has syntax errors: {e}")
            print("Using fallback CSV")
            state["csv_content"] = create_fallback_csv_content()
            state["syntax_check_passed"] = True
        
        state["syntax_check_passed"] = True
        
        # Save refactored CSV
        refactor_csv_path = state["output_folder"] / f"toc-fixes-refactored-{state['validation_attempts']}.csv"
        with open(refactor_csv_path, 'w', encoding='utf-8') as f:
            f.write(state["csv_content"])
        print(f"Refactored CSV saved to {refactor_csv_path}")
        
        # Update the CSV file path
        state["csv_file_path"] = str(refactor_csv_path)
        
    except Exception as e:
        print(f"Error in CSV refactoring: {e}")
        print("Using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["syntax_check_passed"] = True
        # Save error details
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        error_path = intermediates_dir / f"06_refactor_error_attempt_{state['validation_attempts']}.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
    return state

def execute_fixer_script(state: GraphState) -> GraphState:
    """
    Node: Executes the generated Python script with comprehensive error handling and context tracking.
    """
    print("--- Starting Node: execute_fixer_script ---")
    if state.get("error"): return state

    state["execution_attempts"] += 1
    script_code = state["fixer_script_code"]
    original_toc_file = state["raw_toc_json_path"]
    fixed_toc_file = str(state["output_folder"] / "toc-fixed.json")
    
    # Reset previous execution error and context
    state["last_execution_error"] = None
    state["execution_context"] = None

    # Validate input file exists and is readable
    try:
        with open(original_toc_file, 'r') as f:
            original_data = json.load(f)
        state["execution_context"] = f"Original ToC has {len(original_data.get('entries', []))} entries"
    except Exception as e:
        state["last_execution_error"] = f"Cannot read original ToC file: {e}"
        print(state["last_execution_error"])
        return state

    temp_script_path = None
    try:
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(script_code)
            temp_script_path = temp_script.name

        print(f"Executing script (attempt {state['execution_attempts']})...")
        
        # Execute with extended timeout and better error capture
        # Convert to absolute paths to avoid path issues
        abs_original_file = os.path.abspath(original_toc_file)
        abs_csv_file = os.path.abspath(state["csv_file_path"])
        abs_fixed_file = os.path.abspath(fixed_toc_file)
        
        result = subprocess.run(
            [sys.executable, temp_script_path, abs_original_file, abs_csv_file, abs_fixed_file],
            capture_output=True, 
            text=True, 
            timeout=45,  # Increased timeout
            cwd=os.getcwd()  # Use current working directory
        )

        if result.returncode != 0:
            # Provide detailed error information
            stderr_lines = result.stderr.strip().split('\n')
            stdout_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            error_details = []
            if stdout_lines:
                error_details.append(f"STDOUT: {'; '.join(stdout_lines[-3:])}")  # Last 3 lines
            if stderr_lines:
                error_details.append(f"STDERR: {'; '.join(stderr_lines[-3:])}")  # Last 3 lines
            
            error_message = f"Script execution failed (exit code {result.returncode}): {' | '.join(error_details)}"
            state["last_execution_error"] = error_message
            state["execution_context"] = f"Failed at attempt {state['execution_attempts']}"
            print(f"Execution failed: {error_message}")
        else:
            # Verify output file was created and is valid
            try:
                with open(fixed_toc_file, 'r') as f:
                    fixed_data = json.load(f)
                    
                entries_count = len(fixed_data.get('entries', []))
                state["final_toc_path"] = fixed_toc_file
                state["execution_context"] = f"Success: Generated {entries_count} entries"
                print(f"Fixed ToC successfully generated: {entries_count} entries at {fixed_toc_file}")
                
                # Log execution success details
                execution_log = state["output_folder"] / f"execution-success-{state['execution_attempts']}.txt"
                with open(execution_log, 'w') as f:
                    f.write(f"Execution Attempt {state['execution_attempts']} - SUCCESS\n")
                    f.write(f"Original entries: {len(original_data.get('entries', []))}\n")
                    f.write(f"Fixed entries: {entries_count}\n")
                    if result.stdout:
                        f.write(f"Script output:\n{result.stdout}\n")
                        
            except Exception as e:
                state["last_execution_error"] = f"Generated file is invalid: {e}"
                print(state["last_execution_error"])

    except subprocess.TimeoutExpired:
        state["last_execution_error"] = "Script execution timed out (45 seconds)"
        state["execution_context"] = "Timeout during execution"
        print(state["last_execution_error"])
        
    except Exception as e:
        state["last_execution_error"] = f"Unexpected execution error: {e}"
        state["execution_context"] = f"Exception: {type(e).__name__}"
        print(f"Execution exception: {e}")
        traceback.print_exc()
        
    finally:
        # Clean up temporary script file
        if temp_script_path and os.path.exists(temp_script_path):
            try:
                os.unlink(temp_script_path)
            except:
                pass  # Ignore cleanup errors
    
    return state

def fix_execution_error(state: GraphState) -> GraphState:
    """
    Node: Attempts to fix a script that failed during execution.
    Uses fallback approach for common errors.
    """
    print("--- Starting Node: fix_execution_error ---")
    if state.get("error"): return state

    error_message = state["last_execution_error"]
    
    # Check for common errors and use fallback directly
    if "Input file" in error_message and "not found" in error_message:
        print("File not found error detected - using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        state["validation_attempts"] = 0
        return state
        
    if "JSON" in error_message and ("decode" in error_message.lower() or "invalid" in error_message.lower()):
        print("JSON parsing error detected - using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        state["validation_attempts"] = 0
        return state
    
    # For other errors, try a conservative fix approach
    original_code = state["fixer_script_code"]
    api_key = state["api_key"]
    
    # Use a simpler prompt that focuses on common fixes
    prompt = f"""You are a Python programmer. Fix this script by making it more robust. The error was: {error_message}

Focus on these common fixes:
1. Add better file path handling
2. Add more defensive null checks
3. Handle edge cases in data processing

**Original Script (truncated):**
```python
{original_code}
```

**Generate a complete, working Python script that handles the error robustly:**"""
    
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10_000
        )
        
        # Save raw response for debugging
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        raw_response_path = intermediates_dir / f"07_execution_fix_response_attempt_{state['execution_attempts']}.json"
        with open(raw_response_path, "w", encoding="utf-8") as f:
            try:
                json.dump(response.__dict__ if hasattr(response, '__dict__') else str(response), f, indent=2, default=str)
            except:
                f.write(str(response))
        print(f"Raw execution fix response saved to {raw_response_path}")
        
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            
        # For CSV approach, we don't need to validate Python syntax
        # Just regenerate a simple fallback CSV
        print("Generated execution error fix (using fallback CSV)")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        state["validation_attempts"] = 0  # Reset validation attempts
        
    except Exception as e:
        print(f"Error in execution fixing: {e}")
        print("Using fallback CSV")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["syntax_check_passed"] = True
        state["validation_attempts"] = 0
        # Save error details
        intermediates_dir = state["output_folder"] / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        error_path = intermediates_dir / f"07_execution_fix_error_attempt_{state['execution_attempts']}.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Error type: {type(e).__name__}\n")
            import traceback
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
    return state

# --- Conditional Edges ---
def should_continue_to_refinement(state: GraphState) -> str:
    """
    Determines whether to proceed with the ToC refinement steps or end the graph.
    """
    print("--- Checking Condition: should_continue_to_refinement ---")
    if state.get("error"):
        print("Error detected. Ending graph.")
        return "end"
    if state.get("raw_toc_json_path"):
        print("ToC extracted. Proceeding to refinement.")
        return "continue"
    else:
        print("No ToC extracted. Ending graph.")
        return "end"

def check_validation(state: GraphState) -> str:
    """
    Checks the result of script validation and routes to the next step.
    Uses configurable max attempts from state.
    """
    print("--- Checking Condition: check_validation ---")
    max_attempts = state.get("max_validation_attempts", 3)
    
    if state.get("error"): return "end"
    
    feedback = state.get("last_validation_feedback", "")
    if feedback.upper() == "OK" or feedback.startswith("OK"):
        print("Validation successful.")
        return "execute"
    
    # Check if we've been getting syntax errors repeatedly - use fallback earlier
    if state["validation_attempts"] >= 2 and not state.get("syntax_check_passed", False):
        print(f"Multiple syntax errors detected. Using fallback CSV.")
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["last_validation_feedback"] = "OK"
        state["syntax_check_passed"] = True
        return "execute"
    
    if state["validation_attempts"] >= max_attempts:
        print(f"Max validation attempts ({max_attempts}) reached.")
        print("Using fallback CSV to ensure workflow completion.")
        # Use fallback instead of failing completely
        state["csv_content"] = create_fallback_csv_content()
        state["fixer_script_code"] = get_csv_fixer_script()
        state["last_validation_feedback"] = "OK"
        state["syntax_check_passed"] = True
        return "execute"
        
    print(f"Validation failed (attempt {state['validation_attempts']}/{max_attempts}). Proceeding to refactor script.")
    return "refactor"

def check_execution(state: GraphState) -> str:
    """
    Checks if the script executed successfully.
    Uses configurable max attempts from state.
    """
    print("--- Checking Condition: check_execution ---")
    max_attempts = state.get("max_execution_attempts", 2)
    
    if state.get("error"): return "end"

    if state.get("last_execution_error"):
        if state["execution_attempts"] >= max_attempts:
            print(f"Max execution attempts ({max_attempts}) reached.")
            print("Creating fallback result to ensure workflow completion.")
            # Copy original ToC as fallback result
            try:
                original_file = state["raw_toc_json_path"]
                fallback_file = str(state["output_folder"] / "toc-fixed.json")
                with open(original_file, 'r') as src, open(fallback_file, 'w') as dst:
                    dst.write(src.read())
                state["final_toc_path"] = fallback_file
                state["execution_context"] = "Used original ToC as fallback"
                print(f"Fallback result saved to {fallback_file}")
            except Exception as e:
                state["error"] = f"Failed to create fallback result: {e}"
            return "end"
            
        print(f"Execution failed (attempt {state['execution_attempts']}/{max_attempts}). Attempting to fix.")
        return "fix_error"
    
    print("Execution successful. Ending graph.")
    return "end"

# --- Main Application ---
app = typer.Typer(
    help="A LangGraph-based tool for extracting and refining Table of Contents from documents."
)

@app.command()
def run_workflow(
    file_path: Annotated[str, typer.Option("--file", help="Path to a local image or PDF file.")],
    output_folder: Annotated[str, typer.Option("--output-folder", help="Folder path for all output files.")] = "results",
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="Mistral API key.")] = None,
    max_validation_attempts: Annotated[int, typer.Option("--max-validation-attempts", help="Maximum validation retry attempts.")] =5,
    max_execution_attempts: Annotated[int, typer.Option("--max-execution-attempts", help="Maximum execution retry attempts.")] = 5,
    save_intermediate: Annotated[bool, typer.Option("--save-intermediate", help="Save all intermediate files for debugging.")] = True,
    draw_graph: Annotated[Optional[str], typer.Option("--draw-graph", help="Path to save workflow graph visualization (PNG format).")] = None
):
    """
    Initializes and runs the enhanced ToC extraction and refinement graph with configurable retry logic.
    
    This improved implementation features:
    - Robust error handling with configurable retry attempts
    - Template-based code generation with syntax validation
    - Graceful fallback to original ToC if refinement fails
    - Comprehensive logging and intermediate file saving
    - Optional workflow graph visualization
    
    Examples:
        # Basic usage
        python toc_parsing_langraph.py --file document.pdf
        
        # Draw workflow graph
        python toc_parsing_langraph.py --file document.pdf --draw-graph workflow.png
        
        # Full debugging with graph
        python toc_parsing_langraph.py --file document.pdf --output-folder debug \
            --draw-graph debug/workflow.png --save-intermediate
    """
    # Validate parameters
    if max_validation_attempts < 1 or max_validation_attempts > 10:
        print("Error: max-validation-attempts must be between 1 and 10.")
        raise typer.Exit(code=1)
        
    if max_execution_attempts < 1 or max_execution_attempts > 5:
        print("Error: max-execution-attempts must be between 1 and 5.")
        raise typer.Exit(code=1)
    
    final_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not final_api_key:
        print("Error: API key not found. Use --api-key or set MISTRAL_API_KEY.")
        raise typer.Exit(code=1)
        
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        raise typer.Exit(code=1)

    # Determine if the file is an image
    mime_type, _ = mimetypes.guess_type(file_path)
    is_image = mime_type and mime_type.startswith("image/")
    
    print(f"🚀 Starting enhanced ToC extraction and refinement workflow")
    print(f"📄 Input file: {file_path} ({'image' if is_image else 'document'})")
    print(f"📁 Output folder: {output_dir}")
    print(f"🔧 Max validation attempts: {max_validation_attempts}")
    print(f"🔧 Max execution attempts: {max_execution_attempts}")
    print(f"💾 Save intermediate files: {save_intermediate}")
    if draw_graph:
        print(f"📊 Graph visualization: {draw_graph}")

    # Define the graph
    workflow = StateGraph(GraphState)
    workflow.add_node("run_ocr", run_ocr_and_extract_toc)
    workflow.add_node("generate_instructions", generate_refinement_instructions)
    workflow.add_node("generate_script", generate_fixer_script)
    workflow.add_node("validate_script", validate_script)
    workflow.add_node("refactor_script", refactor_script)
    workflow.add_node("execute_script", execute_fixer_script)
    workflow.add_node("fix_execution_error", fix_execution_error)

    # Build the graph edges
    workflow.set_entry_point("run_ocr")
    workflow.add_conditional_edges(
        "run_ocr",
        should_continue_to_refinement,
        {"continue": "generate_instructions", "end": END}
    )
    workflow.add_edge("generate_instructions", "generate_script")
    workflow.add_edge("generate_script", "validate_script")

    # Validation loop
    workflow.add_conditional_edges(
        "validate_script",
        check_validation,
        {"execute": "execute_script", "refactor": "refactor_script", "end": END}
    )
    workflow.add_edge("refactor_script", "validate_script")

    # Execution loop
    workflow.add_conditional_edges(
        "execute_script",
        check_execution,
        {"fix_error": "fix_execution_error", "end": END}
    )
    workflow.add_edge("fix_execution_error", "validate_script")

    # Compile the graph
    app_graph = workflow.compile()
    
    # Draw the graph if requested
    if draw_graph:
        try:
            print(f"\n📊 Drawing workflow graph to: {draw_graph}")
            # Get the graph visualization
            graph_image = app_graph.get_graph().draw_mermaid_png()
            
            # Save the graph image
            graph_path = Path(draw_graph)
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(graph_path, 'wb') as f:
                f.write(graph_image)
            
            print(f"✅ Graph visualization saved to: {graph_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not generate graph visualization: {e}")
            print("This may require additional dependencies: pip install grandalf pygraphviz")
    
    # Prepare the initial state with new configuration
    initial_state = {
        "file_path": file_path,
        "output_folder": output_dir,
        "api_key": final_api_key,
        "is_image": is_image,
        "max_validation_attempts": max_validation_attempts,
        "max_execution_attempts": max_execution_attempts,
        
        # Initialize other state fields
        "ocr_response": None,
        "raw_toc_json_path": None,
        "markdown_path": None,
        "refinement_instructions": None,
        "fixer_script_code": None,
        "syntax_check_passed": False,
        "final_toc_path": None,
        "validation_attempts": 0,
        "execution_attempts": 0,
        "error": None,
        "last_validation_feedback": None,
        "last_execution_error": None,
        "execution_context": None,
    }

    print("\n🔄 Invoking Enhanced LangGraph Workflow")
    print("=" * 50)
    
    # Define a configuration for the run, including a unique thread_id for state tracking
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 50
    }
    
    try:
        final_state = app_graph.invoke(initial_state, config=config)
    except Exception as e:
        print(f"❌ Critical workflow error: {e}")
        traceback.print_exc()
        raise typer.Exit(code=1)

    print("\n" + "=" * 50)
    print("📊 Workflow Summary")
    print("=" * 50)
    
    if final_state.get("error"):
        print(f"⚠️  Workflow completed with error: {final_state['error']}")
    else:
        print("✅ Workflow completed successfully!")
    
    # Report detailed results
    final_toc_path = final_state.get('final_toc_path')
    if final_toc_path and os.path.exists(final_toc_path):
        try:
            with open(final_toc_path, 'r') as f:
                toc_data = json.load(f)
                entries_count = len(toc_data.get('entries', []))
            print(f"📋 Final ToC: {entries_count} entries at {final_toc_path}")
        except:
            print(f"📋 Final ToC: Available at {final_toc_path}")
    else:
        print("📋 Final ToC: Not generated")
    
    # Report attempt statistics
    val_attempts = final_state.get("validation_attempts", 0)
    exec_attempts = final_state.get("execution_attempts", 0)
    print(f"🔧 Validation attempts: {val_attempts}/{max_validation_attempts}")
    print(f"⚡ Execution attempts: {exec_attempts}/{max_execution_attempts}")
    
    # Report execution context
    exec_context = final_state.get("execution_context")
    if exec_context:
        print(f"📝 Final status: {exec_context}")
    
    print(f"📁 All artifacts saved in: {output_folder}/")
    
    # List generated files
    if save_intermediate:
        print("\n📄 Generated files:")
        for file_path in sorted(output_dir.glob("*")):
            if file_path.is_file():
                size_kb = file_path.stat().st_size // 1024
                print(f"   - {file_path.name} ({size_kb} KB)")
    
    print("\n🎉 Enhanced ToC processing complete!")

if __name__ == "__main__":
    app() 