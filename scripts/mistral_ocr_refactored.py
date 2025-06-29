"""
Mistral OCR API Client - Refactored with Official Python Client

A command-line tool for using Mistral's OCR API to extract text from documents and images.
Supports both URL and local file inputs with customizable extraction parameters.
Now includes Table of Contents extraction using structured annotations.

Usage Examples:
    # Process a document from URL
    python mistral_ocr_refactored.py --url https://example.com/document.pdf

    # Process a local file
    python mistral_ocr_refactored.py --file ./path/to/image.png

    # Process specific pages with custom model
    python mistral_ocr_refactored.py --file document.pdf --pages "0,2,4" --model mistral-ocr-latest

    # Include base64 images in response
    python mistral_ocr_refactored.py --url https://example.com/doc.pdf --include-images

    # Save markdown content to a custom file
    python mistral_ocr_refactored.py --file document.pdf --markdown extracted_text.md
    
    # Extract Table of Contents
    python mistral_ocr_refactored.py --file document.pdf --extract-toc --toc-output toc.json
    
    # Extract and refine Table of Contents using AI vision and code generation
    python mistral_ocr_refactored.py --file document.pdf --extract-toc --refine-toc --original-image document_page1.png
"""

import os
import json
import base64
import mimetypes
from typing import Dict, Any, List, Optional

import typer
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models import DocumentURLChunk, ImageURLChunk

# Load API key from environment variable for security
# You can set it with: export MISTRAL_API_KEY="your-api-key"
API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# Available OCR models
DEFAULT_MODEL = "mistral-ocr-latest"

app = typer.Typer(
    help="Mistral OCR CLI tool for document processing with AI-powered text extraction and ToC extraction."
)


# Pydantic models for Table of Contents extraction
class TocEntry(BaseModel):
    """Represents a single entry in the Table of Contents"""

    id: int = Field(
        ...,
        description="The unique identifier for the entry, the index number of the entry in the Table of Contents"
    )
    type: str = Field(
        ..., 
        description="Type of entry: 'section' for main sections or 'subsection' for sub-sections"
    )
    text: str = Field(
        ..., 
        description="The name/title of the section as it appears in the Table of Contents"
    )
    parent: Optional[int] = Field(
        None, 
        description="For subsections, the id of the parent section it belongs to. For sections, this should be null"
    )
    page_number: Optional[int] = Field(
        None, 
        description="The page number where this section starts, if available in the ToC"
    )


class TableOfContents(BaseModel):
    """Structured representation of a document's Table of Contents"""
    entries: List[TocEntry] = Field(
        ..., 
        description="List of all section and subsection entries found in the Table of Contents"
    )


def encode_file_to_base64(file_path: str) -> str:
    """
    Encodes a file to a base64 data URI.

    Args:
        file_path: The path to the file.

    Returns:
        The base64 encoded data URI.

    Raises:
        FileNotFoundError: If the file is not found.
    """
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


def run_mistral_ocr(
    document_input: str,
    api_key: str,
    is_base64: bool = False,
    is_image: bool = False,
    pages: Optional[List[int]] = None,
    include_image_base64: bool = False,
    model: str = DEFAULT_MODEL,
    extract_toc: bool = False,
) -> Dict[str, Any]:
    """
    Sends a request to the Mistral OCR API using the official Python client.

    Args:
        document_input: The URL or base64 data URI of the document to process.
        api_key: Your Mistral API key.
        is_base64: Whether the document_input is a base64 data URI.
        is_image: Whether the input is an image (not a PDF).
        pages: Specific pages to process (0-indexed). If None, processes all pages.
        include_image_base64: Whether to include base64 encoded images in the response.
        model: The model to use for OCR.
        extract_toc: Whether to extract Table of Contents using annotations.

    Returns:
        The response from the API as a dictionary.

    Raises:
        ValueError: If the API key is not provided.
        Exception: For API-related errors.
    """
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set. Please provide a valid API key.")

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    # Build the document object based on input type
    if is_image or (is_base64 and any(document_input.startswith(prefix) for prefix in ["data:image/", "data:application/octet-stream"])):
        # For images, use ImageURLChunk
        document = ImageURLChunk(image_url=document_input)
    else:
        # For PDFs and other documents, use DocumentURLChunk
        if is_base64:
            document = DocumentURLChunk(document_url=document_input)
        else:
            document = DocumentURLChunk(
                document_url=document_input,
                document_name=os.path.basename(document_input) if document_input else "document"
            )

    # Prepare the arguments for the OCR process
    ocr_args = {
        "model": model,
        "document": document,
        "include_image_base64": include_image_base64,
    }

    # Add optional parameters
    if pages is not None:
        ocr_args["pages"] = pages

    # Add annotation format if ToC extraction is requested
    if extract_toc:
        ocr_args["document_annotation_format"] = response_format_from_pydantic_model(TableOfContents)

    try:
        print(f"Sending request to Mistral OCR API using model: {model}...")
        response = client.ocr.process(**ocr_args)
        print("Request successful.")
        
        # Convert response to dict if it's not already
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        else:
            return dict(response)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


def extract_toc_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Table of Contents data from the OCR response.
    
    Args:
        response: The OCR API response
        
    Returns:
        List of ToC entries extracted from bbox annotations
    """
    
    if "document_annotation" not in response:
        return []
    
    try:
        annotation_data = json.loads(response["document_annotation"])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse ToC annotation: {e}")
        return []

    if "entries" in annotation_data:
        return annotation_data["entries"]


def extract_image_annotations_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Table of Contents data from the OCR response.
    
    Args:
        response: The OCR API response
        
    Returns:
        List of ToC entries extracted from bbox annotations
    """
    toc_entries = []
    
    if "pages" in response:
        for page in response["pages"]:
            if "images" in page:
                for image in page["images"]:
                    if "image_annotation" in image:
                        try:
                            # Parse the annotation as TableOfContents
                            annotation_data = json.loads(image["image_annotation"])
                            if "entries" in annotation_data:
                                toc_entries.extend(annotation_data["entries"])
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Failed to parse ToC annotation: {e}")
                            
    return toc_entries


def refine_toc_step1_reasoning(
    markdown_file: str,
    toc_file: str,
    image_file: str,
    api_key: str,
    reasoning_model: str = "pixtral-12b-latest"
) -> str:
    """
    Step 1: Use Mistral's vision model to analyze OCR output and generate fix instructions.
    
    Args:
        markdown_file: Path to the markdown file with OCR content
        toc_file: Path to the ToC JSON file
        image_file: Path to the original image file
        api_key: Mistral API key
        reasoning_model: Vision model to use for analysis
        
    Returns:
        String containing explicit fix instructions
    """
    # Read input files
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        with open(toc_file, 'r', encoding='utf-8') as f:
            toc_data = json.load(f)
        
        # Encode image for the model
        image_base64 = encode_file_to_base64(image_file)
        
    except FileNotFoundError as e:
        raise ValueError(f"Required file not found: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ToC file: {e}")
    
    # Create the reasoning prompt
    prompt = f"""You are an expert document analyst. Your task is to carefully compare the original document image with the extracted Table of Contents (ToC) and markdown content, then generate explicit fix instructions.

**INPUTS:**
1. Original document image (attached)
2. Extracted ToC JSON: {json.dumps(toc_data, indent=2)}
3. Extracted markdown content: {markdown_content[:2000]}...

**YOUR TASK:**
Compare the visual Table of Contents in the original image with the extracted ToC JSON. Look for:
- Incorrect page numbers
- Misspelled or incorrect section names
- Wrong section hierarchy (parent-child relationships)
- Missing or extra entries
- Incorrect entry types (section vs subsection)

**OUTPUT FORMAT:**
Generate explicit fix instructions in this exact format:
- modify the page number of toc entry with id X to Y
- change the text field of toc entry with id X to "correct text"
- change the parent field of toc entry with id X to Y (or null for top-level sections)
- change the type field of toc entry with id X to "section" (or "subsection")
- remove toc entry with id X
- add new toc entry: type="section", text="New Section", parent=null, page_number=10

**IMPORTANT:**
- Reference entries by their id field from the JSON
- Be precise with page numbers, text, and parent relationships
- Only suggest changes that you can clearly see are needed
- If no changes are needed, respond with "No changes needed"
"""

    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    try:
        # Make the chat completion request
        response = client.chat.complete(
            model=reasoning_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": image_base64
                        }
                    ]
                }
            ],
            temperature=0.1,  # Low temperature for more consistent analysis
            max_tokens=2000
        )
        
        # Extract the instructions from the response
        if hasattr(response, 'choices') and response.choices:
            instructions = response.choices[0].message.content
            return instructions.strip()
        else:
            raise ValueError("No response from reasoning model")
            
    except Exception as e:
        raise ValueError(f"Error calling reasoning model: {e}")


def refine_toc_step2_codegen(
    instructions: str,
    api_key: str,
    codegen_model: str = "codestral-latest"
) -> str:
    """
    Step 2: Use Mistral's code generation to create Python script that applies fix instructions.
    
    Args:
        instructions: Fix instructions from Step 1
        api_key: Mistral API key
        codegen_model: Code generation model to use
        
    Returns:
        Python script as a string
    """
    if not instructions.strip() or instructions.strip() == "No changes needed":
        # Return a simple script that doesn't make changes
        return """
import json

def apply_fixes(toc_data):
    # No changes needed
    return toc_data

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        toc_data = json.load(f)
    
    fixed_toc = apply_fixes(toc_data)
    
    with open(sys.argv[2], 'w') as f:
        json.dump(fixed_toc, f, indent=2)
"""
    
    # Create the code generation prompt
    prompt = f"""You are a Python code generator. Generate a Python script that applies the given ToC fix instructions.

**FIX INSTRUCTIONS:**
{instructions}

**REQUIREMENTS:**
1. Create a function called `apply_fixes(toc_data)` that takes a dictionary and returns the modified dictionary
2. The function should apply all the fix instructions to the toc_data
3. The toc_data has this structure: {{"entries": [list of toc entries]}}
4. Each entry has fields: id, type, text, parent, page_number
5. Use hard-coded values based on the instructions
6. Include a main section that reads input file, applies fixes, and writes output file

**EXAMPLE OUTPUT STRUCTURE:**
```python
import json

def apply_fixes(toc_data):
    entries = toc_data.get("entries", [])
    
    # Apply fixes based on instructions
    for entry in entries:
        if entry.get("id") == 1:
            entry["page_number"] = 5
        elif entry.get("id") == 2:
            entry["text"] = "Corrected Title"
    
    # Remove entries if needed
    entries = [e for e in entries if e.get("id") not in [3, 4]]
    
    # Add new entries if needed
    new_entry = {{"id": 99, "type": "section", "text": "New Section", "parent": None, "page_number": 10}}
    entries.append(new_entry)
    
    return {{"entries": entries}}

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        toc_data = json.load(f)
    
    fixed_toc = apply_fixes(toc_data)
    
    with open(sys.argv[2], 'w') as f:
        json.dump(fixed_toc, f, indent=2)
```

**IMPORTANT:**
- Use only standard library imports (json, sys)
- Handle the case where entries might not exist
- Generate robust code that won't crash on missing fields
- Make the script executable with command line arguments
"""

    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    try:
        # Make the chat completion request to codegen model
        response = client.chat.complete(
            model=codegen_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent code generation
            max_tokens=2000
        )
        
        # Extract the generated code
        if hasattr(response, 'choices') and response.choices:
            code = response.choices[0].message.content
            
            # Extract code from markdown code blocks if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
        else:
            raise ValueError("No response from code generation model")
            
    except Exception as e:
        raise ValueError(f"Error calling code generation model: {e}")


def refine_toc_step3_execute(
    generated_script: str,
    original_toc_file: str,
    fixed_toc_file: str = "toc-fixed.json"
) -> str:
    """
    Step 3: Execute the generated Python script to produce the fixed ToC file.
    
    Args:
        generated_script: Python script from Step 2
        original_toc_file: Path to the original ToC JSON file
        fixed_toc_file: Path for the output fixed ToC file
        
    Returns:
        Path to the fixed ToC file
    """
    import tempfile
    import subprocess
    import sys
    
    # Create a temporary file for the script
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(generated_script)
            temp_script_path = temp_script.name
        
        # Execute the script with proper arguments
        result = subprocess.run(
            [sys.executable, temp_script_path, original_toc_file, fixed_toc_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for safety
        )
        
        if result.returncode != 0:
            raise ValueError(f"Script execution failed: {result.stderr}")
        
        # Verify the output file was created and is valid JSON
        try:
            with open(fixed_toc_file, 'r') as f:
                json.load(f)  # This will raise an exception if JSON is invalid
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Generated ToC file is invalid: {e}")
        
        return fixed_toc_file
        
    except subprocess.TimeoutExpired:
        raise ValueError("Script execution timed out - possible infinite loop or hanging")
    except Exception as e:
        raise ValueError(f"Error executing script: {e}")
    finally:
        # Clean up temporary file
        try:
            import os
            os.unlink(temp_script_path)
        except:
            pass  # Ignore cleanup errors


def save_toc_as_markdown(toc_entries: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save Table of Contents entries as a markdown file.
    
    Args:
        toc_entries: List of ToC entries
        output_file: Path to save the markdown file
    """
    markdown_lines = ["# Table of Contents\n\n"]
    
    for entry in toc_entries:
        entry_type = entry.get("type", "section")
        text = entry.get("text", "")
        page = entry.get("page_number")
        
        # Format based on type
        if entry_type == "section":
            line = f"## {text}"
        else:  # subsection
            line = f"   - {text}"
            
        # Add page number if available
        if page is not None:
            line += f" (p. {page})"
            
        markdown_lines.append(line + "\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


@app.command()
def main(
    document_url: Annotated[
        Optional[str],
        typer.Option(
            "--url",
            help="URL of the document (e.g., PDF, PNG, JPEG) to process."
        ),
    ] = None,
    file_path: Annotated[
        Optional[str],
        typer.Option(
            "--file",
            help="Path to a local image or PDF file to process."
        ),
    ] = None,
    output_file: Annotated[
        str, typer.Option(help="Path to save the JSON output.")
    ] = "output.json",
    markdown_file: Annotated[
        Optional[str],
        typer.Option(
            "--markdown",
            help="Path to save the extracted markdown content. Default: output.md"
        ),
    ] = None,
    extract_toc: Annotated[
        bool,
        typer.Option(
            "--extract-toc",
            help="Extract Table of Contents using structured annotations"
        ),
    ] = False,
    toc_output: Annotated[
        str,
        typer.Option(
            "--toc-output",
            help="Path to save the extracted Table of Contents (JSON format)"
        ),
    ] = "toc.json",
    toc_markdown: Annotated[
        Optional[str],
        typer.Option(
            "--toc-markdown",
            help="Path to save the Table of Contents as markdown"
        ),
    ] = None,
    pages: Annotated[
        Optional[str],
        typer.Option(
            "--pages",
            help="Comma-separated list of page numbers to process (0-indexed). Example: '0,1,2'"
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"Model to use for OCR. Default: {DEFAULT_MODEL}"
        ),
    ] = DEFAULT_MODEL,
    include_images: Annotated[
        bool,
        typer.Option(
            "--include-images",
            help="Include base64 encoded images in the response."
        ),
    ] = False,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="Mistral API key. Can also be set via MISTRAL_API_KEY env var."
        ),
    ] = None,
    refine_toc: Annotated[
        bool,
        typer.Option(
            "--refine-toc",
            help="Enable three-step ToC refinement process using reasoning and code generation"
        ),
    ] = False,
    reasoning_model: Annotated[
        str,
        typer.Option(
            "--reasoning-model",
            help="Vision model for ToC analysis (Step 1)"
        ),
    ] = "pixtral-12b-latest",
    codegen_model: Annotated[
        str,
        typer.Option(
            "--codegen-model",
            help="Code generation model for fix script creation (Step 2)"
        ),
    ] = "codestral-latest",
    fixed_toc_output: Annotated[
        str,
        typer.Option(
            "--fixed-toc-output",
            help="Path to save the refined/fixed Table of Contents"
        ),
    ] = "toc-fixed.json",
    original_image: Annotated[
        Optional[str],
        typer.Option(
            "--original-image",
            help="Path to original image file for ToC refinement (required with --refine-toc)"
        ),
    ] = None,
) -> None:
    """Main function to run the Mistral OCR with optional ToC extraction and refinement."""
    if not document_url and not file_path:
        print("Error: Please provide either a --url or a --file.")
        raise typer.Exit(code=1)

    if document_url and file_path:
        print("Error: Please provide either --url or --file, not both.")
        raise typer.Exit(code=1)

    # Use provided API key or fallback to environment variable
    final_api_key = api_key or API_KEY

    if not final_api_key:
        print("Error: No API key provided. "
              "Set MISTRAL_API_KEY environment variable or use --api-key option.")
        raise typer.Exit(code=1)

    # Validate refinement options
    if refine_toc and not extract_toc:
        print("Error: --refine-toc requires --extract-toc to be enabled.")
        raise typer.Exit(code=1)
    
    if refine_toc and not original_image:
        print("Error: --refine-toc requires --original-image to be specified.")
        raise typer.Exit(code=1)

    # Parse pages if provided
    pages_list = None
    if pages:
        try:
            pages_list = [int(p.strip()) for p in pages.split(",")]
        except ValueError:
            print("Error: Invalid pages format. Use comma-separated numbers, e.g., '0,1,2'")
            raise typer.Exit(code=1)

    document_to_process = ""
    is_base64_input = False
    is_image_input = False

    if file_path:
        print(f"Processing local file: {file_path}")
        try:
            # Check if it's an image file
            mime_type, _ = mimetypes.guess_type(file_path)
            is_image_input = mime_type and mime_type.startswith("image/")
            
            document_to_process = encode_file_to_base64(file_path)
            is_base64_input = True
        except FileNotFoundError:
            raise typer.Exit(code=1)

    if document_url:
        document_to_process = document_url
        is_base64_input = False
        # Check if URL points to an image
        is_image_input = any(document_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])
        print(f"Processing document from URL: {document_to_process}")

    try:
        result = run_mistral_ocr(
            document_to_process,
            final_api_key,
            is_base64=is_base64_input,
            is_image=is_image_input,
            pages=pages_list,
            include_image_base64=include_images,
            model=model,
            extract_toc=extract_toc,
        )
        
        # Save main OCR result
        print(f"\nOCR Result successfully generated. Writing to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print("Done.")

        # Extract and save markdown content
        if "pages" in result and len(result["pages"]) > 0:
            # Determine markdown output file path
            md_output_file = markdown_file if markdown_file else "output.md"

            # Extract markdown content from all pages
            markdown_content = []
            for page in result["pages"]:
                if "markdown" in page:
                    page_idx = page.get("index", "?")
                    # Add page separator
                    if markdown_content:  # Not the first page
                        markdown_content.append("\n\n---\n\n")
                    markdown_content.append(f"<!-- Page {page_idx + 1} -->\n\n")
                    markdown_content.append(page["markdown"])

            if markdown_content:
                print(f"\nExtracting markdown content to {md_output_file}...")
                with open(md_output_file, "w", encoding="utf-8") as f:
                    f.write("".join(markdown_content))
                print(f"Markdown content saved to {md_output_file}")

        # Extract and save Table of Contents if requested
        if extract_toc:
            print("\nExtracting Table of Contents...")
            toc_entries = extract_toc_from_response(result)
            
            if toc_entries:
                # Save ToC as JSON
                print(f"Saving Table of Contents to {toc_output}...")
                with open(toc_output, "w", encoding="utf-8") as f:
                    json.dump({"entries": toc_entries}, f, indent=2)
                print(f"Table of Contents saved to {toc_output}")
                
                # Save ToC as markdown if requested
                if toc_markdown:
                    print(f"Saving Table of Contents as markdown to {toc_markdown}...")
                    save_toc_as_markdown(toc_entries, toc_markdown)
                    print(f"Table of Contents markdown saved to {toc_markdown}")
                
                print(f"\nFound {len(toc_entries)} ToC entries")
            else:
                print("No Table of Contents entries found in the document")
        
        # Step 4: Run ToC refinement process if requested
        if refine_toc and extract_toc and toc_entries:
            print("\n=== Starting ToC Refinement Process ===")
            
            # Determine markdown output file path
            md_output_file = markdown_file if markdown_file else "output.md"
            
            try:
                # Step 1: Vision analysis
                print("Step 1: Analyzing ToC with vision model...")
                instructions = refine_toc_step1_reasoning(
                    markdown_file=md_output_file,
                    toc_file=toc_output,
                    image_file=original_image,
                    api_key=final_api_key,
                    reasoning_model=reasoning_model
                )
                print(f"Generated instructions: {instructions[:200]}...")
                
                # Step 2: Code generation
                print("Step 2: Generating fix script...")
                generated_script = refine_toc_step2_codegen(
                    instructions=instructions,
                    api_key=final_api_key,
                    codegen_model=codegen_model
                )
                print("Fix script generated successfully")
                
                # Step 3: Script execution
                print("Step 3: Executing fix script...")
                fixed_toc_path = refine_toc_step3_execute(
                    generated_script=generated_script,
                    original_toc_file=toc_output,
                    fixed_toc_file=fixed_toc_output
                )
                print(f"Fixed ToC saved to: {fixed_toc_path}")
                
                # Load and display summary of changes
                try:
                    with open(fixed_toc_path, 'r') as f:
                        fixed_toc_data = json.load(f)
                    fixed_entries = fixed_toc_data.get("entries", [])
                    print(f"Original ToC had {len(toc_entries)} entries")
                    print(f"Fixed ToC has {len(fixed_entries)} entries")
                except:
                    pass  # Don't fail if we can't show summary
                
                print("=== ToC Refinement Process Completed ===")
                
            except Exception as e:
                print(f"Error during ToC refinement: {e}")
                print("Continuing with original ToC results...")
        
        elif refine_toc and extract_toc and not toc_entries:
            print("Skipping ToC refinement: No ToC entries were extracted")
        elif refine_toc and not extract_toc:
            print("Skipping ToC refinement: ToC extraction was not enabled")

        # Print a summary of the results
        if "pages" in result:
            print(f"\nProcessed {len(result['pages'])} pages.")
            if "usage_info" in result:
                usage = result["usage_info"]
                print(f"Pages processed: {usage.get('pages_processed', 'N/A')}")
                print(f"Document size: {usage.get('doc_size_bytes', 'N/A')} bytes")
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()