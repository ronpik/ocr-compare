"""
Table of Contents (ToC) Extraction and Refinement with LangGraph and Mistral AI

This script implements a multi-step workflow for extracting and refining a document's
Table of Contents using a stateful graph powered by LangGraph. It replicates the
functionality of the `mistral_ocr_refactored.py` script but in a more structured,
graph-based approach.

The workflow consists of the following steps (nodes):
1.  **Run OCR**: Extracts text and an initial, structured ToC from a document (PDF or image)
    using the Mistral OCR API.
2.  **Generate Refinement Instructions**: Uses a powerful vision or text model to analyze
    the initial ToC against the document's content and produce a list of human-readable
    fix instructions.
3.  **Generate Fixer Script**: Feeds these instructions to a code-generation model to
    create a Python script that programmatically applies the fixes.
4.  **Validation & Refinement Loop**: Validates that the generated script matches the
    instructions and attempts to fix it if it doesn't.
5.  **Execution & Fixing Loop**: Executes the script and attempts to fix any runtime
    errors that occur.

Usage:
    python scripts/toc_parsing_langraph.py --file ./path/to/document.pdf
    
    # Specify an output folder for all artifacts
    python scripts/toc_parsing_langraph.py --file ./path/to/image.png --output-folder ./results
"""
import os
import json
import base64
import mimetypes
import tempfile
import subprocess
import sys
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
    """Represents a single entry in the Table of Contents"""
    id: int = Field(..., description="The unique identifier for the entry.")
    type: str = Field(..., description="Entry type: 'section' or 'subsection'")
    text: str = Field(..., description="The title of the section.")
    parent: Optional[int] = Field(None, description="The id of its parent section.")
    toc_page_number: Optional[int] = Field(None, description="Page where entry is in the ToC.")
    start_page_number: Optional[int] = Field(None, description="Page where the section starts.")

class TableOfContents(BaseModel):
    """Structured representation of a document's Table of Contents"""
    entries: List[TocEntry] = Field(..., description="List of all ToC entries.")


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        file_path: Path to the local document to process.
        output_folder: Directory to save all artifacts.
        api_key: The Mistral API key.
        is_image: True if the input file is an image.
        ocr_response: The raw JSON response from the OCR API.
        raw_toc_json_path: Path to the initially extracted ToC file.
        markdown_path: Path to the extracted markdown content file.
        refinement_instructions: The natural language instructions for fixing the ToC.
        fixer_script_code: The generated Python code to apply the fixes.
        final_toc_path: Path to the final, corrected ToC file.
        error: A string to hold any error messages.
        validation_attempts: Counter for validation attempts.
        execution_attempts: Counter for execution attempts.
        last_validation_feedback: Feedback from validation.
        last_execution_error: Error message from execution.
    """
    file_path: str
    output_folder: Path
    api_key: str
    is_image: bool
    ocr_response: Optional[Dict[str, Any]]
    raw_toc_json_path: Optional[str]
    markdown_path: Optional[str]
    refinement_instructions: Optional[str]
    fixer_script_code: Optional[str]
    final_toc_path: Optional[str]
    error: Optional[str]
    validation_attempts: int
    execution_attempts: int
    last_validation_feedback: Optional[str]
    last_execution_error: Optional[str]


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

# --- Graph Nodes ---
def run_ocr_and_extract_toc(state: GraphState) -> GraphState:
    """
    Node 1: Runs the initial OCR process to extract text and a raw Table of Contents.
    """
    print("--- Starting Node: run_ocr_and_extract_toc ---")
    file_path = state["file_path"]
    api_key = state["api_key"]
    output_folder = state["output_folder"]

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

    except Exception as e:
        state["error"] = f"Error in OCR node: {e}"
        print(state["error"])

    return state

def generate_refinement_instructions(state: GraphState) -> GraphState:
    """
    Node 2: Generates natural language instructions to fix the raw ToC.
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
        
        messages = []
        if is_image:
            image_base64 = encode_file_to_base64(state["file_path"])
            prompt = f"""You are an expert document analyst. Compare the original image, the extracted ToC JSON, and markdown to generate explicit fix instructions for the ToC. Focus on page numbers, titles, and hierarchy.

**Extracted ToC JSON:**
{json.dumps(toc_data, indent=2)}

**Extracted Markdown:**
{markdown_content[:2000]}...

**Output Format:**
- modify the start_page_number of toc entry with id X to Y
- change the text field of toc entry with id X to "correct text"
... (and other instructions as needed)
"""
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_base64}
                ]
            })
            model = "pixtral-12b-latest"
        else: # PDF
            prompt = f"""You are an expert document analyst. Analyze the extracted markdown content and ToC JSON to generate explicit fix instructions. Focus on page numbers, titles, and hierarchy based on the full text.

**Extracted ToC JSON:**
{json.dumps(toc_data, indent=2)}

**Full Extracted Markdown:**
{markdown_content[:4000]}...

**Output Format:**
- modify the start_page_number of toc entry with id X to Y
- change the text field of toc entry with id X to "correct text"
... (and other instructions as needed)
"""
            messages.append({"role": "user", "content": prompt})
            model = "mistral-medium-2505"
            
        response = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2000
        )
        
        instructions = response.choices[0].message.content.strip()
        state["refinement_instructions"] = instructions
        
        instructions_path = state["output_folder"] / "toc-refinement-instructions.txt"
        with open(instructions_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        print(f"Refinement instructions saved to {instructions_path}")
        
    except Exception as e:
        state["error"] = f"Error in instruction generation node: {e}"
        print(state["error"])
        
    return state
    
def generate_fixer_script(state: GraphState) -> GraphState:
    """
    Node 3: Generates a Python script to apply the ToC fixes.
    """
    print("--- Starting Node: generate_fixer_script ---")
    if state.get("error"): return state
    
    instructions = state["refinement_instructions"]
    api_key = state["api_key"]
    
    if not instructions or instructions == "No changes needed":
        code = """
import json, sys
def apply_fixes(toc_data): return toc_data
if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f: data = json.load(f)
    with open(sys.argv[2], 'w') as f: json.dump(apply_fixes(data), f, indent=2)
"""
        state["fixer_script_code"] = code
        return state

    prompt = f"""Generate a Python script to apply these ToC fix instructions.
The script must contain a function `apply_fixes(toc_data)` and a main block to handle file I/O.

**Fix Instructions:**
{instructions}
"""
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        state["fixer_script_code"] = code
        
        script_path = state["output_folder"] / "toc-fix-script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Fixer script saved to {script_path}")

    except Exception as e:
        state["error"] = f"Error in script generation node: {e}"
        print(state["error"])
        
    return state

def validate_script(state: GraphState) -> GraphState:
    """
    Node: Validates that the generated script correctly implements the instructions.
    """
    print("--- Starting Node: validate_script ---")
    if state.get("error"): return state
    
    state["validation_attempts"] += 1
    instructions = state["refinement_instructions"]
    script_code = state["fixer_script_code"]
    api_key = state["api_key"]

    prompt = f"""You are a code reviewer. Your task is to validate if the Python script accurately implements the given instructions.
If the script is correct, respond with only the word "OK".
Otherwise, provide a brief, one-sentence explanation of the mismatch.

**Instructions:**
{instructions}

**Python Script to Validate:**
```python
{script_code}
```

**Your response (either "OK" or a one-sentence reason for failure):**
"""
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )
        feedback = response.choices[0].message.content.strip()
        state["last_validation_feedback"] = feedback
        print(f"Validation feedback: {feedback}")
    except Exception as e:
        state["error"] = f"Error in validation node: {e}"
        print(state["error"])
        
    return state

def refactor_script(state: GraphState) -> GraphState:
    """
    Node: Refactors the script based on validation feedback.
    """
    print("--- Starting Node: refactor_script ---")
    if state.get("error"): return state

    instructions = state["refinement_instructions"]
    original_code = state["fixer_script_code"]
    feedback = state["last_validation_feedback"]
    api_key = state["api_key"]

    prompt = f"""You are a Python programmer. Your task is to refactor the given script to correctly implement the instructions, based on the provided feedback.

**Original Instructions:**
{instructions}

**Original (incorrect) Python Script:**
```python
{original_code}
```

**Reason it failed validation:**
{feedback}

**Your Task:**
Rewrite the Python script to fix the issue and correctly implement all instructions.
"""
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        state["fixer_script_code"] = code
        print("Script has been refactored.")
    except Exception as e:
        state["error"] = f"Error in refactoring node: {e}"
        print(state["error"])
        
    return state

def execute_fixer_script(state: GraphState) -> GraphState:
    """
    Node: Executes the generated Python script to produce the final, fixed ToC.
    This node now handles errors and increments the execution attempt counter.
    """
    print("--- Starting Node: execute_fixer_script ---")
    if state.get("error"): return state

    state["execution_attempts"] += 1
    script_code = state["fixer_script_code"]
    original_toc_file = state["raw_toc_json_path"]
    fixed_toc_file = str(state["output_folder"] / "toc-fixed.json")
    
    # Reset previous execution error
    state["last_execution_error"] = None

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(script_code)
            temp_script_path = temp_script.name

        result = subprocess.run(
            [sys.executable, temp_script_path, original_toc_file, fixed_toc_file],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_script_path)

        if result.returncode != 0:
            error_message = f"Script execution failed with return code {result.returncode}:\n{result.stderr}"
            state["last_execution_error"] = error_message
            print(error_message)
        else:
            state["final_toc_path"] = fixed_toc_file
            print(f"Fixed ToC successfully generated at: {fixed_toc_file}")

    except Exception as e:
        error_message = f"An exception occurred during script execution: {e}"
        state["last_execution_error"] = error_message
        print(error_message)
    
    return state

def fix_execution_error(state: GraphState) -> GraphState:
    """
    Node: Attempts to fix a script that failed during execution.
    """
    print("--- Starting Node: fix_execution_error ---")
    if state.get("error"): return state

    original_code = state["fixer_script_code"]
    error_message = state["last_execution_error"]
    api_key = state["api_key"]
    
    prompt = f"""You are a Python programmer. Your task is to debug and fix the given script based on the runtime error it produced.

**Original Python Script:**
```python
{original_code}
```

**Runtime Error Message:**
{error_message}

**Your Task:**
Rewrite the Python script to fix the bug that caused the error. Ensure the script remains functionally correct according to its original purpose.
"""
    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="codestral-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        state["fixer_script_code"] = code
        # Reset validation attempts as we have new code to validate
        state["validation_attempts"] = 0
        print("Attempted to fix execution error.")
    except Exception as e:
        state["error"] = f"Error in execution fixing node: {e}"
        print(state["error"])
        
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
    """
    print("--- Checking Condition: check_validation ---")
    MAX_VALIDATION_ATTEMPTS = 3
    
    if state.get("error"): return "end"
    
    feedback = state.get("last_validation_feedback", "")
    if feedback.upper() == "OK":
        print("Validation successful.")
        return "execute"
    
    if state["validation_attempts"] >= MAX_VALIDATION_ATTEMPTS:
        print(f"Max validation attempts ({MAX_VALIDATION_ATTEMPTS}) reached. Ending graph.")
        state["error"] = "Failed to generate a valid script after multiple attempts."
        return "end"
        
    print("Validation failed. Proceeding to refactor script.")
    return "refactor"

def check_execution(state: GraphState) -> str:
    """
    Checks if the script executed successfully.
    """
    print("--- Checking Condition: check_execution ---")
    MAX_EXECUTION_ATTEMPTS = 2
    
    if state.get("error"): return "end"

    if state.get("last_execution_error"):
        if state["execution_attempts"] >= MAX_EXECUTION_ATTEMPTS:
            print(f"Max execution attempts ({MAX_EXECUTION_ATTEMPTS}) reached. Ending graph.")
            state["error"] = "Script failed to execute after multiple fixing attempts."
            return "end"
        print("Execution failed. Attempting to fix.")
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
    api_key: Annotated[Optional[str], typer.Option("--api-key", help="Mistral API key.")] = None
):
    """
    Initializes and runs the ToC extraction and refinement graph.
    """
    final_api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not final_api_key:
        print("Error: API key not found. Use --api-key or set MISTRAL_API_KEY.")
        raise typer.Exit(code=1)
        
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if the file is an image
    mime_type, _ = mimetypes.guess_type(file_path)
    is_image = mime_type and mime_type.startswith("image/")

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
    
    # Prepare the initial state
    initial_state = {
        "file_path": file_path,
        "output_folder": output_dir,
        "api_key": final_api_key,
        "is_image": is_image,
        "error": None,
        "validation_attempts": 0,
        "execution_attempts": 0,
    }

    print("\n--- Invoking LangGraph Workflow ---")
    final_state = app_graph.invoke(initial_state)

    print("\n--- Workflow Finished ---")
    if final_state.get("error"):
        print(f"Workflow finished with an error: {final_state['error']}")
    else:
        print("Workflow completed successfully.")
        print(f"Final corrected ToC is available at: {final_state.get('final_toc_path', 'N/A')}")
        print(f"All artifacts are in the '{output_folder}' directory.")

if __name__ == "__main__":
    app() 