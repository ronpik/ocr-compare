"""
Pydantic models for Table of Contents structures.

These models define the schema for ToC entries and the overall ToC structure,
extracted from the original LangGraph script.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


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
    section_start_page: Optional[int] = Field(
        None,
        description=(
            "The actual document page number *where the content for this entry begins*. "
            "Use this to jump straight to the section's first page—for example, `chapter_start_page=42`."
        )
    )
    toc_list_page: Optional[int] = Field(
        None,
        description=(
            "The printed or digital page number *where this entry is listed* in the Table of Contents itself. "
            "Use this to jump to the ToC listing—for example, if your ToC spans pages v-vii, you might see "
            "`toc_list_page=vi`."
        )
    )


class TableOfContents(BaseModel):
    """Structured representation of a document's Table of Contents"""
    entries: List[TocEntry] = Field(..., description="List of all ToC entries.")