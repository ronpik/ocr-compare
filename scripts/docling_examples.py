from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs

# pipeline_options = VlmPipelineOptions(
#     vlm_options=vlm_model_specs.SMOLDOCLING_MLX,  # <-- change the model here
# )

# converter = DocumentConverter(
#     format_options={
#         InputFormat.IMAGE: ImageFormatOption(
#             pipeline_cls=StandardPdfPipeline,
#             # pipeline_options=pipeline_options,
#         ),
#     }
# )

converter = DocumentConverter(
    format_options={
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_cls=StandardPdfPipeline,
        ),
    }
)


path = "/Users/ronp/workspace/ocr-compare/examples/images/goren-11-4-toc-1.JPG"
doc = converter.convert(source=path).document

md = doc.export_to_markdown()
print(md)

with open('example-smol.md', 'w') as f:
    f.write(md)