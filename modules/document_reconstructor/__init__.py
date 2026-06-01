"""Document reconstruction module.

Generates a text-only PDF from indexed chunks when the source PDF is missing.
Used as a fallback by the ``/api/document`` endpoint so that broken file
references degrade gracefully into a readable artifact.
"""

from modules.document_reconstructor.service import DocumentReconstructor

__all__ = ["DocumentReconstructor"]
