import uuid
from typing import List, Optional, Union
from pathlib import Path
import fitz
from app.models.chunk import DocumentChunk

class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                new_separators = separators[i+1:]
                break
        
        splits = text.split(separator)
        good_splits: List[str] = []

        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    good_splits.extend(self._split_text(s, new_separators))
                else:
                    good_splits.append(s)
        
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + len(separator)
            if current_len + piece_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk).strip()
                if joined:
                    final_chunks.append(joined)
                
                overlap_text = joined[-self.chunk_overlap:] if len(joined) > self.chunk_overlap else joined
                current_chunk = [overlap_text, piece]
                current_len = len(overlap_text) + piece_len
            else:
                current_chunk.append(piece)
                current_len += piece_len
        
        if current_chunk:
            remaining = separator.join(current_chunk).strip()
            if remaining:
                final_chunks.append(remaining)
        
        return final_chunks
    
    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)

def parse_and_chunk_pdf(
    file_source: Union[str, Path, bytes],
    job_id: uuid.UUID,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[DocumentChunk]:
    if isinstance(file_source, bytes):
        doc = fitz.open(stream=file_source, filetype="pdf")
    else:
        doc = fitz.open(str(file_source))
    
    splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[DocumentChunk] = []
    global_chunk_index = 0

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()

            if not page_text:
                continue
            
            page_number = page_idx + 1
            page_chunks = splitter.split_text(page_text)

            for chunk_text in page_chunks:
                chunk = DocumentChunk(
                    id = uuid.uuid4(),
                    job_id = job_id,
                    chunk_index = global_chunk_index,
                    content = chunk_text,
                    chunk_metadata = {
                        "page_number": page_number,
                        "char_count": len(chunk_text),
                        "document_id": str(job_id)
                    },
                )
                chunks.append(chunk)
                global_chunk_index += 1
    finally:
        doc.close()

    return chunks