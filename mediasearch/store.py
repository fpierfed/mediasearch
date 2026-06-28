import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from .config import EMBED_DIM


def _esc(value: str) -> str:
    """
    Escape single quotes for LanceDB SQL-like filter strings.

    NOTE: This only doubles single quotes. It is safe for the current
    codebase because all interpolated values are filesystem paths or
    validated enum values. If user-supplied strings are ever interpolated
    into filter expressions, a parameterised-query or more comprehensive
    escaping mechanism should be adopted.
    """
    # Reject control characters that have no legitimate reason to appear
    # in a filesystem path or a validated enum value — they signal that a
    # caller is interpolating unsanitised user input and needs a real
    # parameterised-query interface instead.
    for i, ch in enumerate(value):
        if ord(ch) < 0x20:
            raise ValueError(
                f'Unsafe character U+{ord(ch):04X} at position {i} in filter '
                f'value {value!r}.  Use a parameterised query for '
                f'user-supplied strings.'
            )
    return value.replace("'", "''")


def _normalize_scores(
    rows: list[dict], fts: bool = False, fts_k: float = 10.0
) -> None:
    """
    Mutate *rows* in-place, adding a ``"score"`` key to each dict.

    For vector-distance results, score = 1.0 - _distance (cosine similarity).
    For full-text-search results, score = raw / (raw + fts_k) — a bounded
    normalisation that maps [0, +inf) → [0, 1) with half-saturation at *fts_k*.
    """
    for r in rows:
        if fts:
            raw = float(r.get('_score', 0.0))
            r['score'] = raw / (raw + fts_k)
        else:
            r['score'] = 1.0 - float(r.get('_distance', 0.0))


class Store:
    def __init__(
        self,
        index_path: str | Path,
        dim: int = EMBED_DIM,
        text_dim: int = 768,
        fts_score_k: float = 10.0,
    ) -> None:
        """Initialise the store, creating tables if they do not exist."""
        Path(index_path).mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.text_dim = text_dim
        self.fts_score_k = fts_score_k
        self.db = lancedb.connect(str(index_path))
        self._ensure_tables()

    def _emb_schema(self) -> pa.Schema:
        """Return the schema for the embeddings table."""
        return pa.schema(
            [
                pa.field('id', pa.string()),
                pa.field('media_path', pa.string()),
                pa.field('media_type', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), self.dim)),
                pa.field('timestamp', pa.float32()),
                pa.field('frame_idx', pa.int32()),
            ]
        )

    def _files_schema(self) -> pa.Schema:
        """Return the schema for the files tracking table."""
        return pa.schema(
            [
                pa.field('path', pa.string()),
                pa.field('mtime', pa.float64()),
                pa.field('size', pa.int64()),
                pa.field('media_type', pa.string()),
                pa.field('status', pa.string()),
                pa.field('n_vectors', pa.int32()),
                pa.field('error_msg', pa.string()),
                pa.field('indexed_at', pa.float64()),
            ]
        )

    def _transcripts_schema(self) -> pa.Schema:
        """Return the schema for the transcripts table."""
        return pa.schema(
            [
                pa.field('id', pa.string()),
                pa.field('media_path', pa.string()),
                pa.field('media_type', pa.string()),
                pa.field('text', pa.string()),
                pa.field('vector', pa.list_(pa.float32(), self.text_dim)),
                pa.field('start_time', pa.float32()),
                pa.field('end_time', pa.float32()),
            ]
        )

    def _ensure_tables(self) -> None:
        """Ensure all required tables exist in the database."""
        names = self.db.list_tables()
        names = getattr(names, 'tables', names)
        if 'embeddings' not in names:
            self.db.create_table('embeddings', schema=self._emb_schema())
        if 'files' not in names:
            self.db.create_table('files', schema=self._files_schema())
        if 'transcripts' not in names:
            self.db.create_table(
                'transcripts', schema=self._transcripts_schema()
            )
            self.db.open_table('transcripts').create_fts_index(
                'text', replace=True
            )
        self.emb = self.db.open_table('embeddings')
        self.files = self.db.open_table('files')
        self.transcripts = self.db.open_table('transcripts')

    def manifest(self) -> dict:
        """Return a mapping of file paths to their tracking status rows."""
        rows = self.files.to_arrow().to_pylist()
        return {r['path']: r for r in rows}

    def set_file(
        self,
        *,
        path: str,
        mtime: float,
        size: int,
        media_type: str,
        status: str,
        n_vectors: int = 0,
        error_msg: str | None = None,
    ) -> None:
        """Upsert a file's status in the tracking table."""
        self.files.delete(f"path = '{_esc(path)}'")
        self.files.add(
            [
                {
                    'path': path,
                    'mtime': float(mtime),
                    'size': int(size),
                    'media_type': media_type,
                    'status': status,
                    'n_vectors': int(n_vectors),
                    'error_msg': error_msg or '',
                    'indexed_at': time.time(),
                }
            ]
        )

    def errors(self) -> list[dict]:
        """
        Return a list of file rows that encountered errors during indexing.
        """
        rows = self.files.to_arrow().to_pylist()
        return [r for r in rows if r['status'] == 'error']

    def add_embeddings(self, rows: list[dict]) -> None:
        """Add rows to the visual embeddings table."""
        if rows:
            self.emb.add(rows)

    def add_transcripts(self, rows: list[dict]) -> None:
        """Add rows to the audio transcripts table."""
        if rows:
            self.transcripts.add(rows)

    def delete_file(self, path: str) -> None:
        """Remove all data associated with a file from all tables."""
        esc = _esc(path)
        self.emb.delete(f"media_path = '{esc}'")
        self.transcripts.delete(f"media_path = '{esc}'")
        self.files.delete(f"path = '{esc}'")

    def count_vectors(self, media_path: str) -> int:
        """Return the number of visual embeddings stored for a given file."""
        return int(
            self.emb.count_rows(filter=f"media_path = '{_esc(media_path)}'")
        )

    def search(
        self,
        vector: list[float] | np.ndarray,
        top_k: int,
        media_type: str | None = None,
    ) -> list[dict]:
        """Search the visual embeddings table for the closest vectors."""
        q = (
            self.emb.search(np.asarray(vector, dtype=np.float32))
            .metric('cosine')
            .limit(top_k)
        )
        if media_type:
            q = q.where(f"media_type = '{_esc(media_type)}'")
        results = q.to_list()
        _normalize_scores(results)
        return results

    def search_transcripts_vector(
        self, vector: list[float] | np.ndarray, top_k: int
    ) -> list[dict]:
        """Search the transcripts table using semantic vector similarity."""
        q = (
            self.transcripts.search(np.asarray(vector, dtype=np.float32))
            .metric('cosine')
            .limit(top_k)
        )
        results = q.to_list()
        _normalize_scores(results)
        return results

    def search_transcripts_fts(self, query: str, top_k: int) -> list[dict]:
        """Search the transcripts table using full-text search (BM25)."""
        # LanceDB full-text search directly accepts the string query and uses
        # the FTS index
        q = self.transcripts.search(query).limit(top_k)
        results = q.to_list()
        _normalize_scores(results, fts=True, fts_k=self.fts_score_k)
        return results

    def index_dim(self) -> int:
        """
        The visual-embedding dimension the on-disk embeddings table was created
        with.
        """
        return self.emb.schema.field('vector').type.list_size

    def text_index_dim(self) -> int:
        """
        The text-embedding dimension the on-disk transcripts table was created
        with.
        """
        return self.transcripts.schema.field('vector').type.list_size

    def reset(self) -> None:
        """
        Drop and recreate all three tables empty, using the current dimensions.
        """
        names = self.db.list_tables()
        names = getattr(names, 'tables', names)
        for name in ('embeddings', 'files', 'transcripts'):
            if name in names:
                self.db.drop_table(name)
        self._ensure_tables()

    def stats(self) -> dict:
        """Return summary statistics about the indexed files and vectors."""
        import pyarrow.compute as pc

        t = self.files.to_arrow()
        statuses = t.column('status')
        return {
            'files': len(t),
            'vectors': self.emb.count_rows(),
            'done': pc.sum(pc.equal(statuses, 'done')).as_py(),
            'pending': pc.sum(pc.equal(statuses, 'pending')).as_py(),
            'error': pc.sum(pc.equal(statuses, 'error')).as_py(),
        }
