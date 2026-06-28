import numpy as np
import pytest

from mediasearch.store import Store, _esc


def _row(media_path, vec, media_type='image', ts=0.0, idx=0):
    """Helper function to create a mock database row dictionary."""
    return {
        'id': f'{media_path}:{idx}',
        'media_path': media_path,
        'media_type': media_type,
        'vector': list(np.asarray(vec, dtype=np.float32)),
        'timestamp': ts,
        'frame_idx': idx,
    }


def _unit(seed, dim=1152):
    """Helper function to create a deterministic random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_set_and_read_manifest(tmp_path):
    """Test that writing a file status allows retrieving it from the manifest."""
    s = Store(tmp_path / 'idx')
    s.set_file(
        path='/a.png',
        mtime=1.0,
        size=10,
        media_type='image',
        status='done',
        n_vectors=1,
    )
    m = s.manifest()
    assert m['/a.png']['status'] == 'done'
    assert m['/a.png']['n_vectors'] == 1


def test_set_file_upserts(tmp_path):
    """Test that setting a file status updates the existing entry."""
    s = Store(tmp_path / 'idx')
    s.set_file(
        path='/a.png', mtime=1.0, size=10, media_type='image', status='pending'
    )
    s.set_file(
        path='/a.png',
        mtime=1.0,
        size=10,
        media_type='image',
        status='done',
        n_vectors=2,
    )
    m = s.manifest()
    assert len([k for k in m if k == '/a.png']) == 1
    assert m['/a.png']['status'] == 'done'


def test_add_search_and_group(tmp_path):
    """Test that adding vectors allows retrieving them correctly via search."""
    s = Store(tmp_path / 'idx')
    va, vb = _unit(1), _unit(2)
    s.add_embeddings([_row('/a.png', va), _row('/b.png', vb)])
    hits = s.search(va, top_k=5)
    assert hits[0]['media_path'] == '/a.png'  # nearest to itself
    assert hits[0]['score'] > hits[1]['score']
    assert 0.99 <= hits[0]['score'] <= 1.01


def test_delete_file_removes_vectors_and_manifest(tmp_path):
    """Test that deleting a file drops its vectors and manifest entry."""
    s = Store(tmp_path / 'idx')
    s.add_embeddings([_row('/a.png', _unit(1))])
    s.set_file(
        path='/a.png',
        mtime=1.0,
        size=10,
        media_type='image',
        status='done',
        n_vectors=1,
    )
    s.delete_file('/a.png')
    assert s.count_vectors('/a.png') == 0
    assert '/a.png' not in s.manifest()


def test_search_media_type_filter(tmp_path):
    """Test that search properly filters results by media type."""
    s = Store(tmp_path / 'idx')
    s.add_embeddings([_row('/a.png', _unit(1), media_type='image')])
    s.add_embeddings(
        [_row('/v.mp4', _unit(1), media_type='video', ts=2.0, idx=1)]
    )
    hits = s.search(_unit(1), top_k=5, media_type='video')
    assert all(h['media_type'] == 'video' for h in hits)
    assert hits[0]['media_path'] == '/v.mp4'


def test_stats(tmp_path):
    """Test that stats returns accurate counts of files and vectors."""
    s = Store(tmp_path / 'idx')
    s.set_file(
        path='/a.png',
        mtime=1.0,
        size=1,
        media_type='image',
        status='done',
        n_vectors=1,
    )
    s.set_file(
        path='/b.mp4',
        mtime=1.0,
        size=1,
        media_type='video',
        status='error',
        error_msg='boom',
    )
    s.add_embeddings([_row('/a.png', _unit(1))])
    st = s.stats()
    assert (
        st['files'] == 2
        and st['done'] == 1
        and st['error'] == 1
        and st['vectors'] == 1
    )
    assert s.errors()[0]['path'] == '/b.mp4'


def test_reopen_existing_index_does_not_recreate(tmp_path):
    """Test that opening an existing index does not attempt to recreate its tables."""
    # Regression: lancedb list_tables() returns a response object, not a list;
    # reopening an existing index must NOT try to recreate tables.
    path = tmp_path / 'idx'
    s1 = Store(path)
    s1.set_file(
        path='/a.png',
        mtime=1.0,
        size=10,
        media_type='image',
        status='done',
        n_vectors=1,
    )
    s1.add_embeddings([_row('/a.png', _unit(1))])

    s2 = Store(path)  # reopen same path in-process — must not raise
    assert '/a.png' in s2.manifest()
    assert s2.stats()['vectors'] == 1
    assert s2.count_vectors('/a.png') == 1


def test_index_dim_reports_schema(tmp_path):
    """Test that index_dim matches the stored schema's dimension."""
    assert Store(tmp_path / 'idx', dim=768).index_dim() == 768
    assert Store(tmp_path / 'idx2').index_dim() == 1152  # default EMBED_DIM


def test_reset_recreates_tables_at_new_dim(tmp_path):
    """Test that reset wipes the database and recreates tables with updated dimensions."""
    path = tmp_path / 'idx'
    s1 = Store(path, dim=1152)
    s1.add_embeddings([_row('/a.png', _unit(1, dim=1152))])
    assert s1.index_dim() == 1152

    s2 = Store(path, dim=768)  # reopen existing 1152 index with a 768 config
    assert s2.index_dim() == 1152  # still the on-disk dim before reset
    s2.reset()
    assert s2.index_dim() == 768  # recreated at new dim
    assert s2.stats()['vectors'] == 0  # empty after rebuild


def test_transcripts_operations(tmp_path):
    """Test adding and searching transcripts, both semantically and via FTS."""
    s = Store(tmp_path / 'idx')

    rows = [
        {
            'id': 't_0',
            'media_path': '/video.mp4',
            'media_type': 'transcript',
            'text': 'a quick brown fox jumps over the lazy dog',
            'vector': list(np.zeros(768, dtype=np.float32)),
            'start_time': 0.0,
            'end_time': 2.5,
        },
        {
            'id': 't_1',
            'media_path': '/video.mp4',
            'media_type': 'transcript',
            'text': 'hello world and welcome to this video',
            'vector': list(np.ones(768, dtype=np.float32)),
            'start_time': 2.5,
            'end_time': 5.0,
        },
    ]
    s.add_transcripts(rows)

    # test vector search
    v_hits = s.search_transcripts_vector(
        np.ones(768, dtype=np.float32), top_k=1
    )
    assert len(v_hits) == 1
    assert v_hits[0]['id'] == 't_1'

    # test fts search
    f_hits = s.search_transcripts_fts('brown fox', top_k=1)
    assert len(f_hits) == 1
    assert f_hits[0]['id'] == 't_0'

    # test delete_file cleanup
    s.delete_file('/video.mp4')
    f_hits_empty = s.search_transcripts_fts('hello', top_k=1)
    assert len(f_hits_empty) == 0

    # check that transcripts are gone from stats if we add stats for transcripts,
    # but the task doesn't explicitly require updating stats() for transcripts, so this is enough.


def test_esc_rejects_control_characters():
    """_esc raises ValueError for control characters (U+0000–U+001F)."""
    _esc('/path/to/file')  # normal path — no error
    _esc("/path/with'quote")  # single quote — doubled, not rejected
    with pytest.raises(ValueError, match='Unsafe character'):
        _esc('/path/\nbreak')
