from mediasearch.walker import walk, MediaFile


def test_walk_finds_and_classifies(tmp_path, make_image):
    make_image(tmp_path / 'a.png')
    sub = tmp_path / 'sub'
    sub.mkdir()
    make_image(sub / 'b.jpg')
    (tmp_path / 'notes.txt').write_text('ignore me')
    (tmp_path / 'c.mp4').write_bytes(
        b'\x00\x00'
    )  # extension-only; walker doesn't decode

    files = list(walk([tmp_path]))
    by_name = {f.path.name: f for f in files}

    assert set(by_name) == {'a.png', 'b.jpg', 'c.mp4'}
    assert by_name['a.png'].media_type == 'image'
    assert by_name['c.mp4'].media_type == 'video'
    assert isinstance(by_name['a.png'], MediaFile)
    assert by_name['a.png'].size > 0
    assert by_name['a.png'].mtime > 0


def test_walk_is_deterministic(tmp_path, make_image):
    for n in ['z.png', 'a.png', 'm.png']:
        make_image(tmp_path / n)
    names = [f.path.name for f in walk([tmp_path])]
    assert names == sorted(names)


def test_walk_handles_oserror(tmp_path, make_image, monkeypatch):
    """walk skips files whose stat raises OSError (e.g. broken symlinks)."""
    make_image(tmp_path / 'good.png')
    (tmp_path / 'bad').write_text('will cause stat error')

    import mediasearch.walker as mw

    orig_stat = mw.Path.stat

    def _failing_stat(self):
        if self.name == 'bad':
            raise OSError('permission denied')
        return orig_stat(self)

    monkeypatch.setattr(mw.Path, 'stat', _failing_stat)
    names = [f.path.name for f in mw.walk([tmp_path])]
    assert names == ['good.png']
