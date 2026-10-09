from gui.jeeves import jeeves


def write_poems(root, text):
    (root / "Assets").mkdir()
    (root / "Assets" / "jeeves_poems.txt").write_text(text, encoding="utf-8")


def test_load_poems_labels_each_poem_with_title_and_author(tmp_path, monkeypatch):
    write_poems(tmp_path, (
        "A Dream Within a Dream\nBy Edgar Allan Poe\n\nTake this kiss upon the brow!\n"
        "@@@@@\n\u201cOzymandias\u201d\nby: Percy Bysshe Shelley\n\nI met a traveller from an antique land\n"
        "@@@@@\nLone Title\n"
        "@@@@@\n   \n"
    ))
    monkeypatch.setattr(jeeves, "PROJECT_ROOT", tmp_path)
    poems = jeeves.load_poems()
    assert [p["label"] for p in poems] == [
        "A Dream Within a Dream - Edgar Allan Poe",
        "Ozymandias - Percy Bysshe Shelley",
        "Lone Title",
    ]
    assert poems[0]["text"].startswith("A Dream Within a Dream\nBy Edgar Allan Poe")
    assert poems[0]["text"].endswith("Take this kiss upon the brow!")


def test_load_poems_without_the_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(jeeves, "PROJECT_ROOT", tmp_path)
    assert jeeves.load_poems() == []


def test_the_shipped_poems_all_have_labels_and_text():
    poems = jeeves.load_poems()
    assert len(poems) > 5
    assert poems[0]["label"] == "A Dream Within a Dream - Edgar Allan Poe"
    assert all(p["label"] and p["text"].strip() and "@@@@@" not in p["text"] for p in poems)


def test_source_rows_use_the_guide_heading_and_join_wrapped_lines():
    chunk = "### How do I create a database?\nOpen the Create Database tab and\nadd files.\n\nThen click Create."
    rows = jeeves.source_rows([chunk], [{"file_path": r"C:\guide\chunk_001.txt", "similarity_score": 0.81234}])
    assert rows == [{
        "title": "How do I create a database?",
        "path": r"C:\guide\chunk_001.txt",
        "score": 0.81234,
        "text": "Open the Create Database tab and add files.\n\nThen click Create.",
    }]


def test_source_rows_fall_back_to_the_file_name_and_tolerate_missing_metadata():
    rows = jeeves.source_rows(["Plain text without a heading", "Another chunk"], [{"file_name": "notes.txt", "similarity_score": "n/a"}])
    assert rows[0]["title"] == "notes.txt"
    assert rows[0]["text"] == "Plain text without a heading"
    assert rows[0]["score"] is None
    assert rows[1] == {"title": "unknown source", "path": "", "score": None, "text": "Another chunk"}


def test_source_rows_truncate_long_passages():
    rows = jeeves.source_rows(["### Long\n" + "word " * 400], [{"file_path": "x.txt"}])
    assert rows[0]["title"] == "Long"
    assert len(rows[0]["text"]) == 704
    assert rows[0]["text"].endswith(" ...")


def test_speech_text_drops_markdown_marks():
    spoken = jeeves.speech_text("## Creating a database\n**Open** the *Create Database* tab.")
    assert "*" not in spoken and "#" not in spoken
    assert "Open the Create Database tab." in spoken
