import json
import shutil
import subprocess
from pathlib import Path

import pytest

MARKDOWN_JS = Path(__file__).resolve().parent.parent / "gui" / "web_common" / "web" / "markdown.js"
RUNNER = (
    "const fs = require('fs');"
    "globalThis.window = {};"
    "eval(fs.readFileSync(process.argv[1], 'utf8'));"
    "const cases = JSON.parse(fs.readFileSync(0, 'utf8'));"
    "process.stdout.write(JSON.stringify(cases.map((text) => window.Markdown.render(text))));"
)

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


def render(*texts):
    result = subprocess.run(["node", "-e", RUNNER, str(MARKDOWN_JS)], input=json.dumps(list(texts)),
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_headings_keep_a_trailing_hash_that_belongs_to_a_word():
    with_hash, closed = render("## Working with C#", "## Title ##")
    assert with_hash == "<h4>Working with C#</h4>"
    assert closed == "<h4>Title</h4>"


def test_a_code_block_under_a_numbered_step_stays_a_code_block():
    html, = render("1. Install it:\n   ```bash\n   pip install foo\n     --upgrade\n   ```\n2. Run it")
    assert html == ('<ol><li>Install it:</li></ol><pre><code>pip install foo\n  --upgrade</code></pre>'
                    '<ol start="2"><li>Run it</li></ol>')


def test_nested_items_with_uneven_indentation_are_all_kept():
    html, = render("- A\n    - B\n  - C\n- D")
    assert html == "<ul><li>A<ul><li>B</li><li>C</li></ul></li><li>D</li></ul>"


def test_a_link_stops_before_a_closing_quote_or_full_stop():
    quoted, query = render('He wrote "see https://example.com/page" in the note.', "See https://example.com/a?x=1&y=2.")
    assert quoted == ('<p>He wrote &quot;see <a href="https://example.com/page">https://example.com/page</a>&quot; '
                      "in the note.</p>")
    assert query == ('<p>See <a href="https://example.com/a?x=1&amp;y=2">https://example.com/a?x=1&amp;y=2</a>.</p>')


def test_double_underscore_names_are_not_turned_into_bold():
    names, link, bold, labelled = render(
        "Edit __init__.py and __main__ here.",
        "https://github.com/org/repo/blob/main/pkg/__init__.py",
        "This is __really important__ and **bold**.",
        "[**docs**](https://example.com/a_b_c)",
    )
    assert names == "<p>Edit __init__.py and __main__ here.</p>"
    assert link == ('<p><a href="https://github.com/org/repo/blob/main/pkg/__init__.py">'
                    "https://github.com/org/repo/blob/main/pkg/__init__.py</a></p>")
    assert bold == "<p>This is <strong>really important</strong> and <strong>bold</strong>.</p>"
    assert labelled == '<p><a href="https://example.com/a_b_c"><strong>docs</strong></a></p>'
