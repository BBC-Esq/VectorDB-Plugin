from modules.kokoro import _join_clauses


def test_clauses_are_rejoined_with_their_punctuation():
    assert _join_clauses("Yes, no, maybe", ["jEs", "noU", "meIbi"]) == "jEs, noU, meIbi"
    assert _join_clauses("then the law: statute, and precedent", ["a", "b", "c"]) == "a: b, c"
    assert _join_clauses("Is it done?", ["Iz It dVn\n"]) == "Iz It dVn?"


def test_clauses_fall_back_to_commas_when_the_counts_differ():
    assert _join_clauses("It costs 3.5 percent of the U.S. total, he said", ["a", "b"]) == "a, b"
    assert _join_clauses("one two", ["a", "", "b"]) == "a, b"
    assert _join_clauses("", []) == ""
