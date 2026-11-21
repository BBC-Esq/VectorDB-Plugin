import pytest

from modules.kokoro import _join_clauses, split_sentences


@pytest.mark.parametrize("sentence", [
    "Under Fed. R. Civ. P. 12(b)(6), the claim fails.",
    "Id. at 5.",
    "Smith et al. v. Jones was decided in 2000.",
    "John Smith, Esq. filed the brief.",
    "She holds a Ph.D. in physics.",
    "The motion was filed by Gen. Counsel on behalf of the Dept. of Labor.",
])
def test_legal_abbreviations_do_not_end_a_sentence(sentence):
    assert len(split_sentences(sentence)) == 1


def test_real_sentence_ends_still_split():
    assert len(split_sentences("The claim fails. The court agreed! Did it matter?")) == 3


def no_output_device(*args, **kwargs):
    raise OSError("Error opening OutputStream: Invalid device")


def test_kokoro_remembers_a_playback_failure(monkeypatch):
    import queue
    import threading

    import modules.kokoro as kokoro

    monkeypatch.setattr(kokoro.sd, "play", no_output_device)
    tts = kokoro.KokoroTTS.__new__(kokoro.KokoroTTS)
    tts.stop_event = threading.Event()
    tts.audio_queue = queue.Queue()
    tts.audio_queue.put([0.0] * 10)
    tts.playback_error = None
    tts._play_audio()
    assert "Invalid device" in str(tts.playback_error) and tts.stop_event.is_set()


def test_other_voices_raise_after_a_playback_failure(monkeypatch, tmp_path):
    import numpy as np

    import modules.tts as tts

    monkeypatch.setattr(tts.sd, "play", no_output_device)
    (tmp_path / "answer.txt").write_text("One sentence to speak.", encoding="utf-8")
    audio = tts.BaseAudio()
    audio.process_text_to_audio = lambda sentences: audio.audio_queue.put((np.zeros(10, dtype=np.float32), 24000))
    with pytest.raises(RuntimeError, match="could not be played"):
        audio.run(str(tmp_path / "answer.txt"))


def test_clauses_are_rejoined_with_their_punctuation():
    assert _join_clauses("Yes, no, maybe", ["jEs", "noU", "meIbi"]) == "jEs, noU, meIbi"
    assert _join_clauses("then the law: statute, and precedent", ["a", "b", "c"]) == "a: b, c"
    assert _join_clauses("Is it done?", ["Iz It dVn\n"]) == "Iz It dVn?"


def test_clauses_fall_back_to_commas_when_the_counts_differ():
    assert _join_clauses("It costs 3.5 percent of the U.S. total, he said", ["a", "b"]) == "a, b"
    assert _join_clauses("one two", ["a", "", "b"]) == "a, b"
    assert _join_clauses("", []) == ""
