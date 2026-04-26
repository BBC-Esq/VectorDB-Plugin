import yaml
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QGridLayout, QMessageBox, QHBoxLayout
)

from core.constants import WHISPER_SPEECH_MODELS

WHISPER_SPEECH_SPEAKERS = ["default", "classic", "voice_b"]
WHISPER_SPEECH_VOICE_CLONING_LABEL = "Voice Cloning (Coming Soon)"


class TTSSettingsTab(QWidget):
    BACKENDS = {
        "bark": {
            "label": "Bark (GPU)",
            "extras": {
                "size": {
                    "label": "Model",
                    "options": ["normal", "small"],
                    "default": "small",
                },
                "speaker": {
                    "label": "Speaker",
                    "options": [
                        "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
                        "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
                        "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
                        "v2/en_speaker_9",
                    ],
                    "default": "v2/en_speaker_6",
                },
            },
        },
        "whisperspeech": {
            "label": "WhisperSpeech (GPU)",
            "extras": {
                "s2a": {
                    "label": "S2A Model",
                    "options": list(WHISPER_SPEECH_MODELS["s2a"].keys()),
                    "default": list(WHISPER_SPEECH_MODELS["s2a"].keys())[0],
                },
                "t2s": {
                    "label": "T2S Model",
                    "options": list(WHISPER_SPEECH_MODELS["t2s"].keys()),
                    "default": list(WHISPER_SPEECH_MODELS["t2s"].keys())[0],
                },
                "speaker": {
                    "label": "Speaker",
                    "options": WHISPER_SPEECH_SPEAKERS + [WHISPER_SPEECH_VOICE_CLONING_LABEL],
                    "default": WHISPER_SPEECH_SPEAKERS[0],
                },
            },
        },
        "chattts": {
            "label": "ChatTTS (CPU/CPU)",
            "extras": {},
        },
        "chatterbox": {
            "label": "Chatterbox (CPU/GPU)",
            "extras": {},
        },
        "googletts": {
            "label": "Google TTS (CPU)",
            "extras": {},
        },
        "kyutai": {
            "label": "Kyutai (GPU)",
            "extras": {
                "model": {
                    "label": "Model",
                    "options": ["1.6B (EN+FR, ~4.2GB VRAM)", "0.75B (EN, ~2GB VRAM)"],
                    "default": "1.6B (EN+FR, ~4.2GB VRAM)",
                },
                "voice": {
                    "label": "Voice",
                    "options": [
                        "Default Male", "Fast Male 1", "Fast Female", "Fast Male 2", 
                        "Happy Male", "Happy Female 1", "Happy Female 2", "Enunciated Female"
                    ],
                    "default": "Happy Male",
                },
            },
        },
    }

    def __init__(self):
        super().__init__()
        self.widgets_for_backend: dict[str, dict[str, QWidget]] = {}
        self._build_ui()
        self._load_from_yaml()
        self._update_visible_extras()

    def _build_ui(self):
        layout = QGridLayout(self)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 1)

        layout.addWidget(QLabel("TTS Backend:"), 0, 0)
        self.backend_combo = QComboBox()
        for key, spec in self.BACKENDS.items():
            self.backend_combo.addItem(spec["label"], userData=key)
        layout.addWidget(self.backend_combo, 0, 1)

        self._extras_box = QWidget()
        self._extras_layout = QHBoxLayout(self._extras_box)
        self._extras_layout.setContentsMargins(0, 0, 0, 0)
        self._extras_layout.setSpacing(10)
        layout.addWidget(self._extras_box, 0, 2)

        self.widgets_for_backend: dict[str, dict[str, tuple[QLabel, QComboBox]]] = {}
        for key, spec in self.BACKENDS.items():
            wdict = {}
            for extra_key, meta in spec["extras"].items():
                lbl = QLabel(meta["label"])
                cmb = QComboBox()
                cmb.setObjectName(extra_key)
                cmb.addItems(meta["options"])
                if key == "whisperspeech" and extra_key == "speaker":
                    self._disable_voice_cloning_item(cmb)
                cmb.currentTextChanged.connect(self._save_to_yaml)
                wdict[extra_key] = (lbl, cmb)
            self.widgets_for_backend[key] = wdict

        self.backend_combo.currentIndexChanged.connect(self._update_visible_extras)
        self.widgets_for_backend["kyutai"]["model"][1].currentTextChanged.connect(
            self._update_kyutai_voice_visibility
        )

    def _config_path(self) -> Path:
        return Path("config.yaml")

    def _load_from_yaml(self):
        cfg = self._try_read_yaml()

        tts_cfg = cfg.get("tts", {}) if cfg else {}
        backend = tts_cfg.get("model", "whisperspeech")
        idx = self.backend_combo.findData(backend)
        self.backend_combo.setCurrentIndex(idx if idx != -1 else 0)

        bark_cfg = cfg.get("bark", {}) if cfg else {}
        for (lbl, cmb) in self.widgets_for_backend["bark"].values():
            if cmb.objectName() == "size":
                cmb.setCurrentText(bark_cfg.get("size", "small"))
            else:
                cmb.setCurrentText(bark_cfg.get("speaker", "v2/en_speaker_6"))

        if tts_cfg.get("model") == "whisperspeech":
            self.widgets_for_backend["whisperspeech"]["s2a"][1].setCurrentText(
                self._find_key_by_value(
                    WHISPER_SPEECH_MODELS["s2a"], tts_cfg.get("s2a")
                )
            )
            self.widgets_for_backend["whisperspeech"]["t2s"][1].setCurrentText(
                self._find_key_by_value(
                    WHISPER_SPEECH_MODELS["t2s"], tts_cfg.get("t2s")
                )
            )
            speaker = tts_cfg.get("speaker", WHISPER_SPEECH_SPEAKERS[0])
            if speaker not in WHISPER_SPEECH_SPEAKERS:
                speaker = WHISPER_SPEECH_SPEAKERS[0]
            self.widgets_for_backend["whisperspeech"]["speaker"][1].setCurrentText(speaker)

        kyutai_cfg = cfg.get("kyutai", {}) if cfg else {}
        for extra_key, (lbl, cmb) in self.widgets_for_backend["kyutai"].items():
            if extra_key == "model":
                cmb.setCurrentText(kyutai_cfg.get("model_display_name", "1.6B (EN+FR, ~4.2GB VRAM)"))
            elif extra_key == "voice":
                cmb.setCurrentText(kyutai_cfg.get("voice_display_name", "Happy Male"))

    def _save_to_yaml(self):
        cfg = self._try_read_yaml()

        backend_key = self.backend_combo.currentData()
        tts_cfg = cfg.setdefault("tts", {})
        tts_cfg["model"] = backend_key

        if backend_key == "bark":
            bark = cfg.setdefault("bark", {})
            bark["size"] = self.widgets_for_backend["bark"]["size"][1].currentText()
            bark["speaker"] = (
                self.widgets_for_backend["bark"]["speaker"][1].currentText()
            )
        elif backend_key == "whisperspeech":
            tts_cfg["s2a"] = WHISPER_SPEECH_MODELS["s2a"][
                self.widgets_for_backend["whisperspeech"]["s2a"][1].currentText()
            ][0]
            tts_cfg["t2s"] = WHISPER_SPEECH_MODELS["t2s"][
                self.widgets_for_backend["whisperspeech"]["t2s"][1].currentText()
            ][0]
            speaker_choice = self.widgets_for_backend["whisperspeech"]["speaker"][1].currentText()
            if speaker_choice in WHISPER_SPEECH_SPEAKERS:
                tts_cfg["speaker"] = speaker_choice

        elif backend_key == "kyutai":
            kyutai = cfg.setdefault("kyutai", {})

            model_mapping = {
                "1.6B (EN+FR, ~4.2GB VRAM)": ("kyutai/tts-1.6b-en_fr", 32),
                "0.75B (EN, ~2GB VRAM)": ("kyutai/tts-0.75b-en-public", 16),
            }
            selected_model_display = self.widgets_for_backend["kyutai"]["model"][1].currentText()
            hf_repo, n_q = model_mapping[selected_model_display]
            kyutai["model_display_name"] = selected_model_display
            kyutai["hf_repo"] = hf_repo
            kyutai["n_q"] = n_q

            voice_mapping = {
                "Default Male": "expresso/ex04-ex03_default_002_channel2_239s.wav",
                "Fast Male 1": "expresso/ex01-ex02_fast_001_channel1_104s.wav", 
                "Fast Female": "expresso/ex01-ex02_fast_001_channel2_73s.wav",
                "Fast Male 2": "expresso/ex04-ex03_fast_001_channel2_25s.wav",
                "Happy Male": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
                "Happy Female 1": "expresso/ex04-ex02_happy_001_channel1_118s.wav",
                "Happy Female 2": "expresso/ex04-ex02_happy_001_channel2_140s.wav",
                "Enunciated Female": "expresso/ex04-ex03_enunciated_001_channel2_342s.wav",
            }

            selected_voice_display = self.widgets_for_backend["kyutai"]["voice"][1].currentText()
            kyutai["voice"] = voice_mapping[selected_voice_display]
            kyutai["voice_display_name"] = selected_voice_display

            kyutai["temp"] = 0.6
            kyutai["cfg_coef"] = 2.0

        with self._config_path().open("w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    def _try_read_yaml(self):
        try:
            with self._config_path().open() as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            QMessageBox.warning(self, "Configuration Error", str(e))
            return {}

    def _update_visible_extras(self):
        while self._extras_layout.count():
            item = self._extras_layout.takeAt(0)
            if (w := item.widget()):
                w.setParent(None)

        chosen = self.backend_combo.currentData()
        for lbl, cmb in self.widgets_for_backend[chosen].values():
            self._extras_layout.addWidget(lbl)
            self._extras_layout.addWidget(cmb)
            lbl.show()
            cmb.show()

        if chosen == "kyutai":
            self._update_kyutai_voice_visibility()

        self._save_to_yaml()

    def _update_kyutai_voice_visibility(self):
        model_text = self.widgets_for_backend["kyutai"]["model"][1].currentText()
        voice_lbl, voice_cmb = self.widgets_for_backend["kyutai"]["voice"]
        supports_voices = model_text.startswith("1.6B")
        voice_lbl.setVisible(supports_voices)
        voice_cmb.setVisible(supports_voices)

    @staticmethod
    def _find_key_by_value(d: dict, value: str | None):
        for k, v in d.items():
            if v[0] == value:
                return k
        return next(iter(d))

    @staticmethod
    def _disable_voice_cloning_item(cmb: QComboBox):
        idx = cmb.findText(WHISPER_SPEECH_VOICE_CLONING_LABEL)
        if idx == -1:
            return
        model = cmb.model()
        item = model.item(idx)
        if item is not None:
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
            item.setToolTip("Coming soon")
