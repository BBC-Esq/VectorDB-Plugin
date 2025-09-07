# gui_tabs_settings_tts.py
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QGridLayout, QMessageBox, QHBoxLayout
)

from constants import WHISPER_SPEECH_MODELS


class TTSSettingsTab(QWidget):
    # 1. Describe every backend (easy to extend later)
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
                "quality": {
                    "label": "Quality",
                    "options": ["base", "medium", "high"],
                    "default": "high",
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

    # 2. Qt-setup
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
                cmb.currentTextChanged.connect(self._save_to_yaml)
                wdict[extra_key] = (lbl, cmb)
            self.widgets_for_backend[key] = wdict

        self.backend_combo.currentIndexChanged.connect(self._update_visible_extras)

    # 3. YAML helpers
    def _config_path(self) -> Path:
        return Path("config.yaml")

    def _load_from_yaml(self):
        cfg = self._try_read_yaml()

        # backend
        tts_cfg = cfg.get("tts", {}) if cfg else {}
        backend = tts_cfg.get("model", "whisperspeech")
        idx = self.backend_combo.findData(backend)
        self.backend_combo.setCurrentIndex(idx if idx != -1 else 0)

        # Bark
        bark_cfg = cfg.get("bark", {}) if cfg else {}
        for (lbl, cmb) in self.widgets_for_backend["bark"].values():
            if cmb.objectName() == "size":
                cmb.setCurrentText(bark_cfg.get("size", "small"))
            else:
                cmb.setCurrentText(bark_cfg.get("speaker", "v2/en_speaker_6"))

        # WhisperSpeech
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

        # Kyutai
        kyutai_cfg = cfg.get("kyutai", {}) if cfg else {}
        for extra_key, (lbl, cmb) in self.widgets_for_backend["kyutai"].items():
            if extra_key == "quality":
                cmb.setCurrentText(kyutai_cfg.get("quality", "high"))
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

        elif backend_key == "kyutai":
            kyutai = cfg.setdefault("kyutai", {})
            kyutai["quality"] = self.widgets_for_backend["kyutai"]["quality"][1].currentText()

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

            kyutai["hf_repo"] = "kyutai/tts-1.6b-en_fr"
            kyutai["temp"] = 0.6
            kyutai["cfg_coef"] = 2.0

        with self._config_path().open("w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    # 4. Helpers
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

        self._save_to_yaml()

    @staticmethod
    def _find_key_by_value(d: dict, value: str | None):
        for k, v in d.items():
            if v[0] == value:
                return k
        return next(iter(d))
