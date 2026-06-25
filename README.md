<div align="center">

<img width="1536" height="248" alt="splash" src="https://github.com/user-attachments/assets/8ecfa804-8c98-4219-9204-bc5b7aaa69da" />

### Create and search a vector database from a wide variety of file types and get more reliable [responses from an LLM](https://www.youtube.com/watch?v=8-ZAYI4MvtA).  This is commonly referred to as ["retrieval augmented generation."](https://medium.com/@vici0549/search-images-with-vector-database-retrieval-augmented-generation-rag-3d5a48881de5)

</div>


<div align="center">
  <h3><u>Requirements</u></h3>

| Tool                                                                                     | Purpose                                           |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 🪟 Microsoft Windows                                                                     | **Only** for Windows but open to pull requests |
| 🐍 [Python 3.11–3.13](https://www.python.org/downloads/)                                 | Run the application                               |
| 🌿 [Git](https://git-scm.com/downloads)                                                  | Clone / manage the repository                     |
| 🧲 [Git LFS](https://git-lfs.com/)                                                       | Handle large model files                          |
| 📄 [Pandoc](https://github.com/jgm/pandoc/releases)                                      | Document parsing support                          |
| 🛠️ [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | Required for compiling dependencies               |

<details>
<summary>Or you can try running these commands in Powershell on Windows:</summary>

### Install:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --silent --accept-source-agreements --accept-package-agreements --override "--wait --quiet --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

### Verify installation:
```
Test-Path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
```
</details>

</div>

<a name="installation"></a>
<div align="center"> <h2>Installation</h2></div>
  
### Download the latest "release," extract, navigate to the `src` folder, and run the following commands:

```
python -m venv .
```
```
.\Scripts\activate
```
```
python setup_windows.py
```
```
python gui.py
```

<div align="center">

### Inputs → Processing → Vector Database

|                |                                                                                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📂 **Ingest**  | 📄 `.pdf`, `.docx`, `.txt`, `.html`, `.htm`, `.md`, `.csv`, `.xls`, `.xlsx`, `.xlsm`, `.rtf`, `.eml`, `.msg`  <br> 🖼️ `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`, `.tiff`  <br> 🎵 `.mp3`, `.wav`, `.m4a`, `.ogg`, `.wma`, `.flac`  <br> 🎬 video (optional, via [TwelveLabs](#optional-video-understanding-with-twelvelabs)) |
| ⚙️ **Process** | 📝 Extract text from documents  <br> 🖼️ Generate descriptions from images  <br> 🎧 Transcribe speech from audio  <br> 🎬 Describe video with TwelveLabs Pegasus (optional)                                  |
| 🧠 **Store**   | All processed content is embedded and saved into the vector database for searching.                                                                                                              |

### Query → LLM → Output

|                 |                                                             |
| --------------- | ----------------------------------------------------------- |
| ❓ **Ask**       | ⌨️ Type **or** 🎙️ record a question                    |
| 🧠 **Retrieve** | Relevant chunks are pulled from the vector database         |
| 🤖 **Generate** | Sent to an LLM (Local Model, [Kobold](https://github.com/LostRuins/koboldcpp), [LM Studio](https://lmstudio.ai/), or ChatGPT) |
| 💬 **Respond**  | LLM returns an answer based on the context you provided        |
| 🔊 **Optional** | Text-to-speech can read the response aloud                  |

</div>

<div align="center"> <h2>Usage</h2></div>

> [!NOTE]
> Instructions on how to use the program are being consolidated into the `Ask Jeeves` functionality, which can be accessed from the "Ask Jeeves" menu option.  Please create an issue if Jeeves is not working.

<a name="optional-video-understanding-with-twelvelabs"></a>
<div align="center"> <h2>Optional: Video understanding with TwelveLabs</h2></div>

Audio files are transcribed locally with Whisper, but full **video** understanding
(visuals + speech + on-screen text) is available as an *opt-in* cloud backend via
[TwelveLabs](https://twelvelabs.io). It is off by default and changes nothing about
the existing local pipeline.

- **Pegasus** analyzes a video and produces a rich text description, which is written
  to `Docs_for_DB` in the same `{page_content, metadata}` shape as a transcript, so it
  flows through the normal chunk → embed → store → search pipeline.
- **Marengo** produces 512‑dim multimodal embeddings for video and text, for callers
  that want to embed video directly instead of going through a description.

Set your key in `config.yaml`:

```yaml
twelvelabs:
  api_key: "tlk_..."        # or set the TWELVELABS_API_KEY environment variable
  pegasus_model: pegasus1.5
  marengo_model: marengo3.0
```

Then ingest a video by URL:

```python
from modules.twelvelabs_video import TwelveLabsVideoProcessor

TwelveLabsVideoProcessor().start_transcription_process(
    "https://example.com/clips/talk.mp4"
)  # writes Docs_for_DB/talk.json, ready for the vector database
```

You can grab a free API key at https://twelvelabs.io — there's a generous free tier.

<a name="request-a-feature-or-report-a-bug"></a>

<div align="center"> <h2>Request a Feature or Report a Bug</h2></div>

Feel free to report bugs or request enhancements by creating an issue on github and I will respond promptly.

<a name="contact"></a>
<div align="center"><h2>Contact</h2></div>

I welcome all suggestions - both positive and negative.  You can e-mail me directly at "bbc@chintellalaw.com" or I can frequently be seen on the ```KoboldAI``` Discord server (moniker is ```vic49```).  I am always happy to answer any quesitons or discuss anything vector database related!  (no formal affiliation with ```KoboldAI```).
