### What is the VectorDB-Plugin and what can it do?
VectorDB-Plugin is a program that lets you build a vector database from your documents (text files, PDFs, images, etc.) and use it
with a large language model for more accurate answers. This approach is known as Retrieval Augmented Generation (RAG) – the software
finds relevant pieces of your data (embeddings) and feeds them into an AI chat model so the answers are based on your own content.
In simple terms, VectorDB-Plugin "supercharges" a language model by giving it a memory of your files, which improves the factual
accuracy of responses. You can search your database by asking questions in plain language, and the program will retrieve matching
chunks from your data and have the chat model incorporate them into its answer.

### What are the system requirements and prerequisites?
System Requirements for VectorDB-Plugin include a Windows operating system (Windows 10 or 11) and Python (version 3.11, 3.12, 3.13
or 3.14, but not a free-threaded build). You should also have Git installed (with Git LFS for handling large model files).
If you plan to use GPU acceleration or certain models, you'll need a suitable C++ compiler and possibly Visual Studio build tools
on Windows. An NVIDIA GPU (a GeForce GTX 16-series or RTX 20-series card or newer) is optional but greatly speeds up embedding and
model inference. Without one, the program installs in CPU-only mode, which works but is slower and offers fewer models. Make sure
you have sufficient disk space for storing models and databases – vector models and chat models can be several hundred MBs to a
few GBs each.

### Why is Visual Studio required to run this program?
Visual Studio is requried to run this program because some of the libraries that it relies on must be compiled before they can be
installed.  A common order that you will receive if you have not installed Visual Studio will state that
"Microsoft Visual C++ 14.0 or greater is required" making it clear that you have not installed it correctly. Moreover, when
installing Visual Studio you must also install "Build Tools" or select certain features.  For example, when installing
Visual Studio Build Tools 2022 you must choose "Desktop development with C++ workload" from the righthand side and check the boxes
for "MSVC v143 – VS 2022 C++ x64/x86 build tools...", "Windows 10 SDK (10.0.19041.0 or later)," or "Windows 11 SDK (10.0.22621.0),"
"C++ CMake tools for Windows," "C++ CMake tools for Windows," "C++ AddressSanitizer," and potentially others.

### How do I install and launch the VectorDB-Plugin?
Download the latest release from the GitHub repository (look for a ZIP file under Releases). Extract the ZIP archive to a folder of
your choice.  Create a virtual environment by opening a command prompt within the "src" directory of the extracted files by running
the command "python -m venv ." The second step is to activate the virtual environment by running the command ".\Scripts\activate".
Third, run the setup script with the command "python setup_windows.py". The setup script checks whether you have a supported
NVIDIA GPU and installs either the GPU version or the CPU-only version, and it also downloads the Kokoro text to speech model. It is
important to note that this progam is only supported on Windows at this time.  Lastly, you run the program by using the command
"python gui.py". A window should open with this program's graphical user interface.

### Can I use this program without an NVIDIA GPU (CPU-only mode)?
Yes. When you run "python setup_windows.py" the installer checks your hardware. If it does not find a supported NVIDIA GPU (a
GeForce GTX 16-series or RTX 20-series card or newer, meaning CUDA compute capability 7.5 or higher), it tells you and installs the
CPU-only version. Older cards such as the GTX 10-series also use CPU-only mode. The GPU version also needs NVIDIA driver 580 or
newer; with an older driver the program runs in CPU-only mode until you update it. To force a CPU-only install on a computer that has a
GPU, run "python setup_windows.py --force-cpu" instead. In CPU-only mode the program hides models that are too large or too slow to
run on a CPU. You can still use the local chat models LiquidAI .35b, .7b, and 1.2b, Qwen 3 0.6b, 1.7b, and 4b, Granite 3b, and
Gemma 3 4b; the Liquid-VL 480M vision model; every embedding model except the 4B and 8B versions of Qwen3-Embedding and
Octen-Embedding and the 4B version of F2LLM; all float32 whisper models; and the Kokoro, Kyutai Pocket, ChatTTS, and Google TTS
text to speech backends. The
Half precision switch in the Settings Tab is greyed out because half precision only helps on a GPU. Everything runs more
slowly than on a GPU, so smaller models
are recommended, and for chatting with larger models LM Studio is a good choice because it runs them efficiently on a CPU. Ask
Jeeves works the same way in either mode.

### How do I change the program's color theme?
This program ships with a large selection of color themes so you can customize its appearance. To change the theme, open the
'File' menu at the top of the window, hover over 'Themes,' and click any theme name (such as default, matrix, tron, monet, or
steel_ocean) from the list of roughly two dozen options. The new theme is applied immediately -- there is no need to restart the
program -- and your choice is saved automatically so it persists the next time you launch the program. Several themes are designed
with specific needs in mind, such as a colorblind-friendly option.

### What is the system metrics bar at the bottom of the window?
Along the bottom of the main window is a live system-metrics monitor that helps you keep an eye on your hardware while creating or
querying databases. It displays your CPU usage and RAM usage, and -- if you have an NVIDIA GPU -- your GPU utilization, VRAM
usage, and GPU power draw, each as a percentage. Right-click the metrics bar to open a menu where you can change how the metrics
are drawn, choosing among a bar display, a sparkline (a small line graph of recent history), a speedometer gauge, or an arc gauge,
and where you can start or stop the monitoring. Watching the VRAM meter is especially helpful for making sure a model and its
contexts fit within your GPU's memory.

### How do I download or add embedding models?
The Models Tab lets you browse and download embedding models.  Each model has a card that shows its name, provider, license,
number of parameters (e.g. 109M means 109 million), and size on disk, along with colored labels for its dimensions, its maximum
sequence length in tokens, and its precision.  The precision label shows the format the creator saved the model in (e.g. float32 or
bfloat16) and, after an arrow, the format it will actually run in on your computer with your current Half precision setting.  Hover over any
label or score for more details.  Each card also lists the model's benchmark scores for English, multilingual, legal, and code
retrieval, and a star marks the best score in each category.  To download a model, click the "Download" button on its card.  This
saves the model files to the "Models/vector" folder, and the button changes to "Downloaded" when it finishes.  Only one model
downloads at a time.  Use the search box and the filters above the cards to show only the models you want, for example models with
at least 1,024 dimensions or a maximum sequence of 8,192 tokens, and use the Sort menu to rank the models by a benchmark score, size,
or other property.  The List button switches to a compact table that you can sort by clicking a column heading.  Clicking a model's
name opens its page on Hugging Face.

### What does the precision label on each model card in the Models Tab mean?
The precision label shows the floating point format that the model was saved in and, when it differs, an arrow followed by the
format that the model will actually run in on your computer.  For example, "fp32 → bf16" means a float32 model will run in
bfloat16.  The format depends on your compute device and the Half precision switch in the Settings Tab.  On a CPU, every model runs in
float32.  On an NVIDIA GPU with Half precision off, every model also runs in float32.  With Half precision on, models run in bfloat16
on GPUs that support it (RTX 30-series and newer) or in float16 on older GPUs, except for the few models that do not work
correctly in float16, which run in float32 instead.  Hover over the label to see every combination for your computer.

### How do I query the database for answers?
Choose the database to search from the Database menu at the bottom of the Query Database tab, and choose what answers from the
'Answered by' menu: a Local Model (with its own Model menu), Kobold, LM Studio, ChatGPT, or one of the MiniMax models. Type your
question in natural language, for example what does the quarterly report say about revenue, and click 'Ask' or press Ctrl+Enter.
The program searches the database for the most relevant chunks and sends them with your question to the chosen backend, and the
answer streams into a card together with the sources it came from; click a source to open the file, or its folder icon to show it
in its folder. Each question adds a new card, so earlier answers stay on screen until you click 'Clear'. The tab also shows your
current query settings (contexts, similarity, search term, file type and device) with a link to the Settings tab, and it remembers
the database you chose the next time you start the program. If you only want to see the retrieved chunks without asking a chat
model, turn on 'Chunks only'.

### What do the Copy and Speak buttons on an answer do?
Every finished answer card in the Query Database tab has a 'Copy' button that copies the answer and its list of sources (or the
retrieved chunks, for a Chunks only search) to the clipboard as plain text so you can paste it elsewhere, and a 'Speak' button
that reads the answer aloud with the text to speech backend chosen on the Settings tab. While an answer is being read, a 'Stop'
control lets you end it early.

### How do I eject a local chat model to free up memory?
When the Local Model backend is selected in the Query Database tab and a model is loaded, an 'Eject' button appears next to the
Model menu. Eject unloads the chat model from memory, which frees up VRAM without closing the program, handy when you want to
create a database or simply reclaim memory; the model loads again automatically the next time you ask a question with it selected.
The Model menu also shows each model's approximate memory use and whether it still needs to be downloaded or requires a Hugging
Face access token, and answers from a local model show how much of the model's context window the instructions, question, sources
and answer used.

### Which chat backend should I use?
The program offers four options for generating answers from your database content. The Local Models backend uses chat models downloaded
directly from Huggingface and does not rely on any exernal program. The Kobold backend connects to a Kobold server that has already
loaded a chat model.  You must download Kobold prior to using this backend and set it up correctly.  The LM Studio backend is similar
in that it requires downloading an external program prior to using it and setting it up correctly.  The ChatGPT backed uses the API
from Openai and connects to one of several models. You must first create an account with Openai and get an API key, which must then
be entered into this program from the menu at the top.  Unlike the other backends, the ChatGPT backend cannot run without an Internet
connection.  If you do not have a supported NVIDIA GPU, the LM Studio backend is recommended because it runs larger chat models
much faster on a CPU than the Local Models backend does.

### What is LM Studio chat model backend?
LM Studio is an application that allows users to run and interact with local language models on their own hardware. This program
integrates with LM Studio, and the GitHub repository contains detailed instructions for setup and usage. When you query the vector
database within the Query Database tab you can choose LM Studio as the backend that ultimately receives the query (along with the
contexts from the vector database) and provides a response to your question.  LM Studio can be downloaded from this website:
https://lmstudio.ai/.  The documentation regarding how to properly set up the program is here: https://lmstudio.ai/docs/app.

### What is Kobold chat model backend?
Kobold is an application that allows users to run and interact with local language models on their own hardware. This program
integrates with Kobold, and the GitHub repository contains detailed instructions for setup and usage. When you query the vector
database within the Query Database tab you can choose Kobold as the backend that ultimately receives the query (along with the
contexts from the vector database) and provides a response to your question.  You can get the latest release from Kobold from this
website: https://github.com/LostRuins/koboldcpp.  On Windows machines, it is crucial that you do two things before using Kobold.  First,
right-click on the file and check the "Unblock" checkbox near the bottom.  Secondly, you must click the "Compatibility" tab and check
the box that says "Run this program as an administrator."  Without these steps it will likely fail.  The documentation regarding how
to use Kobold is here: https://github.com/LostRuins/koboldcpp/wiki.

### What is the OpenAI GPT Chat Model Backend?
The Chat GPT models backend allows you to send queries directly to OpenAI and get a response.  To do so you must first have an API key.
To get an API key for accessing OpenAI's large language models, first create an account by visiting OpenAI's signup page and completing
the registration. Once logged in, go to the API keys page, click "Create new secret key," optionally name it, and then click
"Create secret key" to generate it. Make sure to copy and store the key securely, as it won't be shown again. To activate the key,
visit the Billing section and add your payment details. For a more detailed walkthrough, you can refer to this step-by-step tutorial.

### What is the MiniMax chat backend?
MiniMax is a newer online chat-model backend, joining the Local Models, Kobold, LM Studio, and ChatGPT options. Like ChatGPT, it
is a cloud service: it sends your query and the retrieved contexts to MiniMax's servers and returns a response, so it requires an
Internet connection and an API key. To use it, first enter your MiniMax API key by going to the 'File' menu and selecting 'MiniMax
API Key.' Then, in the Query Database Tab, choose one of the MiniMax options from the 'Answered by' menu and ask your question as
usual. MiniMax offers several model variants (including a high-speed option) that you select directly from that menu. If the key
is missing, the tab says so and offers a button that opens the MiniMax API key prompt. It is a good choice if you want access to a
powerful hosted model without running anything locally.

### What is the Chat Backend Settings dialog?
The Chat Backend Settings dialog, opened from the 'File' menu via 'Chat Backend Settings,' is where you configure the external
chat-model backends in one place, with a tab for each. The ChatGPT tab is the most detailed: you enter your OpenAI API key (with a
Show/Hide button), choose which OpenAI model to use, and -- for the newer models -- set a 'Verbosity' level and a 'Reasoning
Effort' level, while a small panel shows the per-million-token input and output costs of the selected model so you can gauge
expense. The LM Studio tab lets you set the server port to match your LM Studio installation and toggle whether the model's
thinking process is shown. The Kobold and MiniMax tabs are placeholders for now (Kobold uses its default connection, and the
MiniMax API key is entered from the File menu). Click OK to save. The 'Backend settings' button on the Query Database tab opens
the same dialog on the page for the selected backend.

### What local chat models are available and how can I use them?
The "local models" option within the Query Database Tab downloads chat models directly from Huggingface and requires no external
program. You can select a local model from the Model menu, which shows each model's approximate memory use and which ones still
need to be downloaded; when you use a model for the first time it is downloaded automatically and can then be used for later
queries. Without a supported NVIDIA GPU only the smaller chat models are listed. Please note that certain models are "gated,"
which means that you must first enter a huggingface access token; the tab tells you when a gated model needs one and offers a
button to enter it. You can create an access token on Huggingface's website and then enter it within the "File" menu within this
program in the upper left. You must do this before trying to use certain "gated" "local models". To get a Huggingface access token
you must create a huggingface account and then go to your profile. On the left-hand side will be an "Access Tokens" option. Then
in the upper right is a "Create new token" button. Check the box that says "Read access to contents of all public gated repos you
can access" then click "Create token."

### How do I get a huggingface access token?
Some chat models in this program are "gated" and require a Huggingface access token.  If a model is gated and you haven't provided an
access token this program will notify you.  To obtain an access token you must create a huggingface account and then go to your profile.
On the left-hand side will be an "Access Tokens" option.  Once clicked, in the upper right is a "Create new token" button.  Check the
box that says "Read access to contents of all public gated repos you can access" then click "Create token."  You can then enter the
access token in this program by going to the "File" menu and selecting "Huggingface Access Token."  You can subsequently change your
access token within this program by repeating the same steps.

### What is a context limit or maximum sequence length?
The phrase "context limit" refers to the maximum number of tokens that a model can handle at once.  With chat model the phrase
"context limit" is usually used and with embedding models it is customary to use the phrase "maximum sequence length."  Regardless,
it refers to the same thing.  When you choose a chunk size in this program it is important to make sure that the chunk size does not
exceed the maximum sequence length of the embedding model.  You can see each model's limit in the Models Tab.  Remember, these limits
are given in tokens wherease the chunk size setting is in characters.  This is because the text extraction and splitting operates in
terms of characters.  On average, one token is three to four character so you will need to do some rough math when setting the chunk
size setting to make sure that it does not exceed the embedding model's maximum sequence length.

### What happens if I exceed the maximum sequence length of an embedding model?
If the chunks you create will exceed the embedding model's maximum sequence length they will be truncated, leading to suboptimal search
results.  In other words, if a chunk is too long the end will be cut off before the embeddings are created in order to ensure that
the chunk is less than the maximum sequence length.  This obviously leads to suboptimal search results because some meaning is lost.
You can check the maximum sequence length for all embedding models that this program uses by inspecting the model within the Models Tab.
It is very important that you know the maximum sequence length before using an embedding model.

### How many contexts should I retrieve when querying the vector database?
For simple question-answer use cases, 3-6 chunks should suffice. For a typical book, a chunk size of 1200 characters with an
overlap of 600 characters can return up to 6 contexts. Advanced embedding models are often capable of retrieving the most relevant
context in the first or second result.  If you are not getting relevant results in the first three to six results then you desperately
need to revise your queries because the issue is not with the number of contexts being returned.  The type of query and how your phrase
it can be even more important than the actual number of chunks returned.  With that said, there are use cases for returning a lot of
chunks as well for more complex scenarios, especially now that a lot of chat models have extended context limits.  To give one example,
let's say that you embed a lot of court cases and then ask a question of "What are the exceptions to the hearsay rule of evidence?"
It might be reasonable to request 20-30 contexts, which are then fed to the chat model for a synthesized response.

### What does the Chunks only switch do?
Typically when you ask a question within the Query Database Tab the retrieved chunks are sent to your chosen backend to get a
response from a chat model. If you turn on the "Chunks only" switch, the 'Ask' button becomes 'Search' and the tab shows only the
chunks retrieved from the vector database, each with its file name, page, similarity score, and buttons to open the file or its
folder; long chunks are shortened until you click 'Show more'. This is good for seeing verbatim what would be sent to the chat
model backend in case you need that level of detail, but the primary purpose is to enable users to see the quality of the chunks
that they are creating. For example, it gives you an idea of whether the chunk size setting you chose is sufficient, or whether a
particular embedding model is creating high enough quality embeddings for their particular use case.

### What are embedding or vector models?
Embedding models, which are sometimes referred to as vector models, are large language models specifically trained to convert a
chunk of text into a number that represents the meaning of that number.  This number, referred to as an "embedding" or "vector" can
then be entered into database to be searched for similar vectors.

### Which embedding or vector model should I choose?
There are several considerations when choosing which embedding model to use, which are important to understand because it can take
significant time and compute resources to create a vector database.  First, the size of the embedding model and how much VRAM it uses
is a factor.  In general, the large and more compute resources required for a model, the higher quality embeddings that it will produce.
Also, the maximum sequence lengh of the model can be a factor.  Most embedding models have traditionally had a 512 token limit but
modern models now have limits of 8192 tokens or even higher.  Thirdly, some embedding models are trained on specific languages like
English while others are multilingual.  All of these characteristics can be viewed within the Models Tab as well as the hyperlinks
on the Models Tab to repository for each model so you can read more about each model.

### What are the dimensions of a vector or embedding model?
The dimensions of a vector model refers to the level of detail of the embeddings that an embedding model will create.  The more
dimensions means a greater level of detail and higher quality embedding, but will require more time and computer resources to create.
Technically speaking, the number of dimensions refers to the size of the array of numbers that is the "embedding," which, as
described previously, represents the semantic meaning of a chunk of text.  For example, the array of numbers might have 384 numbers,
because the embedding model has 384 dimensions.

### What are some general tips for choosing an embedding model?
Try to use as high of a quality of an embedding model as your system resources will allow.  Although there are exceptions for newer
embedding models, embedding models typically do not use as much VRAM as typical chat models, so the real limitation when choosing
an embedding model is how much compute time you are willing to spend before the vector database is create.  It is highly recommented
to choose as high a quality of embedding model as possible.  Also, if compute resources are limited make sure to turn on the Half
precision switch within the Settings Tab.  This will run the embedding model in either bfloat16 or float16 (commonly referred to as half
precision).  Studies show that there is very little loss in quality between full precision and half precision.  Half
precision only works with a supported NVIDIA GPU, so the switch is greyed out in CPU-only mode.  Lastly, always use "cuda" within the
Settings Tab when creating embeddings if you have a GPU.  On a CPU, smaller embedding models create databases much faster, and the
Models Tab shows approximate CPU times for each size.

### What Are Vision Models?
Vision models are a category of large language models trained to understand what is in an image.  For purposes of this program,
they are used to understand what's in an image, generate a summary for an image, which can then be put into the vector database.
This program allows you to choose from multiple vision models within the Settings Tab.  Before you take a lot of time to process
a lot of images it is highly recommended that you test the various vision models within the Tools Tab to find one that suits you.

### What vision models are available in this program?
The vision models that you can use in this program can be seen within the Settings Tab in the pulldown menu where you select the
vision model you want to use.  Each of these vision models can be researched on the huggingface website if you need more details.
Also, you can Ask Jeeves for more information about a specific family of models.  In general, the visions models are arranged within
this pulldown menu from smallest at top to largest at the bottom.  The larger the model generally means the higher quality results you
will get, but not always.  Without a supported NVIDIA GPU only Liquid-VL 480M is offered, because the larger vision models are far
too slow on a CPU.  Smaller vision models that are newer sometimes outperform larger but older vision models.  Also, some
vision models excel at certain types of images over other types. The best strategy to choose an appropriate vision models before
committing to processing a large number of images is to go to the Tools Tab and test the various vision models.  You can Ask Jeeves
for details of how to do this.

### Do you have any tips for choosing a vision model?
When choosing a vision model it is recommended to choose the highest quality model that your system can run taking into consideration
the amount of compute time you are willing to spend.  Each vision model requires a certain amount of VRAM to use, which is typically
much higher than embedding models.  It is highly recommended to test all the models on a single image, which you can do within the
Tools Tab, or if you already know your VRAM limitations, only test the vision models you know you have the resources to run.  The
Tools Tab allows you to test a particular vision model on multiple images or multiple visions models on a single image.  Either way
it's important to get a feel for the vision models' quality and compute resources required before committing to procesdsing a lot
of images that will be put into a vector database.

### What is whisper and how does this program use voice recording or transcribing an audio file?
Whisper is an advanced speech recognition model developed by OpenAI that transcribes audio into text. This program uses whisper models
in two ways.  First, to allow users to record their voice into the question box when querying the vector database.  This can be done
within the Query Database Tab; simply click the "Speak a question" button, record your question, and it will be added to the
question box.
Secondly, whisper models are used to create transcriptions of audio files that can subsequently be entered into a vector database.
You can create these transcriptions within the Tools Tab.  This will create a transcript of an audio file, which you will see within
the Create Database Tab before creating the vector database.

### How can I record my question for the vector database query?
To dictate a question, go to the "Query Database" tab, click the "Speak a question" button, and speak clearly; the button shows a
timer while it records. Click "Stop recording" when you are done. The recording is transcribed on your computer and the text is
added to the end of the question box, so you can edit it before asking. If the microphone cannot be used, or the recording was too
short, the tab tells you.

### How can I transcribe an audio file to be put into the vector database?
To transcribe an audio file, go to the Transcribe Audio section of the Tools tab, choose a Model and a Precision, click the
"Choose an audio file" box to pick the file (most file formats are supported such as .mp3, .wav, .m4a, .ogg, .wma, and .flac) and
click the Transcribe button. The section shows a timer while it works, and when it finishes it names the transcript it saved in
the Docs_for_DB folder. You can then see the transcript in the "Create Database" tab and it will be entered into the vector
database when you create it.  The transcribing functionality uses the powerful `WhisperS2T` library with the `Ctranslate2`
backend.  Make sure to adjust the "Batch size" setting (1 to 150) when transcribing an audio file depending on the size of the
whisper model you choose. Increasing the batch size can improve speed but demands more VRAM, so care should be taken not to exceed
your GPU’s capacity.  You cannot transcribe while a database is being created, because a transcript saved during a build would be
removed when the build ends.

### What are the distil variants of the whisper models when transcribing and audio file?
Distil variants of Whisper models use approximately 70% of the resources of their full counterparts and are faster with very little
loss in quality.  Distil Whisper large-v3.5 is the newest distilled large model.  It is English-only and uses about the same
resources as Distil Whisper large-v3.

### What whisper model should I choose to transcribe a file?
When transcribing an audio file in order to put it into a vector database it is generally recommended to use as high a quality of
a whisper model as your hardware will support.  The quality of a whisper model is determined by a few factors.  Firstly, its size
is the most important factor - e.g. large versus medium versus small.  Secondly, the precision of the model that you use.  This
program allows you to choose float32 for the highest quality or bfloat16 or float16 (i.e. half precision) with the Precision
switch next to the Model menu.  In CPU-only mode only float32 can be chosen because half precision requires a GPU, and bfloat16
also needs an NVIDIA GPU with compute capability 8.0 or newer.  In general, using
half precision results in about 95% of the quality of float32 for half the compute resources needed.  Lastly, some of the whisper
models come in "distil" variants that have certain layers of the model removed.  Again, this typically gives approximately 95%
of the non-distil variant for half the compute resources.  Whisper large-v3 turbo is Whisper large-v3 with its decoder cut from 32
layers to 4, which makes it much faster with only a small loss in accuracy.  It is highly recommended to test the various whisper models on a small
audio file first before committing to transcribing a large audio file, which can be done within the Tools Tab.

### What are floating point formats, precision, and quantization?
Understanding floating point formats is key when making decisions about model selection and quantization. Floating point formats
represent real numbers in binary using a combination of sign, exponent, and fraction (mantissa) bits. The sign bit indicates whether
the number is positive or negative. The exponent bits determine the range or magnitude of the value. The fraction or mantissa bits
control the precision of the value.

### What are the common floating point formats?
float32 32-bit floating point with 1 sign bit 8 exponent bits and 23 fraction bits this format provides high precision and a wide
range making it a standard choice for many computing tasks float16 16-bit floating point comprising 1 sign bit 5 exponent bits
and 10 fraction bits float16 offers reduced precision and range but uses less memory and computational power bfloat16 brain floating
point this format features 1 sign bit 8 exponent bits and 7 fraction bits it has the same range as float32 but with lower precision
making it particularly useful for deep learning applications range and precision comparison format float32 approximate range plus
or minus 1.4 times 10 to the minus 45 to plus or minus 3.4 times 10 to the 38 precision in decimal digits 6 to 9 format float16
approximate range plus or minus 6.1 times 10 to the minus 5 to plus or minus 6.5 times 10 to the 4 precision in decimal digits 3 to 4
format bfloat16 approximate range plus or minus 1.2 times 10 to the minus 38 to plus or minus 3.4 times 10 to the 38 precision in
decimal digits 2 to 3

### What are precision and range regarding floating point formats and which should I use?
The choice of floating point format has several key implications precision affects the detail and accuracy of computations range
determines the scale of values that can be represented trade-offs arise when opting for lower precision formats as they reduce
memory usage and increase processing speed but may slightly reduce accuracy

### What is Quantization?
Quantization reduces the precision of the numbers used to represent a model's parameters which results in smaller models and lower
computational requirements the main goals of quantization are to improve model speed reduce memory usage ram or vram and enable models
to run on resource-constrained hardware there are two main methods of quantization post-training quantization is applied after the
model is trained quantization-aware training incorporates quantization during the training process to minimize accuracy loss common
quantization levels include int8 8-bit integer which significantly reduces model size but may introduce quantization errors and
float16 or bfloat16 which reduces size with minimal impact on accuracy

### What are the aspects or effects of quantization?
model size reduction smaller data types take up less storage performance increase reduced data size speeds up computation potential
accuracy loss reduced precision may introduce errors though often negligible for many applications

## What settings are available in this program and how can I adjust them?
The "Settings" Tab has four sections: Database Query, Database Creation, Text to Speech, and Vision Model.  The settings for LM Studio
and the other chat backends are in the Chat Backend Settings dialog in the File menu.  Please ask me a question about the specific
setting or group of settings you're interested in.

### How do I change and save a setting on the Settings Tab?
Every setting on the Settings Tab is saved as soon as you change it, so there is no separate save button.  Buttons, switches, and
pulldown menus save the moment you click them.  For boxes where you type a value, such as Chunk Size or Similarity, type the new
value and press Enter or click somewhere else to save it, or press Escape to undo your typing.  A green "Saved" check briefly appears
next to the setting's name to confirm the change.  If a value is not allowed, for example a Similarity above 1 or a Chunk Overlap that
is not smaller than the Chunk Size, a red message appears under the box and nothing is saved until you fix it.  Hover over the small
"i" next to a setting's name for an explanation of what it does.

### What are the LM Studio Server settings?
When using LM Studio as the chat model backend you can adjust a few settings in the Chat Backend Settings dialog, which you open from
the 'File' menu.  In general, however, the LM Studio program has all the settings that you should adjust.  For purposes of this
program you can set the port to match what you set within LM Studio, and you can choose whether to show the thinking process if the
model you are running within LM Studio has chain of thought.

### What are the database creation settings?
The Device setting allows you to choose either CPU or CUDA when creating a vector database.  It is always recommended to choose
CUDA if available.  The Chunk Size setting determines the size of the chunks of text that your documents will be broken into before
being turned into embeddings.  It is crucial to remember that this setting is in number of characters, not tokens, and that you must
keep the chunks within the maximum sequence length of the embedding model you are using, as expressed in tokens, and which you can
see within the Models Tab.  Remember, each token is approximately 3-4 characters.  Below the Chunk Size box the program estimates how
many tokens your chunks will be and shows the limit of the embedding model selected in the Create Database Tab, turning the estimate
yellow if your chunks might be too long.  The Chunk Overlap setting refers to how many characters at the beginning of a chunk are
from the preceding chunk.  When a document is processed sometimes it is split in the middle of an important concept and this setting
ensures that there is an overlap to avoid losing meaning.  A good rule of thumb is to set the Chunk Overlap to 25-50 percent of the
Chunk Size, and the current percentage is shown below the box.  The Half precision switch, when turned on, will run the embedding
model in half precision resulting in a slight reduction in quality but half the compute resources.  It only applies to GPUs, so it is
greyed out in CPU-only mode.

### What does the Half precision switch do and why is it greyed out?
The Half precision switch is in the Database Creation section of the Settings Tab. When it is on, the embedding model runs in half
precision (bfloat16 or float16) instead of full float32 precision while a vector database is being created. This uses about half
the memory and compute with very little loss in quality, so it is a good choice when your GPU's VRAM is limited. Half precision only
helps on a supported NVIDIA GPU, so in CPU-only mode the switch is greyed out and turned off, and databases are always created in
full precision.

### What is the Pipeline Performance setting?
The Pipeline Performance setting, found in the Database Creation settings within the Settings Tab, controls how much of your CPU
the program uses while building a vector database. It does not change the resulting database -- only how fast it is created and
how many CPU cores are kept busy. The options are Minimal (sequential, a single worker), Low (light parallelism), Normal (moderate
parallelism and the default), High (aggressive parallelism), and Maximum (all available CPU cores). Higher settings create
databases faster but make your computer less responsive for other work during creation, while lower settings leave more CPU free
at the cost of speed. Normal is a sensible default; choose Maximum when you want the fastest possible ingestion and do not need
the machine for anything else.

### What are the database query settings?
Within the Settings Tab you can adjust several settings when searching a vector database.  The Device setting allows you to choose
between CPU and CUDA.  In contrast to creating a vector database, it is recommended to always use CPU.  The Similarity setting is the
minimum relevance, from 0 to 1, that a chunk of text must have before it will be returned as a result.  A higher value returns fewer,
more relevant chunks and a lower value returns more; you should never use 1, which would return almost nothing.  The Contexts setting determines the maximum
number of chunks that will be returned, again, subject to the Similarity setting.  The Search Term Filter will require that any chunks
returned include the specified term.  The File Type setting allows you to only search for chunks of text that originated from a
particular file type.

### How does the Contexts setting work exactly?
Within the Settings Tab the Contexts setting when searching a vector database will return up to that many chunks of text assuming they
all meet the Similarity setting that you choose.  In other words, it sets the upper limit.  If there are not that many chunks that also
meet the Similarity setting it is possible to receive fewer chunks than the Contexts setting.

### What is the similarity setting?
Within the Settings Tab the Similarity setting controls the requisite relevance of a chunk related to your query in order for it to
possibly be returned.  I say "possibly" because even though a chunk might meet the Similarity setting it might not be returned if, for
example, your Contexts setting limits the numbe of chunks that will be returned.  By defaut, this program will return chunks in order
from highest relevance to lowest.  It will return the most relevant chunks that meet the Similarity setting up to the maximum
number of chunks specified in the Contexts setting.  The Similarity setting is a minimum, so a higher value means fewer chunks will be
returned, because each one must be more similar to your question, and a lower value means more chunks will be returned.  The program
ships with 0.8; if you get few or no chunks, lower it (for example to 0.5).  Do not use 1, which would return almost nothing.

### What is the search term filter setting?
Within the Settings Tab the Search Term Filter setting allows you to require that any chunks returned contain the specified search term.
It is not case-sensitive, but it does require an exact match.  For example, if you specify “child” it will only return chunks that
include the term "child" somewhere in it.  This would not include chunks that have the word "children" in it, however, since it
requires a verbatim match.  With that said, since it is not case-sensitive it would also include chunks with "Child" in them.  This
setting is especially useful when you know that a relevant chunk has a certain key word in it; otherwise, it is best to leave this blank.
To turn the filter off, click the X at the right end of the box, or delete the text and press Enter.  Lastly, it is important to understand that this setting only applies after both
the Similarity and Contexts settings.  Therefore, if the Similarity setting is too high or the Contexts setting is too low you might not
receive any chunks with your specified search term.

### What is the File Type setting?
Within the Settings Tab the File Type setting allows you to limit the chunks that are returned based on whether they originated from
a particular type of file.  Current options include images, documents, audio or all files.  It is best to use the all files option
unless you are sure that the chunks you are looking from originated from a particular type of file.

### What are text to speech models (aks TTS models) and how are they used in this program?
Text to speech models (TTS) are large language models that were specifically trained to take text as input and output audio in a spoken
voice format.  This program allows you to use TTS models to speak the response that you get after querying the vector database.

### What text to speech models are availble in this program to use?
You choose a text-to-speech (TTS) backend within the Settings Tab. The current options are Bark, WhisperSpeech, ChatTTS,
Chatterbox, Google TTS, Kokoro, Kyutai, and Kyutai Pocket. Bark and WhisperSpeech are GPU-only and produce very high quality speech;
Bark lets you pick a model size (normal or small) and a speaker voice (such as v2/en_speaker_6, usually the highest quality, or
v2/en_speaker_9, the only female voice), while WhisperSpeech lets you choose its S2A and T2S models and a speaker. Chatterbox and
Kyutai (GPU) also require a GPU. Kokoro, ChatTTS, and Kyutai Pocket can run on a CPU or a GPU; Kokoro lets you choose a voice and
a Slow, Medium, or Fast speed. Google TTS is the lightest option but is not local -- it connects to a free online Google service
and therefore requires an Internet connection. In CPU-only mode the list shows only Kokoro, Kyutai Pocket, ChatTTS, and Google TTS.
Whichever backend you select is used by the 'Speak' button on each answer in the Query Database Tab.

### What is the Bark text to speech?
Bark TTS by Suno AI is a fully generative, open-source text-to-audio model that produces highly expressive and realistic speech,
even capable of non-verbal vocalizations like laughter or sighs. Unlike traditional TTS systems that strictly follow input text,
Bark can "freestyle," deviating for prosodic expressiveness or ambient cues, which makes it especially useful for creative
applications like character dialogue, storytelling, and game development. It supports over 100 built-in speaker presets and
auto-detects more than a dozen languages, although English remains the most polished. Bark uses EnCodec and a GPT-style transformer
under the hood, trading speed for quality, and typically requires GPU acceleration. Despite its occasional unpredictability, its
rich emotional output and open MIT license make it a standout for experimental and expressive use cases.

### What is the WhisperSpeech text to speech?
WhisperSpeech by Collabora is a cutting-edge open-source project that "reverses" OpenAI's Whisper speech-to-text model to synthesize
speech from semantic audio tokens, offering an exciting glimpse into the future of modular, multilingual TTS. Inspired by Google’s
SPEAR-TTS, WhisperSpeech leverages Whisper’s deep linguistic understanding and language-neutral token representations to build a
multilingual, speaker-aware system that supports voice cloning and polyglot speech (e.g. the same voice speaking in multiple languages).
Though still under heavy development, early results show surprisingly natural and expressive audio, particularly given the open
model’s small size. It’s not yet plug-and-play like Bark or ChatTTS, but its transparency, voice customization potential, and strong
multilingual foundation make it a compelling choice for developers interested in training their own flexible, high-quality TTS pipeline.

### What is the ChatTTS text to speech?
ChatTTS is an open-source conversational TTS model specifically designed for dialogue generation, with a focus on natural prosody,
expressive timing, and multi-speaker interactions. Trained on over 100,000 hours of English and Chinese speech, it delivers highly
realistic and emotionally resonant voices tailored for chatbots and AI companions. Unlike many TTS engines, ChatTTS includes
conversational structure like speaker turns and can even insert interjections like laughter using special tokens. While it lacks a
large preset voice library like Bark, it can produce distinct speakers and supports fine-tuning on custom data. It runs efficiently
on consumer GPUs, can also run more slowly on a CPU, and offers Python bindings, making it one of the most practical and expressive TTS options for developers aiming to
build natural, back-and-forth conversational agents in English or Mandarin.

### What is the Kokoro text to speech?
Kokoro is a remarkably lightweight open-source text-to-speech model with only 82 million parameters, built on the StyleTTS 2
architecture and released under the permissive Apache-2.0 license. Despite its tiny size it produces very natural, high-quality
speech, has consistently ranked at or near the top of community text-to-speech leaderboards, and runs quickly even on a CPU. In
this program, Kokoro is the voice of the Ask Jeeves help assistant, and you can also select it in the Settings Tab as the backend
for the 'Speak' button in the Query Database Tab, where you can pick its voice and a Slow, Medium, or Fast speed. The setup
script downloads Kokoro during installation, and if it is ever missing the program offers to download it again.

### What is the Chatterbox text to speech?
Chatterbox, developed by Resemble AI, is an open-source text-to-speech model released under the permissive MIT license. Its
standout features include zero-shot voice cloning -- mimicking a voice from just a few seconds of reference audio -- and emotion-
exaggeration control. Its alignment-informed inference produces ultra-stable, natural-sounding speech, making it well suited to
real-time uses like voice assistants. In blind evaluations it has been preferred over some proprietary models such as ElevenLabs.
Within this program it requires a supported NVIDIA GPU, so it is not offered in CPU-only mode.

### What is the Google TTS text to speech?
Google TTS offers industry-leading neural speech synthesis via a cloud API, producing ultra-clear, stable voices across many
languages. It is not open-source and is not run locally -- instead this program connects to a free online Google service, which
means it requires an Internet connection. Its advantage is that it places almost no load on your own hardware (no GPU needed), so
it is a good choice on machines without a capable GPU, as long as you are comfortable sending the text to be spoken to an online
service. It provides a generous free tier suitable for typical personal use.

### What is the Kyutai text to speech?
Kyutai is a newer family of text-to-speech models offered in two forms. 'Kyutai (GPU)' is the full version and lets you choose
between a 1.6B model (English and French, roughly 4.2 GB of VRAM) and a smaller 0.75B model (English, roughly 2 GB of VRAM), along
with a selection of expressive named voices such as 'Happy Male,' 'Fast Female,' and 'Enunciated Female.' 'Kyutai Pocket (CPU)' is
a lightweight, CPU-friendly version that runs without a GPU and offers its own set of named voices (such as 'alba' and 'anna'); it
also has an optional int8 quantization checkbox that the developers say substantially reduces RAM use and speeds up inference with
no measurable loss in quality. Choose Kyutai (GPU) for the highest quality if you have the VRAM, or Kyutai Pocket for quick, local
speech on the CPU.

### Which text to speech backend or models should I use
It is recommended to experiment with each backend to find the voice you like. In general, Bark and WhisperSpeech produce the
highest quality results but require a GPU, as does Chatterbox. Kokoro is fast and natural sounding on either a CPU or a GPU, which
makes it a good default, and ChatTTS and Kyutai Pocket are also strong options that run on either, making them a good choice if you
do not have a powerful GPU. Kyutai (GPU) offers expressive named voices if you have the VRAM for it. Google TTS is comparable in
quality but requires an Internet connection because it uses an online service rather than running locally.

### Can I back up or restore my databases and are they backed up automatically
When you create a vector database it is automatically backed up.  However, if you want to manually back up all databases you can go
to the Database Backup section of the "Tools" tab and click the Back Up button, which replaces the previous backup with a copy of
every current database.  Likewise, the Restore button replaces your current databases with the ones in the backup.  Both ask for
confirmation first.  The section also shows how many databases you have and names any that are not in the backup yet.  Back Up is
greyed out when there are no databases, and Restore is greyed out when there is no backup.  To back up a single database
that has no backup copy yet, or to restore one whose folder is missing, use the 'Back up' and 'Restore from backup' buttons on
the Manage Databases tab.

### What happens if I lose a configuration file and can I restore it?
This program cannot function without the config.yaml file if you lose it accidentally or it gets corrupted for some reason you can
restore a default version by if necessary copy the original configyaml from the assets folder to the main directory delete old files
and folders in vector_db and vector_db_backup to prevent conflicts

### What are some good tips for searching a vector database?
To improve your search results when searching a vector database it is important to understand the relationship between the various
settings within the Settings Tab.  When a vector database is searched it will first identify candidate chunks to return that meet the
Similarity setting.  Once it does that it will return the most relevant chunks up to the limit of the number of chunks that you set
with the Contexts setting.  After that, it will apply the Search Term Filter setting to remove any chunks that do not contain the
verbatim search term (remember, this is case-insensitive howver).  After that, these chunks are what are then sent to the chat model
along with your initial query to get a response.

### General VRAM Considerations
To conserve VRAM, disconnect secondary monitors from the GPU and, if available, use motherboard graphics ports instead. This requires
enabling integrated graphics in the BIOS, which is often disabled by default when a dedicated GPU is installed. This can be
particularly useful if your CPU has integrated graphics, such as Intel CPUs without an "F" suffix, which support motherboard
graphics ports.

### How can I manage vram?
For optimal performance, ensure that the entire LLM is loaded into VRAM. If only part of the model is loaded, performance can be
significantly degraded. It’s also important to manage VRAM efficiently by ejecting unused models when creating the vector database
and reloading the LLM after the database creation is complete. When querying the vector database, using the CPU instead of the GPU
is recommended to conserve VRAM for the LLM, as querying is less resource-intensive and can be effectively handled by the CPU.

### What are the speed and VRAM requirements for the various chat models?
Chat models vary widely in speed and VRAM use depending on their size. In general, very small models such as Qwen 3 - 0.6b deliver
exceptional speed (well over 200 characters per second) while requiring minimal VRAM (around 1.3 GB); mid-range models in the
2-to-9-billion-parameter range offer a sweet spot for most users, with speeds of roughly 150-400 characters per second and VRAM
use between about 2.5 and 9.5 GB; and the largest 24-to-32-billion- parameter models provide the strongest reasoning at the cost
of slower speeds (roughly 95-140 characters per second) and substantial VRAM requirements (around 15-20 GB). When choosing a local
chat model, pick the largest one that comfortably fits within your GPU's available VRAM alongside the contexts you retrieve.

### What are the speed and VRAM requirements for the various vision models?
Vision models show a clear inverse relationship between speed and model size: smaller models process images significantly faster
while larger models generally provide higher accuracy at the cost of reduced throughput. In general, the smallest models such as
Liquid-VL (480M) and InternVL3 - 1b offer the fastest processing and the lowest VRAM requirements; mid-range models such as
Granite Vision - 2b and the Qwen VL models in the 2B to 4B range offer a balance of speed and quality; and the largest models such
as Qwen VL - 7b and InternVL3 - 8b provide the highest quality at the cost of more VRAM and slower processing. Because newer small
models sometimes outperform older larger ones, it is best to test the candidates on a sample image using the 'Test Vision Models'
feature in the Tools Tab before committing to a large batch.

### What are maximunm context length and maximum sequence length and how to they relate?
Each embedding model has a maximum sequence length, and exceeding this limit can result in truncation. To avoid this, regularly
check the maximum sequence length of the model and adjust your settings accordingly. Reducing chunk size or the number of contexts
can help stay within these limits. Maximum "context length" refers to chat models and is very similar to maximum sequence length.
The key thing to understand is that the chunks you put into the vector database should be within the max sequence length of the
vector or embedding model you choose and the maximum context or chunks you retrieve from the vector database multiplied by their
length should stay within the chat model's context length limit.  And make sure to leave enough context for a response.

### What is the scrape documentaton feature?
Within the Tools tab you can select python libraries and scrape their documentation, up to six at the same time.  Multiple .html files will be downloaded
and you can subsequently create a vector database out of them.  Larger more complex libraries can take a significant amount of time
to scrape to make sure you have a stable Internet connection.

### Which vector or embedding models are available in this program?
All of the embedding models that this program uses are listed on the Models Tab.  You can click on a hyperlink for each one to find
out more information.  The embedding models sometimes change as different versions of this program are released and newer and better
embedding models are released.  This program vets all embedding models, however, before including them for usage.  Without a
supported NVIDIA GPU, the 4B and 8B versions of the Qwen3 and Octen embedding models and the 4B version of the F2LLM embedding
model are hidden because they are far too slow on a CPU, and a note at the top of the Models Tab gives approximate CPU times for
the remaining models.

### What is the Manage Databases tab?
The Manage Databases tab lists every vector database you have created, with the embedding model it uses, how many files and chunks
it holds, its size on disk and the date it was created; click a column heading to sort the list. Badges point out databases that
need attention: 'No backup' when there is no backup copy yet, 'Folder missing' when the database's folder was removed outside the
program, and 'Leftover folder' for a folder left behind by a database that was never finished. Click a database to see its details
(embedding model, dimensions, chunk size and overlap, size, creation time and backup) and the files it was created from. If the
embedding model it was created with is no longer downloaded, the tab warns you, because the database can't be searched until you
download that model again on the Models tab. The file list can be filtered by file type or by name and shows how many chunks each
file produced. Double-click a file, or select it and press Enter, to open it in your system's default program, or click the folder
icon at the end of its row to open the folder that contains it. When a database is created the location of each original file is
saved, so a file you have since moved or deleted is marked 'Not found' and can't be opened, although the database still searches
its text. The 'Query' button switches to the Query Database tab.

### How do I delete, back up or restore a database on the Manage Databases tab?
Select the database and click 'Delete…'. The tab asks you to confirm and says exactly what will be removed: the database and its
backup copy. Your original files are never touched, and deleting can't be undone. The same button removes a database whose folder
is missing from the list, and deletes a leftover folder, which also frees its name for a new database. A database without a backup
copy has a 'Back up' button, and a database whose folder is missing but whose backup copy still exists has a 'Restore from backup'
button. These buttons are unavailable while the database is being created on the Create Database tab or while a backup or restore
is running on the Tools tab, and the program waits for a delete, backup or restore to finish before it closes. If some files can't
be deleted because another program is using them, the tab lists them; close that program and delete the database again.

### How can I create a vector database?
Go to the Create Database tab and add the files that you want in the database with the 'Add Files' or 'Add Folder' button; you can
repeat this as many times as you like, and file types that cannot be added are skipped and listed.  Then enter a name, choose a
downloaded embedding model, and click 'Create Database'.  The tab shows the database creation settings it will use (chunk size,
overlap, precision, device and pipeline) with a link to the Settings Tab, so adjust them there first if needed, and the Create
Database button stays disabled with a short explanation until everything is ready.  To add audio transcriptions to the database
you must first transcribe audio files individually, which can only be done within the Tools Tab.  To input descriptions of images
into the vector database choose an appropriate vision model in the Settings Tab; any images you add are then processed by that
vision model when you create the database.

### What does the Create Database tab show while a database is being created?
While a database is being created, the file list is locked and the tab shows each stage of the build: reading the files,
describing images and adding transcripts, splitting the text into chunks, embedding the chunks, and saving the database.  A
progress bar counts the embedding batches, a timer shows how long the build has been running, and the 'Show log' link opens the
detailed output of the build, which helps if something goes wrong.  Click 'Cancel' to stop the build; any partial files are
removed.  When the build finishes, the tab reports how long it took and how many documents and chunks it created, lists any files
that could not be fully added, and links to the Query Database tab.  The new database is backed up automatically, and the file
list is cleared, because the program empties the list of files to add after every successful build so it is ready for the next
database.  If a build fails or is cancelled, the files stay in the list so you can try again.

### What file types can I add to a vector database?
When creating a database, this program accepts a range of document and image formats. Supported document types include .pdf,
.docx, .txt, .rtf, .html, .htm, .md, .csv, .xls, .xlsx, .xlsm, .eml, and .msg. Supported image types include .png, .jpg, .jpeg,
.bmp, .gif, .tif, and .tiff; images are turned into text descriptions by the vision model you select in the Settings Tab and then
embedded like any other text. Audio files cannot be added directly here -- you must first transcribe them in the Tools Tab, after
which the transcript can be added like any other document. If you select a file whose type is not supported, the program skips it,
adds the rest, and lists the skipped files in the Create Database tab.

### What are the rules for naming a vector database?
When you create a vector database you must give it a name, and the name has a few rules. It may contain only lowercase letters,
numbers, underscores, and hyphens; as you type, uppercase letters become lowercase, spaces become underscores, and other characters
are left out. The name must be at least three characters long, cannot be 'null' or 'none', and cannot be longer than the limit that
Windows path lengths allow for the folder the program is installed in. Each database name must be unique; the tab tells you as you
type if a database with that name already exists. The name is
how the database appears in the Query Database tab's dropdown menu and in the Manage Databases tab's list, so it is worth choosing
something descriptive, such as 'tax_records_2024' or 'project-notes.'

### What is the PDF OCR check when creating a database?
If the files you add include PDFs, a switch in the Create Database tab offers to check the PDFs for missing text first, meaning
whether any of them need OCR (optical character recognition); it is on by default. This matters because a PDF that is really just
scanned images has no extractable text layer, and embedding it would add nothing useful to the database. With the switch on, the
program inspects the pages of your PDFs when you click 'Create Database' and shows how many it has checked. If any PDF appears to
need OCR, it lists those files and stops before creating the database; you can open the full list, remove those PDFs from the file
list with one click, or run the OCR tool in the Tools Tab on them and add the resulting '_OCR' PDFs instead. The check does not
perform OCR itself. For a large number of PDFs this check can be time-consuming, but it is strongly recommended because it prevents
image-only PDFs from being silently added with no searchable text.

### How do I select files or a whole folder when creating a database?
In the Create Database Tab, click 'Add Files' to choose individual files (you can multi-select any number of supported files) or
'Add Folder' to choose a whole folder. For a folder, the program scans it for supported files and, if it finds compatible files in
subfolders as well, asks whether to include those subdirectory files too. Adding shows a progress bar with a 'Cancel' button, and
you can repeat this as many times as you like to keep adding files. The files appear in a list with their file type; the buttons
above the list show how many files of each type you added and filter the list when clicked, and the filter box narrows it by
name. Double-click a file to open it in its default program. To remove files before creating the database, click the X at the
right end of a row, or select files (click, Ctrl+click, Shift+click, or Ctrl+A for all) and press Delete or click 'Remove'.
Removing only takes a file off the list and does not delete your original file, except for transcripts made in the Tools Tab,
which are deleted because the list holds the only copy.

### Can I use images and audio files in my database?
You can use both images and audio in your vector database. Images: When you add image files (like PNG, JPG, BMP), the selected vision
model creates a text description of each image, which is then embedded like a regular text document. For example, a chart might be
described as “A line graph showing revenue over time with an upward trend.” You can then search with queries like “What does the
revenue trend look like?” and retrieve the image. Make sure you choose a vision model in the Settings Tab first and use the Test
Vision Models tool within the Tools Tab ot preview captions before using a particular model. Audio: You can't add audio files directly,
but you can use the Transcribe Audio tool (powered by OpenAI’s Whisper model) to convert audio to text. This transcript can then be
added like any other document during database creation. If you try to add audio files directly, the program skips them and points you to the Tools Tab to transcribe
it first. By converting images and audio to text, the system supports rich, multi-modal queries — as long as content is processed
correctly.

### What chat models are available with the local models option?
Within the Query Database Tab if you choose the local models option it will allow you to use a specified number of chat models that
will be downloaded directly from the Huggingface website.  On computers without a supported NVIDIA GPU the list is limited to eight
smaller models: LiquidAI .35b, .7b, and 1.2b, Qwen 3 0.6b, 1.7b, and 4b, Granite 3b, and Gemma 3 4b.  All of these models have been specifically chosen for their strength
in question answering using contexts provided by a vector database.  Please ask about a particular family of chat models for more
information or you can visit the repository for the various chat models on Huggingface for more detailed information.  The available
chat models that this program uses sometimes changes as newer models come out with higher capabilities.  All chat models that are
added or removed will be noted in the release notes on Github for the record.

### What are the Qwen 3 Chat Models?
Qwen3 is the latest release in the Qwen family of large language models from Alibaba. This program offers several sizes -- 0.6b,
1.7b, 4b, 8b, and 14b -- all usable under the liberal Apache 2.0 license. The Qwen3 models are capable of a step-by-step
"thinking" mode, but this program runs them in non-thinking mode so they answer directly and concisely without a visible reasoning
trace, which suits retrieval augmented generation well. The Qwen3 models are multilingual and are touted as supporting up to 119
languages. They were trained on approximately 36 trillion tokens, double the amount used for Qwen 2.5. Qwen has consistently
produced some of the best open source and free models available, and they are a staple of this program.

### What are the Granite 4.1 Chat Models?
The Granite 4.1 chat models are the latest in the Granite series developed by IBM and are released under the Apache 2.0 license.
They are dense, decoder-only transformers trained on roughly 15 trillion tokens and support a long context window, which makes
them well suited for retrieval augmented generation purposes.  Version 4.1 improves upon prior Granite releases in both reasoning
and general response quality.

### What is the Mistral Small Chat Model?
The Mistral Small chat model is the third iteration of Mistral models and has 24 billion parameters.  It is released under the
Apache 2.0 license for liberal usage.  Compared to larger models such as LLaMA 3.3 with 70 billion parameters and Qwen 2.5 with
32 billion parameters, the Mistral Small 3 model achieves comparable quality results across a wide range of benchmarks.  What is
unique about the Mistral Small 3 model is its size of 24 billion parameters, which often sits in the sweet spot for VRAM usage for
users having 24 gigabytes of VRAM.  Sometimes larger models having 32 billion parameters will exceed the available VRAM with longer
contexts but Mistral Small 3 leaves sufficient VRAM avaialble in such circumstances. Benchmark results also show that it excels at
reasoning, coding, math, and instruction following, oftentimes producing more succinct answers than other similarly sized models.

### What are the LiquidAI (LFM2) Chat Models?
The LiquidAI chat models are part of Liquid AI's LFM2 (Liquid Foundation Models 2) family and are built for fast, memory-efficient
inference on everyday hardware and edge devices. Instead of a standard transformer they use a hybrid architecture that combines
short convolutions with attention, which lets them run very quickly while staying small. This program offers the .35b (LFM2-350M),
.7b (LFM2-700M), and 1.2b (LFM2.5-1.2B Instruct) sizes, all instruction-tuned for direct, helpful answers. They are multilingual
and released under Liquid AI's open LFM license. Because of their small size and speed they are an excellent choice for users with
limited VRAM, and these same lightweight LiquidAI models power the built-in Ask Jeeves help assistant.

### What is the Phi 4 Chat Model?
Phi 4 is a 14-billion-parameter chat model developed by Microsoft and released in December 2024 under the permissive MIT license.
It is part of Microsoft's "Phi" series of small language models, which are known for achieving the quality of much larger models
by training heavily on carefully curated and synthetic data. Phi 4 is particularly strong at reasoning, mathematics, and coding,
and it follows instructions well, which makes it well suited for retrieval augmented generation. Despite its relatively modest
14-billion-parameter size, it often matches or exceeds the quality of substantially larger models on reasoning benchmarks while
remaining runnable on a single consumer GPU.

### What are the Gemma 3 Chat Models?
Gemma 3 is Google's latest family of open models, released in March 2025 under the Gemma license. This program uses the 4-billion
and 12-billion parameter instruction-tuned variants as chat models. Gemma 3 models are natively multimodal (capable of
understanding images as well as text) and support a very long context window of up to 128,000 tokens along with more than 140
languages, which makes them flexible for retrieval augmented generation across large sets of contexts. Please note that the Gemma
3 models are "gated," meaning you must enter a Huggingface access token before downloading them; see the entries on access tokens
for details. They offer strong general reasoning and response quality for their size.

### What are the BGE Embedding Models?
The BGE family of embedding models were created by BAAI and have long been a staple within the embedding community and this program
in particular.  They are well-respected as producing high quality embeddings for reasonable compute resources.  Although they are
over a year old now, they are still regarded as producing quality embeddings for a reasonable compute cost for most use cases.  At
the time of their release they were state of the art for open source and free embedding models.

### What are the Intfloat Embedding Models?
Similar to the BGE embedding models produced by BAAI, the Intfloat embedding models have long been a staple of high quality embedding
models in the community and this program.  They include "small," "base," and "large' variants for your particular use case.  They offer
high quality embeddings for the compute resources required and often go head-to-head in comparison with the "bge" models from BAAI.
Although they are well over a year old now they still offer high quality embeddings for a reasonable compute cost and many other
embedding models have been built upon the e5 family of models.

### What are the Qwen3 Embedding Models?
Released in June, 2025, Alibaba’s Qwen 3 Embedding family delivers state-of-the-art text embeddings while staying friendly to everyday hardware.  They are based on the popular Qwen 3 chat models but have special training to make them suitable for generating embeddings.
As of June, 2025, they hold the top three ranked spots on the Huggingface leaderboard.  They are primarily trained on English and
Chinese data, but a fair amount of their training data is also from numerous other languages so they can be reliably used for multilingual
embedding tasks as well.  They are released under the liberal Apache-2.0 license. The Qwen 3 family of embedding models comes in three
practical sizes—“small” (0.6 B parameters), “base” (4 B), and “large” (8 B). Even the 0.6 B version outperforms older 7 B embedding models, which is a phenomenal accomplishment while the 8 B model often edges out commercial offerings. All variants support long contexts (up to 32 k tokens). In CPU-only mode only the 0.6 B version is offered.

### What is the EmbeddingGemma Embedding Model?
EmbeddingGemma is a 300-million-parameter embedding model released by Google in September 2025 and built on the Gemma 3
architecture. Despite its small size it is one of the highest quality open embedding models in its class, and it is multilingual,
having been trained on over 100 languages. It produces 768-dimensional embeddings, supports a maximum sequence length of 2,048
tokens, and uses Matryoshka representation learning, which allows the embedding dimensions to be shortened for faster search with
very little loss in quality. Like the Gemma chat models, EmbeddingGemma is "gated," so you must enter a Huggingface access token
before downloading it. It is released under the Gemma license.

### What are the Octen Embedding Models?
The Octen embedding models are high-quality embedding models fine-tuned from Alibaba's Qwen3-Embedding models. This program offers
the 0.6-billion, 4-billion, and 8-billion parameter versions, which produce 1024-, 2560-, and 4096-dimensional embeddings
respectively and all support a long maximum sequence length of 8,192 tokens. They excel at domain-specific retrieval --
particularly legal, financial, healthcare, and code embeddings -- while also serving as strong generalist models for everyday
text. Like the Qwen3 embedding models they are based on, they are multilingual (with a focus on English and Chinese) and rank
strongly on embedding leaderboards for their size, often punching above their weight class. They are released under the liberal
Apache-2.0 license. They are a good option for users who want strong multilingual embeddings and long-context support without the
compute cost of a multi-billion-parameter model. In CPU-only mode only the 0.6-billion parameter version is offered.

### What are the Harrier (Microsoft) Embedding Models?
The Harrier embedding models (officially named harrier-oss-v1) were released by Microsoft in March 2026 under the permissive MIT
license. This program offers two sizes: a 270-million parameter version that produces 640-dimensional embeddings, and a
0.6-billion parameter version that produces 1024-dimensional embeddings. Both are well suited to everyday hardware and are used
here with an 8,192-token maximum sequence length. They are instruction-tuned embedding models, meaning a short instruction is
automatically added to your search queries; this program handles that for you, so no special setup is required on your part.

### Are the Harrier embedding models good for multilingual or non-English text?
Yes -- multilingual quality is the standout strength of Microsoft's Harrier embedding models. They were trained across roughly 94
languages, including Arabic, Chinese, French, German, Hindi, Japanese, Korean, Russian, Spanish, and many more, and they achieve
top results on multilingual benchmarks such as the Multilingual MTEB (MMTEB). A particular strength is cross-lingual retrieval:
they work well even when you ask a question in one language and the matching text was embedded from a document written in a
different language -- for example, asking in English and retrieving passages originally written in Japanese or Spanish. This makes
the Harrier family an excellent choice when your documents or questions span more than one language. If your collection is heavily
non-English, the Harrier models are among the strongest options in this program; for English-only collections the lighter BGE,
Intfloat, or ModernBERT models may give you similar quality at a lower compute cost.

### What is the Jasper Token Compression Embedding Model?
Jasper-Token-Compression-600M is a 0.6-billion parameter embedding model released in November 2025 under the permissive MIT
license by Dun Zhang, the author of the earlier Stella and Jasper embedding models. It is built on Qwen3-Embedding-0.6B and was
trained to mimic two much larger embedding models, Qwen3-Embedding-8B and QZhou-Embedding, followed by additional training to
improve search results. It produces 2048-dimensional embeddings and scores close to the 8-billion parameter Qwen3 model on the
English and Chinese MTEB benchmarks. Its distinguishing feature is token compression: before a chunk longer than 80 tokens reaches
the model's attention layers, everything after the first 80 tokens is averaged down to about half as many tokens. This makes it
noticeably faster than a traditional 0.6-billion parameter model, especially for longer chunks, while using less memory. It
supports English and Chinese, is offered in CPU-only mode, and, like the Qwen3 models, automatically adds a short instruction to
your search queries. It was trained on texts of up to roughly 1,000 tokens, so it works best with normal chunk sizes. On older
NVIDIA GPUs that do not support bfloat16 it runs in full precision even when Half precision is turned on.

### What is the Yuan Embedding Model?
Yuan-embedding-2.0-en is a 0.6-billion parameter embedding model designed specifically for English text retrieval and released
under the liberal Apache-2.0 license by IEITYuan, the team behind the Yuan family of language models. It is built on Alibaba's
Qwen3-Embedding-0.6B and was further trained for search using carefully filtered training examples and questions rewritten by the
team's own Yuan2 language model. It produces 1024-dimensional embeddings and is used here with an 8,192-token maximum sequence
length. On the English retrieval benchmark it outscores even the 8-billion parameter Qwen3 model while needing only the compute of
the 0.6-billion parameter Qwen3 model, which makes it an excellent choice for English-only collections; for other languages the
multilingual models in this program are better suited. Like the Qwen3 models, it automatically adds a short instruction to your
search queries, and it is offered in CPU-only mode.

### What are the F2LLM (CodeFuse) Embedding Models?
F2LLM-v2 is a family of fully open, general-purpose embedding models released by CodeFuse under the liberal Apache-2.0 license;
besides the models themselves, CodeFuse publishes their training data and training code. They are built on the Qwen3 architecture
and were trained on about 60 million publicly available examples covering more than 200 languages, with particular emphasis on
languages that most embedding models handle poorly. This program offers two sizes: a 1.7-billion parameter version that produces
2048-dimensional embeddings and a 4-billion parameter version that produces 2560-dimensional embeddings, both used here with an
8,192-token maximum sequence length. They are especially strong at searching programming code and technical documentation -- the
4B version scores within a point of the 8-billion parameter Qwen3 model on the code retrieval benchmarks -- while remaining good
general-purpose models. Like the Qwen3 models, they automatically add a short instruction to your search queries. In CPU-only mode
only the 1.7B version is offered, and it takes about two and a half times as long as the 0.6-billion parameter models.

### What is the GeeVec Lite Embedding Model?
GeeVec-Embeddings-1.0-Lite is a lightweight multilingual embedding model released by GeeVec under the liberal Apache-2.0 license.
It is built on a Qwen3-style model with only 12 layers and about 366 million parameters, yet as of April 2026 it was the
top-scoring model under one billion parameters on the multilingual retrieval benchmark (MMTEB), where it matches the 8-billion
parameter Qwen3 model. It produces 4096-dimensional embeddings, as many as the largest models in this program, so the vectors in
its databases take about four times the space of a 1024-dimension model's, but it is fast: on a CPU it creates databases in
about half the time of the 0.6-billion parameter models. It is used here with an 8,192-token maximum sequence length and,
like the Qwen3 models, automatically adds a short instruction to your search queries. The model can also specialize in code or
reasoning searches, but this program uses its general-purpose mode, which suits most documents. On older NVIDIA GPUs that do not
support bfloat16 it runs in full precision even when Half precision is turned on.

### What are the ModernBERT (Free Law Project) Embedding Models?
These embedding models were fine-tuned by the Free Law Project, a non-profit focused on legal data, and are built on ModernBERT, a
modernized successor to the original BERT architecture that offers faster inference and a longer context. This program offers two
variants: one tuned for a 512-token maximum sequence length and one for an 8,192-token maximum sequence length, both producing
768-dimensional embeddings. Because they were fine-tuned on legal text, they can be especially effective for embedding court
opinions, statutes, and other legal documents, though they also work well as general-purpose English embedding models. They are
released into the public domain under the CC0 license.

### What is the Scrape Documentation tool?
Scrape Documentation automatically downloads documentation from online sources to build vector databases without manual copy-pasting.
In the Scrape Documentation section of the Tools tab, select a documentation source from the dropdown menu (many common libraries
are pre-configured; type in the menu's search box to find one quickly) and click "Scrape." Each running scrape gets its own row
showing how many pages it has saved and how long it has been running, with buttons to cancel it or open its folder, and up to six
scrapes can run at the same time. Scraped content is stored in the Scraped_Documentation folder, one subfolder per source. Once
complete, you'll need to add these files to a vector database through the Create Database tab - the scraper only retrieves and
saves the docs but doesn't vectorize them.  Sources you have already scraped are marked "scraped" in the menu, and scraping one again
asks whether to Resume (skip the pages already saved), Start Fresh (delete them and start over), or Cancel. If a website starts
limiting requests, the row says so and keeps the pages saved so far; scrape it again and choose Resume to continue. This feature is
particularly useful for creating searchable knowledge bases from official documentation for technical Q&A using the VectorDB-Plugin.

### How do I test vision models on images?
The Test Vision Models section of the Tools tab lets you preview how vision models describe your images before adding them to a
database. It offers two tests. (1) Your images and chosen model: add image files in the Create Database tab and choose your vision
model in Settings; the section shows how many images you added and which model is chosen. Click "Describe" to have that model
describe every image without creating a database. When it finishes it reports the average and longest description length and opens
a text file with every description. Keep your chunk size above the longest description so each description fits in one chunk.
(2) One image, several models: click the "Choose an image" box to pick an image, use the Models menu to tick the models to compare
(they're listed with VRAM requirements, and in CPU-only mode the GPU-only models are greyed out and marked "requires GPU"), and click
"Compare." The models run one at a time; each one shows its status and, when done, its description length and time, and you can
cancel between models. A comparison file with each model's description, length and speed opens when it finishes, and the "Open
results" link reopens it. This helps you balance quality versus speed when selecting a vision model.

### What is Optical Character Recognition?
Optical character recognition (aka OCR) refers to whether a .pdf file has a text layer embedded within it representing the actual text
in the document.  The exact structure of the .pdf file format in general is beyond the scope of this tutorial, but generally a .pdf
will have a "glyph" layer that contains the visual representations of text as we commonly understand them being in different "fonts" or
other representations and styles.  The "text layer" refers to a text representation of these common glyphs that a .pdf may or may not
have, which is unseen but which is ultimately extracted when text is extracted from a .pdf document.  If a .pdf does not have this text
layer then text cannot be extracted from a .pdf unless OCR has been done on it, which you can do with this program.  To do so, go to
the Tool Tab, select a .pdf, and perform OCR.  You can Ask Jeeves for more details regarding this if need be.

### How can I extract text from scanned PDFs with OCR?
The OCR tool, found in the Tools tab, turns scanned, image-only PDFs into searchable PDFs. To use it:
(1) Go to the "Optical Character Recognition" section in the Tools tab.
(2) Choose an OCR engine with the Engine switch. RapidOCR is selected by default; Tesseract is also available.
(3) Click the "Choose a PDF" box to select your scanned PDF (the tool accepts PDF files only); its page count appears next to it.
(4) Click "Run OCR" to start extracting text. A progress bar shows how many pages are done.
When processing is complete, the tool saves a new PDF with an "_OCR" suffix in the same folder as the original. It looks the same
as the original but has an invisible, searchable text layer, and the "Open PDF" link in the completion message opens it. Add that "_OCR" PDF to
your vector database using the Create Database tab. RapidOCR also reports quality notes when it finishes, such as low-confidence
pages worth reviewing, pages with visible content but no text, and pages it rotated to read. OCR accuracy depends on the clarity
and quality of the scan, so review the results carefully when accuracy is critical.

### What do the Database Backup and Compare GPUs sections of the Tools tab do?
The bottom of the Tools tab has two small sections. Database Backup: click 'Back Up' to copy the entire Vector_DB folder to a
backup (this overwrites any existing backup), and 'Restore' to overwrite your current databases with that backup; both ask for
confirmation first because they are destructive. The section also shows how many databases you have and names any that are not in
the backup yet. Compare GPUs: choose the least and most VRAM (in GB) and click 'Compare' to open a list of the graphics cards in
that range, showing each card's architecture, release year, compute capability, VRAM and memory type, with a bar comparing its CUDA
cores -- useful for deciding which GPU can run a given model. You can filter the list by name or architecture and sort it by name,
compute capability, VRAM or CUDA cores, and your own GPU is highlighted when it is in the range. Press Escape or click 'Close' to
return to the tools.

### What is Ask Jeeves and how do I use it?
Ask Jeeves is the help assistant built into the VectorDB-Plugin. Click 'Jeeves' in the menu bar at the top of the main window and
the Ask Jeeves window opens, with the familiar Jeeves picture at the top. Jeeves searches this user guide with the
bge-small-en-v1.5 embedding model, so download that model from the Models Tab before the first use. Choose a model from the
'Model' menu; the first time, the model is downloaded, and the status then says it is ready. Type a question in the box at the
bottom, for example "How do I add a PDF to my database?" or "What does chunk overlap mean?", and press Enter or click 'Ask'.
Jeeves looks through the user guide and writes the answer in the conversation, and 'Show sources' under an answer lists the
passages of the guide it used along with their similarity scores. 'Clear' starts the conversation over, and 'Eject' unloads the
model to free the memory it uses. Now and then Jeeves offers to recite a poem; click 'Yes, please' to pick one from the list or
'No, thank you' to carry on. If Jeeves does not respond or appears broken, please report it on GitHub. And yes, the name is a
playful reference to the classic "Ask Jeeves" search engine.

### Can Jeeves read answers aloud?
Yes. Click 'Speak Response' under an answer or a recited poem in the Ask Jeeves window to hear it read aloud with the Kokoro text
to speech model, and click 'Cancel Playback' to stop. The 'Voice' and 'Speed' menus at the bottom of the window choose which voice
Jeeves uses and how fast it speaks. If the Kokoro model has not been downloaded yet, the program offers to download it when you
open Jeeves.

### What are the InternVL3 Vision Models?
InternVL3, released in April 2025, is an advanced open-source multimodal LLM series trained natively on interleaved text, image,
and video data. It follows a ViT-MLP-LLM architecture with vision encoders up to 6B parameters and integrates with LLMs like
InternLM 3 and Qwen2.5. A major innovation is Variable Visual Position Encoding (V2PE), which enhances long-context visual
reasoning by using finer positional increments for visual tokens. The model employs Native Multimodal  re-Training, combining
language and vision learning in one stage, improving performance without separate alignment stages. InternVL3 also introduces
Mixed Preference Optimization and uses dynamic image tiling, JPEG compression, and over 300K instruction-following samples for
training. A Visual Process Reward Model improves inference via best-of-N reasoning chains. Empirically, InternVL3 achieves top
scores across benchmarks like MMMU, MathVista, and OCRBench, outperforming previous models at all scales. It extends capabilities
beyond traditional multimodal reasoning to tool use, 3D perception, GUI interaction, and industrial analysis.

### What are the Liquid-VL Vision Models?
The Liquid-VL vision models are Liquid AI's LFM2-VL family of vision-language models, built on the same efficient LFM2 backbone as
the LiquidAI chat models. They are designed for fast, low-memory image understanding on consumer hardware and edge devices, and
this program offers them in 480M, 1.6B, and 3B parameter sizes; in CPU-only mode only the 480M size is offered. Like the other vision models in this program, they generate a text
description of an image that can then be embedded into a vector database. Because of their small size and speed they are a good
first choice for users who want to caption a large number of images without a high-end GPU, though as always it is recommended to
test them in the Tools Tab against the larger vision models to compare quality. They are released under Liquid AI's open LFM
license.

### What are the Granite Vision Models?
Granite Vision is IBM's enterprise-focused vision-language model optimized for visual document understanding, and this program
uses the Granite Vision 3.2 (2B) version. It has around 3 billion parameters and uses a SigLIP vision encoder, a two-layer GELU-
activated MLP connector, and an instruction-tuned Granite language model. Trained on millions of images and tens of millions of
instructions using public and synthetic data, Granite Vision excels at layout parsing, text recognition, and UI analysis,
especially for charts and tables. It matches or surpasses models like Phi3.5v and InternVL2 on document benchmarks such as DocVQA,
ChartQA, and TextVQA. The model, based on the LLaVA architecture, is open-source under the Apache 2.0 license and supports
commercial use, making it a strong choice for document-focused vision-language tasks.

### What are the Qwen VL Vision Models?
The Qwen VL vision models are the vision-language models in the Qwen family, and this program offers models from both the
Qwen2.5-VL and the newer Qwen3-VL generations in several sizes (2B, 3B, 4B, and 7B). They excel at visual understanding tasks such
as object recognition, text and chart analysis, and document parsing. They use a ViT-based vision encoder with window attention
and dynamic-resolution training, which allows precise visual localization and robust multimodal reasoning across flexible image
sizes. The Qwen VL models consistently rank among the strongest open vision-language models at their respective sizes and resist
hallucination relatively well. They integrate tightly with the underlying Qwen language models, sharing their tokenizer and text
processing while extending them with specialized vision handling. In this program they are used to generate text descriptions of
images that are then embedded into a vector database.