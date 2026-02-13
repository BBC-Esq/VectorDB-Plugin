### What is the VectorDB-Plugin and what can it do?
VectorDB-Plugin is a program that lets you build a vector database from your documents (text files, PDFs, images, etc.) and use it
with a large language model for more accurate answers. This approach is known as Retrieval Augmented Generation (RAG) – the software
finds relevant pieces of your data (embeddings) and feeds them into an AI chat model so the answers are based on your own content.
In simple terms, VectorDB-Plugin "supercharges" a language model by giving it a memory of your files, which improves the factual
accuracy of responses. You can search your database by asking questions in plain language, and the program will retrieve matching
chunks from your data and have the chat model incorporate them into its answer.

### What are the system requirements and prerequisites?
System Requirements for VectorDB-Plugin include a Windows operating system (Windows 10 or 11) and Python (version 3.11 or 3.12 is
recommended). You should also have Git installed (with Git LFS for handling large model files) and Pandoc (a document converter).
If you plan to use GPU acceleration or certain models, you'll need a suitable C++ compiler and possibly Visual Studio build tools
on Windows. An NVIDIA GPU is optional but can greatly speed up embedding and model inference (the program will also work on CPU,
just more slowly). Make sure you have sufficient disk space for storing models and databases – vector models and chat models can
be several hundred MBs to a few GBs each.

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
Third, run the setup script with the command "python setup_windows.py". It is important to note that this progam is only supported
on Windows at this time.  Lastly, you run the program by using the command "python gui.py". A window should open with this program's
graphical user interface.

### How do I download or add embedding models?
The Models Tab lets you browse and download embedding models.  Models are grouped by providers with properties listed for each
embedding model.  To download a model, click the radio button next to the modle you want to download and then click
"Download Selected Model".  This will save the necessary model files to the "Models/Vector/" folder if you want to inspect them. The
Original Precision of an embedding model is the original floating point format that a model was saved to by the creator - e.g. float32,
float16 etc. The Parameters of an embedding model refers to how many parameters a particular model has - e.g. 109m means 109 million
parameters. The Dimensions of an embedding model refers to how complex of embeddings that a particular model created.  More complexity
means the higher quality generally within the same embedding model family.  For example, dimensions such as 768 or 1024. The Max
Sequence of an embedding model refers to the maximum amount of tokens that an embedding model can process at a given time.  The size
of a model refers to the size on disk.

### How do I query the database for answers?
Select the database you want to query from the dropdown menu. Choose a backend model for answering Local Models built-in AI Kobold
LM Studio or ChatGPT each option uses different AI systems to generate responses. Enter your question in natural language in the
text box for example what does the quarterly report say about revenue. If you only want to see the retrieved information without
AI processing check the chunks only box. Click Submit Question the system searches your database for relevant content using semantic
similarity. The results will display both the retrieved chunks so you can verify sources and a complete answer generated by your
chosen AI model based on those chunks. You can continue with follow-up questions or new queries as needed.

### Which chat backend should I use?
The program offers four options for generating answers from your database content. The Local Models backend uses chat models downloaded
directly from Huggingface and does not rely on any exernal program. The Kobold backend connects to a Kobold server that has already
loaded a chat model.  You must download Kobold prior to using this backend and set it up correctly.  The LM Studio backend is similar
in that it requires downloading an external program prior to using it and setting it up correctly.  The ChatGPT backed uses the API
from Openai and connects to one of several models. You must first create an account with Openai and get an API key, which must then
be entered into this program from the menu at the top.  Unlike the other backends, the ChatGPT backend cannot run without an Internet
connection.

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

### What local chat models are available and how can I use them?
The "local models" option within the Query Database Tab downloads chat models directly from Huggingface and requires no external program.
You can select a local model from the pulldown menu and when you use it for the first time it will automatically download the model and
it can then be used thereafter for subsequent queries.  Please note that certain models are "gated," which means that you must first
enter a huggingface access token.  You can create an access token on Huggingface's website and then enter it within the "File" menu
within this program in the upper left. You must do this before trying to use certain "gated" "local models".  To get a Huggingface
access token you must create a huggingface account and then go to your profile.  On the left-hand side will be an "Access Tokens"
option.  Then in the upper right is a "Create new token" button.  Check the box that says "Read access to contents of all public
gated repos you can access" then click "Create token."

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

### What does the chunks only checkbox do?
Typically when you submit a query within the Query Database Tab it connects to your chosen backend to get a response from a chat model.
However, if you check the "chunks only" checkbox it will only return the chunks retrieved from the vector database.  This is good
for seeing verbatim what would be sent to the chat model backend in case you need that level of detail, but the primary purpose is to
enable users to see the quality of the chunks that they are creating.  For example, it gives you an idea of whether the chunks size
setting you chose is sufficient, or it gives users an idea of whether a particular embedding model is creating a high enough quality
of embeddings for their particular use case.

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
to choose as high a quality of embedding model as possible.  Also, if compute resources are limited make sure and check the "half"
checkbox within the Settings Tab.  This will run the embedding model in either bfloat16 or float16 (commonly referred to as half
precision).  Studies show that there is very little loss in quality between full precision and half precision.  Lastly, always use
"cuda" within the Settings Tab when creating embeddings if you have a GPU.

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
will get, but not always.  Smaller vision models that are newer sometimes outperform larger but older vision models.  Also, some
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
within the Query Database Tab; simply click the "Voice Recorder" button, record your question, and it will be output to the query box.
Secondly, whisper models are used to create transcriptions of audio files that can subsequently be entered into a vector database.
You can create these transcriptions within the Tools Tab.  This will create a transcript of an audio file, which you will see within
the Create Database Tab before creating the vector database.

### How can I record my question for the vector database query?
To transcribe a spoken question, go to the "Query Database" tab, click the "Voice Recorder" button to begin recording
and then speak clearly. Click the button again to stop recording, and the transcribed text will appear in the question box.

### How can I transcribe an audio file to be put into the vector database?
To transcribe an audio file, navigate to the Tools tab, select an audio file (most file formats are supported such as .mp3, .wav,
.m4a, .ogg, .wma, and .flac) and click the Transcribe button. After the transcription is complete you can see it in the
"Create Database" tab and it will be entered into the vector database when you create it.  The transcribing functionality uses
the powerful `WhisperS2T` library with the `Ctranslate2` backend.  Make sure to adjust the "Batch" setting when transcribing an
audio file depending on the size of the whisper model you choose. Increasing the batch size can improve speed but demands more
VRAM, so care should be taken not to exceed your GPU’s capacity.

### What are the distil variants of the whisper models when transcribing and audio file?
Distil variants of Whisper models use approximately 70% of the resources of their full counterparts and are faster with very little
loss in quality.

### What whisper model should I choose to transcribe a file?
When transcribing an audio file in order to put it into a vector database it is generally recommended to use as high a quality of
a whisper model as your hardware will support.  The quality of a whisper model is determined by a few factors.  Firstly, its size
is the most important factor - e.g. large versus medium versus small.  Secondly, the precision of the model that you use.  This
program allows you to choose float32 for the highest qualityy or bfloat16 or float16 (i.e. half precision).  In general, using
half precision results in about 95% of the quality of float32 for half the compute resources needed.  Lastly, some of the whisper
models come in "distil" variants that have certain layers of the model removed.  Again, this typically gives approximately 95%
of the non-distil variant for half the compute resources.  It is highly recommended to test the various whisper models on a small
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
The "Settings" Tab contains most of the settings for LM Studio, querying the database, creating the database, the text to speech
functionality, and the vision models.  Please ask me a question about the specific setting or group of settings you're interested in?

### What are the LM Studio Server settings?
When using LM Studio as the chat model backend you can adjust a few settings from within the Settings Tab.  In general, however,
the LM Studio program has all the settings that you should adjust.  For purposes of this program you can adjust the port to match
what you set within LM Studio.  Also, there is a checkbox you can check to see the thinking process if the model you are running
within LM Studio has chain of thought.

### What are the database creation settings?
The Device setting allows you to choose either CPU or CUDA when creating a vector database.  It is always recommended to choose
CUDA if available.  The Chunk Size setting determines the size of the chunks of text that your documents will be broken into before
being turned into embeddings.  It is crucial to remember that this setting is in number of characters, not tokens, and that you must
keep the chunks within the maximum sequence length of the embedding model you are using, as expressed in tokens, and which you can
see within the Models Tab.  Remember, each tokens is approximately 3-4 characters.  The Overlap setting refers to how many characters
at the beginning of a chunk are from the preceding chunk.  When a document is processed sometimes it is split in the middle of an
important concept and this setting ensures that there is an overlap to avoid losing meaning.  A good rule of thumb is to set the
Overlap setting to 30-49 percent of the Chunk Size setting.  The half-precision setting, if checked, will run the embedding model
in half precision resulting in a slight reduction in quality but half the compute resources.

### What are the database query settings?
Within the Settings Tab you can adjust several settings when searching a vector database.  The Device setting allows you to choose
between CPU and CUDA.  In contrast to creating a vector database, it is recommended to always use CPU.  The Similarity setting sets
a threshhold of relevance for a chunk of text before it will be returned as a result.  You can set a value between zero and 1.  A
higher value will result in more chunks being returned but you should never use 1.  The Contexts setting determines the maximum
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
number of chunks specified in the Contexts setting.  A higher Similarity setting means that more chunks will possibly be returned.
A good default value is .8, but do not go above 1.

### What is the search term filter setting?
Within the Settings Tab the Search Term Filter setting allows you to require that any chunks returned contain the specified search term.
It is not case-sensitive, but it does require an exact match.  For example, if you specify “child” it will only return chunks that
include the term "child" somewhere in it.  This would not include chunks that have the word "children" in it, however, since it
requires a verbatim match.  With that said, since it is not case-sensitive it would also include chunks with "Child" in them.  This
setting is especially useful when you know that a relevant chunk has a certain key word in it; otherwise, it is best to leave this blank.
Click the Clear Filter button to clear any filters.  Lastly, it is important to understand that this setting only applies after both
the Similarity and Contexts settings.  Therefore, if you set those settings too low you might not receive any chunks with your specified
search term.

### What is the File Type setting?
Within the Settings Tabe the File Type setting allows you to limit the chunks that are returned based on whether they originated from
a particular type of file.  Current options include images, documents, audio or all files.  It is best to use the all files option
unless you are sure that the chunks you are looking from originated from a particular type of file.

### What are text to speech models (aks TTS models) and how are they used in this program?
Text to speech models (TTS) are large language models that were specifically trained to take text as input and output audio in a spoken
voice format.  This program allows you to use TTS models to speak the response that you get after querying the vector database.

### What text to speech models are availble in this program to use?
You can choose various text to speech models within the Settings Tab.  The current options are Bark, WhisperSpeech, ChatTTS, and
Google TTS.  The Bark backend has a Normal size that produces slightly higher quality and and a Small version that uses fewer
resources.  With Bark you can choose different speaker voices such as v2/en_speaker_6, which is usually considered the highest
quality or v2/en_speaker_9, which is the only female voice.  Using Bark requires a GPU, however. The WhisperSpeech backend consists
of two models that you choose within the Settings Tab, both of which determine the quality.  Experiment with both to find a setting
that works with your hardware.  WhisperSpeech, like Bark, requires a GPU but is generally less compute intensive than Bark at roughly
the same quality.  The ChatTTS backend is also a good option that can be run both on GPU or CPU.  It produces audio slightly less
quality than Bark or WhisperSpeech.  Lastly, the Google TTS backend is the least compute intensive.  However, it does not require a
GPU and will instead connect to a free online Google service that provides TTS.

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
on consumer GPUs and offers Python bindings, making it one of the most practical and expressive TTS options for developers aiming to
build natural, back-and-forth conversational agents in English or Mandarin.

### What is the Google TTS text to speech?
Google TTS (Free Tier) offers industry-leading neural speech synthesis via a cloud API, backed by WaveNet and Neural2 models that
produce ultra-clear, stable, and emotionally nuanced voices across 380+ voices and 50+ languages. Although not open-source, it
provides a generous free tier (up to 4 million characters/month for standard voices and 1 million for WaveNet), making it highly
accessible for small-scale use. Developers can fine-tune pronunciation and pacing using SSML, and even select expressive “Newscaster”
or “Lively” voice styles. With near real-time performance via a fast cloud API and seamless Python integration, Google TTS is the gold
standard for high-quality, multilingual TTS — ideal for production-ready applications where speech quality, reliability, and global
language support outweigh the need for open-source control.

### What is the Chatterbox text to speech?
Chatterbox, developed by Resemble AI, is a cutting-edge open-source text-to-speech (TTS) model that sets a new standard in voice
synthesis. Released under the permissive MIT license, it offers developers and creators unparalleled freedom to use, modify, and
distribute the software. Chatterbox's standout features include zero-shot voice cloning, allowing it to mimic any voice with just a
few seconds of reference audio and emotion exaggeration control. Its alignment-informed inference ensures ultra-stable and
natural-sounding speech, making it ideal for real-time applications like voice assistants and interactive media. In blind evaluations,
Chatterbox has been consistently preferred over proprietary models like ElevenLabs, highlighting its superior performance in
generating high-quality, expressive speech. With its combination of advanced features, open-source accessibility, and exceptional
speech synthesis quality, Chatterbox stands out as a powerful tool for developers seeking a versatile and ethical TTS solution.

### Which text to speech backend or models should I use
Generally it's recommended to experiment with each to your liking.  However, in general Bark and WhisperSpeech produce the highest
quality results, Chat TTS and Chatterbox are below them but can be run on GPU as well as CPU, and Google TTS is comparable to
Chat TTS in terms of quality but requires an Internet connection.

### Can I back up or restore my databases and are they backed up automatically
When you create a vector database it is automatically backed up.  However, if you want to manually backup all databases you can go
to the "Tools" tab and click the Backup All Databases button.  Likewise, you can restore all backed up databases within the Tools Tab.

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
You can always check the VRAM and speed for local models within the Tools Tab by clicking the "Chat Models" button, which will display
a nice chart.  However, in general smaller models like Qwen 3 - 0.6b deliver exceptional speed at over 200 characters per second while
requiring minimal VRAM (1.3GB), mid-range models in the 2-9 billion parameter range offer a sweet spot for most users, with speeds
ranging from 150-400 characters per second and VRAM usage between 2.5-9.5GB. Notable standouts include the GLM4-Z1 - 9b, which achieves
an impressive 395 CPS while using under 10GB VRAM, and the Exaone models, which consistently deliver faster performance than
similarly-sized alternatives. For users with high-end GPUs, the larger 24-32 billion parameter models provide enhanced reasoning
capabilities at the cost of reduced speed (95-140 CPS) and substantial VRAM requirements (15-20GB).

### What are the speed and VRAM requirements for the various vision models?
Vision models demonstrate a clear inverse relationship between speed and model size, with smaller models delivering significantly
faster image processing while larger models provide enhanced accuracy at the cost of reduced throughput. The fastest performers are
models like Ovis2 - 2b at 312 characters per second (CPS) and InternVL2.5 - 1b (289 CPS) with relatively low VRAM usage of 2.3-5.8GB.
Florence-2 models, which can be run on a CPU, showcase interesting trade-offs.  For example, Florence-2-Base achieves an impressive
971 CPS on GPU with only 2.6GB VRAM, CPU-only operations drops performance to 157 CPS. Mid-range models like
Granite Vision - 2b (218 CPS, 4.1GB) and THUDM glm4v - 9b (201 CPS, 9.8GB) offer balanced performance for most use cases. The
largest models such as Qwen VL - 7b (174 CPS, 9.6GB) require more resources.

### What are maximunm context length and maximum sequence length and how to they relate?
Each embedding model has a maximum sequence length, and exceeding this limit can result in truncation. To avoid this, regularly
check the maximum sequence length of the model and adjust your settings accordingly. Reducing chunk size or the number of contexts
can help stay within these limits. Maximum "context length" refers to chat models and is very similar to maximum sequence length.
The key thing to understand is that the chunks you put into the vector database should be within the max sequence length of the
vector or embedding model you choose and the maximum context or chunks you retrieve from the vector database multiplied by their
length should stay within the chat model's context length limit.  And make sure to leave enough context for a response.

### What is the scrape documentaton feature?
Within the Tools tab you can select multiple python libraries and scrape their documentation.  Multiple .html files will be downloaded
and you can subsequently create a vector database out of them.  Larger more complex libraries can take a significant amount of time
to scrape to make sure you have a stable Internet connection.

### Which vector or embedding models are available in this program?
All of the embedding models that this program uses are listed on the Models Tab.  You can click on a hyperlink for each one to find
out more information.  The embedding models sometimes change as different versions of this program are released and newer and better
embedding models are released.  This program vets all embedding models, however, before including them for usage.

### What is the manage databaes tab?
The Manage Databases Tab allows you to see all of the vector databases that you have created thus far and what documents are in them.
Select the database you want to view from the pulldown menu and you can see the files that have been embedded.  Also, you can
doubleclick any of the files to open it in your system's default program.  When a vector database is created the location of the
original file is saved as metadata.  As long as you haven't moved the original file on your computer, this metadata will be used to
locate the file and open it in the default program on your system.

### How can I create a vector database?
Go to the Create Database tab and choose the files that you want to add to the vector database.  If you select any file types that are
not supported, the program will let you know and give you an option to automatically exclude them.  Remember, you can repeat this
process as many times as you with.  Also, you can choose whether to select all of the files in a particular directory or simply
choose individual files.  To add audio transcriptions to the database you must first transcribe audio files individually, which can
only be done within the Tools Tab.  To input descriptions of images into the vector database choose an appropriate vision model from
the Settings Tab.  Any images you select will then automatically be processed by that vision model when you create the database.
Remember, make sure and adjust the database creation settings within the Settings Tab before creating the database.

### Can I use images and audio files in my database?
You can use both images and audio in your vector database. Images: When you add image files (like PNG, JPG, BMP), the selected vision
model creates a text description of each image, which is then embedded like a regular text document. For example, a chart might be
described as “A line graph showing revenue over time with an upward trend.” You can then search with queries like “What does the
revenue trend look like?” and retrieve the image. Make sure you choose a vision model in the Settings Tab first and use the Test
Vision Models tool within the Tools Tab ot preview captions before using a particular model. Audio: You can't add audio files directly,
but you can use the Transcribe File tool (powered by OpenAI’s Whisper model) to convert audio to text. This transcript can then be
added like any other document during database creation. If you try to upload audio directly, the program will prompt you to transcribe
it first. By converting images and audio to text, the system supports rich, multi-modal queries — as long as content is processed
correctly.

### What chat models are available with the local models option?
Within the Query Database Tab if you choose the local models option it will allow you to use a specified number of chat models that
will be downloaded directly from the Huggingface website.  All of these models have been specifically chosen for their strength
in question answering using contexts provided by a vector database.  Please ask about a particular family of chat models for more
information or you can visit the repository for the various chat models on Huggingface for more detailed information.  The available
chat models that this program uses sometimes changes as newer models come out with higher capabilities.  All chat models that are
added or removed will be noted in the release notes on Github for the record.

### What are the Qwen 3 Chat Models?
Qwen3 is the latest release in the Qwen family of large language models.  they come in six sizes ranging from .6 billion parameters
to 32 billion gparameters and can be used under the Apache 2.0 license.  A key innovation with the Qwen3 series is the hybrid
"thinking" versus "non-thinking" modes that are available.  This program has opted to use the thinking mode for all Qwen3 models as
it tends to produce the best results for retrieval augmented generation purposes.  The Qwen3 models are multilinguals and are touted
as supporting up to 119 languages.  They were trained on approximately 36 trillion tokens, which is double the amount used for Qwen 2.5.
Qwen has consistently created some of the best open source and free models available and they are a staple of this program.

### What are the Granite 3.3 Chat Models?
The Granite 3.3 chat models are the latest in the Granite series developed by IBM and are released under the Apache 2.0 license.
They are "thinking" or "reasoning" models and have improved upon prior iterations in this regard.  The Granite models were trained
on synthetically generated datasets for long-context tasks and are good for retrieval augmented generation purposes.  Version 3.3
of the models exceed the performance of Granite 3.1 and 3.2 by a significant margin.

### What are the GLM-Z1 Chat Models?
The Z1 family of chat models are created by THUDM and demonstract strong performance across a wide range of tasks, including retrieval
augmented generation.  The benchmarks show that they are particularly strong in general-purpose question answering across a wide range
of domains - e.g. science, math, and other areas.  They come in a 9 billion parameter and 32 billion parameter variants and are a
staple of this program due to their high quality on question answering tasks.

### What is the Mistral Small Chat Model?
The Mistral Small chat model is the third iteration of Mistral models and has 24 billion parameters.  It is released under the
Apache 2.0 license for liberal usage.  Compared to larger models such as LLaMA 3.3 with 70 billion parameters and Qwen 2.5 with
32 billion parameters, the Mistral Small 3 model achieves comparable quality results across a wide range of benchmarks.  What is
unique about the Mistral Small 3 model is its size of 24 billion parameters, which often sits in the sweet spot for VRAM usage for
users having 24 gigabytes of VRAM.  Sometimes larger models having 32 billion parameters will exceed the available VRAM with longer
contexts but Mistral Small 3 leaves sufficient VRAM avaialble in such circumstances. Benchmark results also show that it excels at
reasoning, coding, math, and instruction following, oftentimes producing more succinct answers than other similarly sized models.

### What are the MiniCPM-4 chat models?
The MiniCPM-4 chat models are ultra-efficient, open-source LLMs built by the OpenBMB team for edge devices, offered in lightweight
0.5 B-parameter and 8 B-parameter versions.  The 8b variant achieves comparable performance to Qwen3-8B using only 22% of the training
data.  The 0.5B parameter variant, despite having fewer parameters, significantly outperforms Qwen3-0.6B, Llama3.2-1B, and Gemma3-1B.
The 8b variant matches Qwen3 8b and outperforms GLM4-9B, and exceeds larger models such as Gemma3-12B and approaches Phi4-14B.  The
major advantage is on 128k context sequences where it achieves 7x faster decoding than Qwen3-8b due to the fact that it only attends
to ~6k tokens out of 128k (5% sparsity).  These chat models focus on the English and Chinese languages.

### What is the Deepseek R1 Chat Model?
The Deepseek R1 chat model was previously removed from this program, but it has been re-added since Deepseek released a newer and
improved version in late May, 2025.  This new version, based on the Qwen3 architecture, has significantly improved both its reasoning
and generic response tasks and is an excellent choice for retrieval augmented generation.  It claims to rival much larger open source
models such as Qwen 3 32b and Phi-4 14b and even claims to outperform closed-source models such as OpenAI's os-mini (medium) and
Google's Gemini 2.5 Flash, which is quite impressive.

### What are the BGE Embedding Models?
The BGE family of embedding models were created by BAAI and have long been a staple within the embedding community and this program
in particular.  They are well-respected as producing high quality embeddings for reasonable compute resources.  Although they are
over a year old now, they are still regarded as producing quality embeddings for a reasonable compute cost for most use cases.  At
the time of their release they were state of the art for open source and free embedding models.

### What are the Granite Embedding Models?
The Granite family of embedding models were created by IBM and are lightweight embedding models based on the RoBERTa architecture as
opposed to the BERT architecture like most other embedding models.  IBM touts these models as being suitable for "enterprise" use
cases and come in 30.3 and 125 million parameter sizes.  Along with the Snowflake Arctic embedding models, they are one of the fastest
embedding models that this program offers when considered in relation to the quality of embeddings that they produce.  In contrast to
the Snowflake Arctic embedding models, however, they do not rely upon the Xformers library to achieve this, which is not supported by
all graphics cards.  The Granite embedding models were released in early 2025 under the liberal Apache-2.0 license.  This program only
usese the English-trained variations of the models.

### What are the Intfloat Embedding Models?
Similar to the BGE embedding models produced by BAAI, the Intfloat embedding models have long been a staple of high quality embedding
models in the community and this program.  They include "small," "base," and "large' variants for your particular use case.  They offer
high quality embeddings for the compute resources required and often go head-to-head in comparison with the "bge" models from BAAI.
Although they are well over a year old now they still offer high quality embeddings for a reasonable compute cost and many other
embedding models have been built upon the e5 family of models.

### What are the Arctic Embedding Models?
Snowflake's Arctic-embed models are retrieval-optimized text embedding models built on E5-small and E5-large embedding models created
by Intfloat. Despite their relatively modest sizes, these models outperformed larger competitors on several benchmarks.  They are
also significantly faster than similarly sized models due to their reliance on the Xformers library.  These models can, however, be
run with or without reliance on the Xformers library depending on whether a user's hardware supports it.  The Snowflake Arctic embedding
models are also unique in that they have a maximum sequence length of 8192 tokens, which is far greater than the typical 512 token limit
of other embedding models.

### What are the Qwen3 Embedding Models?
Released in June, 2025, Alibaba’s Qwen 3 Embedding family delivers state-of-the-art text embeddings while staying friendly to everyday hardware.  They are based on the popular Qwen 3 chat models but have special training to make them suitable for generating embeddings.
As of June, 2025, they hold the top three ranked spots on the Huggingface leaderboard.  They are primarily trained on English and
Chinese data, but a fair amount of their training data is also from numerous other languages so they can be reliably used for multilingual
embedding tasks as well.  They are released under the liberal Apache-2.0 license. The Qwen 3 family of embedding models comes in three
practical sizes—“small” (0.6 B parameters), “base” (4 B), and “large” (8 B). Even the 0.6 B version outperforms older 7 B embedding models, which is a phenomenal accomplishment while the 8 B model often edges out commercial offerings. All variants support long contexts (up to 32 k tokens). 

### What is the Scrape Documentation tool?
Scrape Documentation automatically downloads documentation from online sources to build vector databases without manual copy-pasting.
Located in the Tools tab, simply select a documentation source from the dropdown menu (many common libraries are pre-configured) and
click "Scrape." The program will fetch all relevant pages, showing progress as it works. Scraped content is stored in
src/Scraped_Documentation/<NameOfDoc>/. Once complete, you'll need to add these files to a vector database through the Create Database
tab - the scraper only retrieves and saves the docs but doesn't vectorize them.  If documentation has been previously scraped, the
entry appears in red, and you'll be warned before overwriting existing data. This feature is particularly useful for creating
searchable knowledge bases from official documentation for technical Q&A using the VectorDB-Plugin.

### How do I test vision models on images?
The Test Vision Models tool in the Tools tab lets you preview how vision models describe your images before adding them to a database.
It offers two main options: (1) Multiple Files + One Vision Model, which tests one vision model on multiple images. First, select
image files in the Create Database tab, then choose your vision model in Settings. Return to Tools and click "Multiple Files + One
Vision Model – Process." The tool generates descriptions for all images without creating a database, showing average description
length to help you evaluate the model's performance.  Single Image + All Vision Models: Compare multiple vision models on one image.
Click this option, select an image, then choose which vision models to test from the dialog (they're listed with VRAM requirements).
The tool will sequentially process your image through each model and produce a comparison showing each model's description and
processing time. This helps you balance quality versus speed when selecting a vision model.

### What is Optical Character Recognition?
Optical character recognition (aka OCR) refers to whether a .pdf file has a text layer embedded within it representing the actual text
in the document.  The exact structure of the .pdf file format in general is beyond the scope of this tutorial, but generally a .pdf
will have a "glyph" layer that contains the visual representations of text as we commonly understand them being in different "fonts" or
other representations and styles.  The "text layer" refers to a text representation of these common glyphs that a .pdf may or may not
have, which is unseen but which is ultimately extracted when text is extracted from a .pdf document.  If a .pdf does not have this text
layer then text cannot be extracted from a .pdf unless OCR has been done on it, which you can do with this program.  To do so, go to
the Tool Tab, select a .pdf, and perform OCR.  You can Ask Jeeves for more details regarding this if need be.

### How can I extract text from PDFs or images with OCR?
The OCR tool, found in the Tools tab, converts image-based documents into searchable text using the built-in Tesseract engine. To use it:
(1) Go to the "OPTICAL CHARACTER RECOGNITION" section in the Tools tab.
(2) Ensure "Tesseract" is selected from the dropdown (it’s usually pre-selected).
(3) Click "Choose PDF" to upload your scanned PDF or image file.
(4) Click "Process" to start extracting text.
Once processing is complete, the tool generates two outputs:
(1) A new PDF file with an "_OCR" suffix that includes the original document along with an invisible, searchable text layer.
(2) A plain text file containing all the recognized text, including page markers like [[page1]].
You can then upload either the OCR-enhanced PDF or the plain text file to your vector database using the Create Database tab. The
tool works best with PDFs, including multi-page ones, but it also supports image files. OCR accuracy varies depending on the clarity
and quality of the input, so it's important to review the results carefully when accuracy is critical.

### What other features does the Misc tab have?
In addition to backup and restore, the Misc tab includes three visualization tools: GPU Comparison Chart: Click the "GPUs" button to
open a chart that compares graphics cards based on performance and memory. You can filter results by VRAM range (e.g., 4–6 GB, 8 GB,
10–12 GB), making it easier to evaluate which GPUs are suitable for running various models. Chat Models Comparison: Selecting
"Chat Models" brings up a chart comparing local chat models, displaying estimated VRAM usage and token generation speeds. Models are
typically color-coded by category (e.g., general use vs. coding), giving you a clear picture of which ones align with your GPU
capabilities. Vision Models Comparison: Clicking "Vision Models" launches a comparison of available vision captioning models,
highlighting their size, VRAM requirements, and performance benchmarks such as processing time per image. All visualizations open
in separate windows using matplotlib. These tools are purely informational, aimed at helping users make informed choices about
model compatibility and system requirements. To return to the application, simply close the chart window.

### What is Ask Jeeves and how do I use it?
Ask Jeeves is an integrated help assistant built into the VectorDB-Plugin, designed to serve as an in-app guide or Q&A tool. You can
access it from the menu bar—look for the "Ask Jeeves" option. When launched, it opens a new window where you can type in questions
about using the program. For instance, you might ask, “How do I add a PDF to my database?” or “What does chunk overlap mean?” Ask
Jeeves will respond with helpful answers sourced from the documentation. Ask Jeeves is ideal for getting quick guidance while
actively using the program, without needing to leave the interface or consult external resources. If the feature doesn’t respond
or appears broken, users are encouraged to report the issue on GitHub, as it may indicate a problem with loading the help content.
Think of Ask Jeeves as your on-demand tutor—just click it, type a plain-English question about the VectorDB-Plugin, and get clear
explanations or step-by-step instructions. And yes, the name is a playful reference to the classic “Ask Jeeves” search engine,
suggesting you can ask it anything!

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

### What are the Ovis2 Vision Models?
Ovis2 launched in January 2025 as a second-generation multimodal large language model optimized for compact sizes (1B and 2B). It
integrates Apple’s AIMv2 vision transformer and supports Qwen2.5 or InternLM 2.5 as its language backend. A key innovation is its
visual embedding table, which structurally aligns image patches with textual tokens using a shared embedding strategy, improving
coherence across modalities. Unlike traditional connector-based MLLMs, Ovis2 maps visual inputs into probabilistic tokens that
interact with a large visual vocabulary (131,072 visual words), allowing for sparse, efficient visual representation. The model is
instruction-tuned on diverse multimodal data, including videos, multilingual OCR, and charts, boosting chain-of-thought reasoning.
Though not trained with quantization, 4-bit GPTQ versions were made available in March 2025. Ovis2 achieves state-of-the-art results
across various benchmarks, including 89.1 on OCRBench and 83.6 on MMBench-V1.1 for the 8B version. Overall, Ovis2’s architectural
advancements enable high performance on vision-language tasks while maintaining efficiency in smaller model sizes.

### What are the Florence-2 Vision Models?
Florence-2, released by Microsoft in June 2024, comes in two sizes—Base (232M parameters) and Large (771M)—and uses a sequence-to-sequence
architecture built on DaViT and Transformer layers. The model is trained on FLD-5B, a dataset with 5.4 billion annotations across
126 million images, created by the automated Florence data engine. Florence-2 integrates visual inputs with textual prompts and excels
in zero-shot tasks, outperforming much larger models like Flamingo-80B on benchmarks such as COCO captioning and TextVQA. It performs
well across multiple levels of granularity, from full images to specific regions and pixels, enabling state-of-the-art performance in
various tasks. Its design allows for multitask learning without the need for separate modules, improving efficiency and simplifying
deployment. Fine-tuning on public datasets further boosts its accuracy and robustness in real-world applications. Unlike traditional
dual-encoder models like CLIP, Florence-2 uses a single Transformer stack with joint vision-text training, accepting both images and
text prompts as input and producing outputs in text or structured formats.

### What are the Granite Vision Models?
Granite Vision is IBM's enterprise-focused vision-language model optimized for visual document understanding released in February 2025.
It has around 3 billion parameters and uses a SigLIP vision encoder, a two-layer GELU-activated MLP connector, and the
granite-3.1-2b-instruct language model. Trained on 13 million images and 80 million instructions using public and synthetic data.
Granite Vision excels at layout parsing, text recognition, and UI analysis, especially for charts and tables, achieving up to 95%
accuracy in chart extraction. It matches or surpasses models like Phi3.5v and InternVL2 on document benchmarks such as DocVQA, ChartQA,
and TextVQA. Unique features include sparse attention-based safety mechanisms and multi-layer feature extraction. The model, based on
the LLaVA architecture, is open-source under the Apache 2.0 license and supports commercial use. Granite Vision consistently outperforms
or matches Phi3.5v and InternVL2 across key benchmarks, highlighting its strong advantage in document-focused vision-language tasks.

### What are the Qwen2.5VL Vision Models?
Qwen2.5-VL is the latest vision-language model in the Qwen family. It excels in visual understanding tasks like object recognition,
text and chart analysis, and document parsing. The model features a streamlined ViT-based vision encoder with window attention,
SwiGLU activations, RMSNorm, and dynamic resolution/frame rate training for video, enhanced by mRoPE in the time dimension. These
architectural updates allow precise visual localization and robust multimodal reasoning. Qwen2.5-VL-7B outperforms peers like
InternVL2.5-8B, MiniCPM-o 2.6, and GPT-4o-mini in multiple benchmarks: Document QA: DocVQA 95.7%, InfoVQA 82.6%, ChartQA 87.3%
Text recognition: TextVQA 84.9%, OCRBench 864, CC_OCR 77.8% General VLU: MMBench 82.6%, MMVet 67.1% Math reasoning: MathVista 68.2%,
MathVision 25.07% It also resists hallucination better than GPT-4o-mini (HallBench: 52.9% vs. 46.1%). The model integrates tightly
with the Qwen2.5 LLM, sharing its tokenizer and text processing, while extending it with specialized vision-language handling and
support for flexible image resolutions.

### What is the GLM-4V-9B Vision Model?
GLM-4V-9B, developed by Zhipu AI and Tsinghua University, is a 9B-parameter bilingual (Chinese/English) multimodal model released
in mid-2024 as part of the GLM (OpenGLM) series. It integrates vision into the pretrained GLM-4 LLM, supporting high-resolution
inputs up to 1120×1120 and enabling general vision-language tasks like image QA, captioning, and reasoning. The model uses standard
attention and likely linear patch embeddings, with training on large multilingual image-text datasets. GLM-4V-9B incorporates Mixed
Preference Optimization (MPO) to enhance chain-of-thought alignment, similar to InternVL. It supports FP16 precision and an 8K context
window, though quantization is not emphasized. Benchmarks show strong performance: it scored 81.1 on English MMBench and 786 on
OCRBench, outperforming many open models and reportedly rivaling or exceeding GPT-4-turbo and Gemini 1.0 Pro on several vision tasks.

### What is the Molmo-D-0924 Vision Model?
Molmo-D-0924 is a 7–8B parameter open-source vision-language model released by the Allen Institute (AI2) in September 2024, as part
of the larger Molmo project. It combines Qwen2-7B as the language backbone with OpenAI’s CLIP-ViT as the vision encoder and is trained
on a proprietary PixMo dataset of 1M high-quality image–text pairs. A key innovation is its support for multi-turn “pointing” in images
via a special OLMo module, allowing the model to interactively highlight regions in response to queries—moving beyond standard text-only
outputs. The model is decoder-only, optimized for interactive use, and runs efficiently on commodity GPUs with FP16 or bfloat16
precision. While users can’t fine-tune quality knobs beyond image size, it offers real-time responsiveness. On benchmarks, Molmo-7B-D
performs between GPT-4V and GPT-4o and achieves state-of-the-art results among similarly sized open models, as confirmed by academic
and human evaluations.