FROM mlrun/mlrun-gpu

# Update apt-get to install ffmpeg (support audio file formats):
RUN apt-get update -y
RUN apt-get install ffmpeg -y

# Install demo requirements:
RUN pip install -U mlrun
RUN pip install -U git+https://github.com/huggingface/transformers.git
RUN pip install tqdm mpi4py
RUN pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install pyannote.audio faster-whisper bitsandbytes accelerate datasets peft optimum
RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
RUN pip install langchain openai
RUN pip install git+https://github.com/suno-ai/bark.git
RUN pip install streamlit st-annotated-text spacy librosa presidio-anonymizer presidio-analyzer nltk flair
RUN python -m spacy download en_core_web_lg

# Align onnxruntime to use gpu:
RUN pip uninstall -y onnxruntime-gpu
RUN pip uninstall -y onnxruntime
RUN pip install onnxruntime-gpu
