FROM mlrun/mlrun-gpu
RUN apt-get update -y
RUN apt-get install ffmpeg -y
RUN pip install tqdm torch bitsandbytes transformers accelerate  \
    openai-whisper streamlit spacy librosa presidio-anonymizer  \
    presidio-analyzer nltk flair
RUN python -m spacy download en_core_web_lg