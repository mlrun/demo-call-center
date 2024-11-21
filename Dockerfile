FROM mlrun/mlrun-gpu:1.7.0

# Update apt-get to install ffmpeg (support audio file formats):
RUN apt-get update -y
RUN apt-get install ffmpeg -y

# Install demo requirements:

RUN pip install transformers==4.44.1
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install bitsandbytes==0.41.1 accelerate==0.24.1 datasets==2.14.6 peft==0.5.0 optimum==1.13.2
RUN pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
RUN pip install langchain==0.0.327 openai==0.28.1
RUN pip install git+https://github.com/suno-ai/bark.git
RUN pip install streamlit==1.28.0 st-annotated-text==4.0.1 spacy==3.7.2 librosa==0.10.1 presidio-anonymizer==2.2.34 presidio-analyzer==2.2.34 nltk==3.8.1 flair==0.13.0
RUN python -m spacy download en_core_web_lg
RUN pip install -U SQLAlchemy

# Align onnxruntime to use gpu:
RUN pip uninstall -y onnxruntime-gpu
RUN pip uninstall -y onnxruntime
RUN pip install onnxruntime-gpu
