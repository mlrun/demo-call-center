# Call center demo

This demo showcases how to use LLMs to turn audio files from call center conversations between customers and agents into valuable data, all in a single workflow orchestrated by MLRun. It illustrates the potential power of LLMs for feature extraction, and the simplicity of working with MLRun.

## Overview

MLRun automates the entire workflow, auto-scales resources as needed, and automatically logs and parses values between the different workflow steps.

The demo demonstrates two usages of GenAI:
- Unstructured data generation: Generating audio data with ground truth metadata to evaluate the analysis.
- Unstructured data analysis: Turning audio calls into text and tabular features.

The demo contains two notebooks [notebook 1](./notebook_1_generation.ipynb) and [notebook 2](./notebook_2_analysis.ipynb).

Most of the functions are imported from [MLRun's hub](https://https://www.mlrun.org/hub/), which contains a wide range of functions and modules that can be used for a variety of use cases. See also the [MLRun hub documentation](https://docs.mlrun.org/en/stable/runtimes/load-from-hub.html). All functions used in the demo include links to their source in the hub. All of the python source code is under [/src](./src).

> **⚠️ Important This demo can take up to couple of hours to complete when running without GPUs.**

## Prerequisites

This demo uses:

* [**OpenAI's Whisper**](https://openai.com/research/whisper) &mdash; To transcribe the audio calls into text.
* [**Flair**](https://flairnlp.github.io/) and [**Microsoft's Presidio**](https://microsoft.github.io/presidio/) &mdash; To recognize PII so it can be filtered out.
* [**HuggingFace**](https://huggingface.co/) &mdash; The main machine-learning framework to get the model and tokenizer for the features extraction. 
* [**Vizro**](https://vizro.mckinsey.com/) &mdash; To view the call center DB and transcriptions, and to play the generated conversations. 
* [**MLRun**](https://www.mlrun.org/) &mdash; the orchestrator to operationalize the workflow. MLRun 1.9 and higher, Python 3.11, with CPU or GPU.
* [**SQLAlchemy**](https://www.sqlalchemy.org/) &mdash; Manage the MySQL DB of calls, clients and agents. Installed together with MLRun.
- SQLite

## Installation

This project can run in different development environments:
* Local computer (using PyCharm, VSCode, Jupyter, etc.)
* Inside GitHub Codespaces 
* Other managed Jupyter environments

### Install the code and the MLRun client 

To get started, fork this repo into your GitHub account and clone it into your development environment.

To install the package dependencies (not required in GitHub codespaces) use:
 
    make install-requirements
    
If you prefer to use Conda, use this instead (to create and configure a conda env):

    make conda-env

Make sure you open the notebooks and select the `mlrun` conda environment 
 
### Install or connect to the MLRun service/cluster

The MLRun service and computation can run locally (minimal setup) or over a remote Kubernetes environment.


If your development environment supports Docker and there are sufficient CPU resources (support for Docker setup will be deprecated), run:

    make mlrun-docker
    
The MLRun UI can be viewed in: http://localhost:8060
    
If your environment is minimal, run mlrun as a process (no UI):

    [conda activate mlrun &&] make mlrun-api
 
For MLRun to run properly, set up your client environment. This is not required when using **codespaces**, the mlrun **conda** environment, or **iguazio** managed notebooks.

Your environment should include `MLRUN_ENV_FILE=<absolute path to the ./mlrun.env file> ` (point to the mlrun .env file 
in this repo); see [mlrun client setup](https://docs.mlrun.org/en/stable/install/remote.html) instructions for details.  
     
> Note: You can also use a remote MLRun service (over Kubernetes): instead of starting a local mlrun: 
edit the [mlrun.env](./mlrun.env) and specify its address and credentials.


#### Setup

- Set `run_with_gpu = False`, `use_sqlite = True`, `engine = "remote"`.
- `.env` must include `OPENAI_API_KEY`, `OPENAI_API_BASE`

### Configure the tokens and URL

> **⚠️ Important** Fill in the following variables in your `.env` file.

> Note: The requirement for the OpenAI token will be removed soon in favor of an open-source LLM.

Tokens are required to run the demo end-to-end:
* [OpenAI ChatGPT](https://chat.openai.com/) &mdash; To generate conversations, two tokens are required:
  * `OPENAI_API_KEY`
  * `OPENAI_API_BASE`

## Demo flow

1. Create the project
- **Notebook**: [notebook_1_generation.ipynb](notebook_1_generation.ipynb)
- **Description**: 
- **Key steps**: Create the MLRun project. 
- **Key files**:
  - [project_setup.py](./project_setup.py)

2. Generate the call data

- **Notebook**: [notebook_1_generation.ipynb](notebook_1_generation.ipynb)
- **Description**: Generate the call data. (You can choose to skip this step ans use call data that is already generated and available in the demo.)
- **Key steps**: To generate data, run: Agents & clients data generator, Insert agents & clients data to DB, Get agents & clients from DB, Conversation generation, Text to Audio, and Batch Creation. and Batch creation. Then run the workflow.

- **Key files**:
  - [Insert agents & clients data to the DB and Get agents & clients from the DB](.src/calls_analysis/data_management.py)
  - [Conversation generation and Batch creation](./src/calls_generation/conversations_generator.py)

- **MLRun hub functions:**
  - [Agents & Clients Data Generator](https://www.mlrun.org/hub/functions/master/structured_data_generator/)
  - [Text to audio](https://www.mlrun.org/hub/functions/master/text_to_audio_generator/)

3. Calls analysis

- **Notebook**: [notebook_2_analysis.ipynb](notebook_2_analysis.ipynb)
- **Description**: Insert the call data to the DB, use diarization to analyze when each person is speaking, transcribe and translate the calls into text and save them as text files, recognice and remove any PII
, anaylze text (call center conversation) with an LLM, postprocess the LLM's answers before updating them into the DB. Then run the all analysis workflow.
- **Key steps**: Insert the calls data to the DB, perform speech diarization, transcribe, recognize PII, analysis. Then run the workflow.

- **Key files:**
  - [Insert the calls data to the DB](.src/calls_analysis/db_management.py)
  - [Postprocess analysis answers](.src/postprocess.py)

- **MLRun hub functions:**
  - [Perform speech diarization](https://www.mlrun.org/hub/functions/master/silero_vad)
  - [Transcribe](https://www.mlrun.org/hub/functions/master/transcribe)
  - [Recognize PII](https://www.mlrun.org/hub/functions/master/pii_recognizer)
  - [Analysis](https://www.mlrun.org/hub/functions/master/question_answering)

4. View the data

- **Notebook**: [notebook_2_analysis.ipynb](notebook_2_analysis.ipynb)
- **Description**: View the data and features, as they are collected, in the MLRun UI. Deploy [Vizro](https://vizro.mckinsey.com/) to visualize the data in the DB.