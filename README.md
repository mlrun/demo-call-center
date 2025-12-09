# MLRun's Call Center Demo

This demo showcases how to use LLMs to turn audio files from call center conversations between customers and agents into valuable data, all in a single workflow orchestrated by MLRun. It illustrates the potential power of LLMs for feature extraction, and the simplicity of working with MLRun.

## Overview

MLRun automates the entire workflow, auto-scales resources as needed, and automatically logs and parses values between the different workflow steps.

The demo demonstrates two usages of GenAI:
- Unstructured data generation: Generating audio data with ground truth metadata to evaluate the analysis.
- Unstructured data analysis: Turning audio calls into text and tabular features.

The demo contains a single [notebook](./call-center-demo.ipynb) that encompasses the entire demo.

Most of the functions are imported from [MLRun's hub](https://https://www.mlrun.org/hub/), which contains a wide range of functions and modules that can be used for a variety of use cases. See also the [MLRun hub documentation](https://docs.mlrun.org/en/stable/runtimes/load-from-hub.html). All functions used in the demo include links to their source in the hub. All of the python source code is under [/src](./src).

> **⚠️ Important This demo can take an hour to complete when running without GPUs.**

## Prerequisites

This demo uses:
* [**OpenAI's Whisper**](https://openai.com/research/whisper) &mdash; To transcribe the audio calls into text.
* [**Flair**](https://flairnlp.github.io/) and [**Microsoft's Presidio**](https://microsoft.github.io/presidio/) &mdash; To recognize PII so it can be filtered out.
* [**HuggingFace**](https://huggingface.co/) &mdash; The main machine-learning framework to get the model and tokenizer for the features extraction. 
* [**MLRun**](https://www.mlrun.org/) &mdash; as the orchestrator to operationalize the workflow.

- This demo requires MLRun 1.9 and higher, Python 3.11, with CPU or GPU, a MySQL database.
- MySQL database. SQLite is not currently supported.


<a id="installation"></a>
## Installation

This project can run in different development environments:
* Local computer (using PyCharm, VSCode, Jupyter, etc.)
* Inside GitHub Codespaces 
* Other managed Jupyter environments

### Install the code and the mlrun client 

To get started, fork this repo into your GitHub account and clone it into your development environment.

To install the package dependencies (not required in GitHub codespaces) use:
 
    make install-requirements
    
If you prefer to use Conda, use this instead (to create and configure a conda env):

    make conda-env

Make sure you open the notebooks and select the `mlrun` conda environment 
 
### Install or connect to the MLRun service/cluster

The MLRun service and computation can run locally (minimal setup) or over a remote Kubernetes environment.

If your development environment supports Docker and there are sufficient CPU resources, run:

    make mlrun-docker
    
The MLRun UI can be viewed in: http://localhost:8060
    
If your environment is minimal, run mlrun as a process (no UI):

    [conda activate mlrun &&] make mlrun-api
 
For MLRun to run properly, set up your client environment. This is not required when using **codespaces**, the mlrun **conda** environment, or **iguazio** managed notebooks.

Your environment should include `MLRUN_ENV_FILE=<absolute path to the ./mlrun.env file> ` (point to the mlrun .env file 
in this repo); see [mlrun client setup](https://docs.mlrun.org/en/latest/install/remote.html) instructions for details.  
     
Note: You can also use a remote MLRun service (over Kubernetes): instead of starting a local mlrun: 
edit the [mlrun.env](./mlrun.env) and specify its address and credentials.

### Install the requirements

This demo requires:
* [**MLRun**](https://www.mlrun.org/) &mdash; Orchestrate the demo's workflows.
* [**SQLAlchemy**](https://www.sqlalchemy.org/) &mdash; Manage the MySQL DB of calls, clients and agents.
* [**Vizro**](https://vizro.mckinsey.com/) &mdash; To view the call center DB and transcriptions, and to play the generated conversations.

```
!pip install SQLAlchemy==2.0.31 pymysql dotenv
```
### Setup
Please set the following configuration - choose compute device: CPU or GPU, choose the language of the calls, and whether to skip the calls generation workflow and use pre-generated data.

#### Setup in Iguazio cluster

- This demo is limited to run with MLRun 1.9.x Python 3.11, with CPU or GPU, a mysql database and run the pipeline with `engine = "remote"`.
- Need to setup a MySQL database for the demo. SQLite is not currently supported.
- Set `run_with_gpu = False`, `use_sqlite = False`, `engine = "remote"`.
- .env must include OPENAI_API_KEY, OPENAI_API_BASE, and MYSQL_URL.

#### Setup in Platform McK

- GPU is not supported at the moment.
- SQLite is supported.
- Set `run_with_gpu = False`, `use_sqlite = True`, `engine = "remote"`.
- .env include OPENAI_API_KEY, OPENAI_API_BASE, and S3_BUCKET_NAME.

### Configure the tokens and URL

> **⚠️ Important** Fill in the following variables in your `.env` file.

Tokens are required to run the demo end-to-end:
* [OpenAI ChatGPT](https://chat.openai.com/) &mdash; To generate conversations, two tokens are required:
  * `OPENAI_API_KEY`
  * `OPENAI_API_BASE`
  
> Note: The requirement for the OpenAI token will be removed soon in favor of an open-source LLM.

* [MySQL](https://www.mysql.com/) &mdash; A URL with username and password for collecting the calls into the DB.
    * `MYSQL_URL`
> If you wish to install mysql using helm chart you can use the command below - 
> * `helm install -n <"namesapce"> myrelease bitnami/mysql --set auth.rootPassword=sql123 --set auth.database=mlrun_demos --set primary.service.ports.mysql=3111 --set primary.persistence.enabled=false`
> * Example for MYSQL_URL if you use the above command - `mysql+pymysql://root:sql123@myrelease-mysql.<"namesapce">.svc.cluster.local:3111/mlrun_demos`

For Platform Mck, an S3 bucket name needs to be in the `.env`
* [S3 Bucket]() &mdash; 
  * `S3_BUCKET_NAME`


```
  # True = run with GPU, False = run with CPU
run_with_gpu = False
language = "en" # The languages of the calls, es - Spanish, en - English
skip_calls_generation = False
```

```
import dotenv
import os
import sys
import mlrun
dotenv_file = ".env"
sys.path.insert(0, os.path.abspath("./"))

dotenv.load_dotenv(dotenv_file)
```

```
assert not run_with_gpu
assert os.environ["OPENAI_API_BASE"]
assert os.environ["OPENAI_API_KEY"]
```

```
if not mlrun.mlconf.is_ce_mode():
    assert os.environ["MYSQL_URL"]
    use_sqlite = False
else:
    use_sqlite = True
    ```


## Demo flow

1. Create the project
- **Notebook**: [call-center-demo.ipynb](call-center-demo.ipynb)
- **Description**: 
- **Key steps**: Create the MLRun project. 
- **Key files**:
  - [project.yaml](./project.yaml)
  - [project_setup.py](./project_setup.py)

2. Generate the call data

- **Notebook**: [call-center-demo.ipynb](call-center-demo.ipynb)
- **Description**: Generate the call data. (You can choose to skip this step ans use call data that is already generated and available in the demo.)
- **Key steps**: To generate data, run: Agents & clients data generator, Insert agents & clients data to DB, Get agents & clients from DB, Conversation generation, Text to Audio, and Batch Creation. and Batch creation. Then run the workflow.
- 
- **Key files**:
  - [Insert agents & clients data to the DB and Get agents & clients from the DB](.src/calls_analysis/data_management.py)
  - [Conversation generation and Batch creation](./src/calls_generation/conversations_generator.py)

- **MLRun functions:**
  - [Agents & Clients Data Generator](https://www.mlrun.org/hub/functions/master/structured_data_generator/)
  - [Text to audio](https://www.mlrun.org/hub/functions/master/text_to_audio_generator/)

3. Calls analysis

- **Notebook**: [call-center-demo.ipynb](call-center-demo.ipynb)
- **Description**: Insert the call data to the DB, use diarization to analyze when each person is speaking, transcribe and translate the calls into text and save them as text files, recognice and remove any PII
, anaylze text (call center conversation) with an LLM, postprocess the LLM's answers before updating them into the DB. Then run the all analysis workflow.
- **Key steps**: Insert the calls data to the DB, perform speech diarization, transcribe, recognize PII, analysis. Then run the workflow.

- **Key files:**
  - [Insert the calls data to the DB](.src/calls_analysis/db_management.py)
  - [Postprocess analysis answers](.src/postprocess.py)

- **MLRun functions:**
  - [Perform speech diarization](https://www.mlrun.org/hub/functions/master/silero_vad)
  - [Transcribe](https://www.mlrun.org/hub/functions/master/transcribe)
  - [Recognize PII](https://www.mlrun.org/hub/functions/master/pii_recognizer)
  - [Analysis](https://www.mlrun.org/hub/functions/master/question_answering)

4. View the data

- **Notebook**: [call-center-demo.ipynb](call-center-demo.ipynb)
- **Description**: View the data and features, as they are collected, in the MLRun UI. Deploy [Vizro](https://vizro.mckinsey.com/) to visualize the data in the DB.