# <img src="https://uxwing.com/wp-content/themes/uxwing/download/business-professional-services/boy-services-support-icon.png" style="height: 40px"/> MLRun's Call Center Demo

<img src="./images/call-center-readme.png" alt="huggingface-mlrun" style="width: 600px"/>

In this demo we will be showcasing how we used LLMs to turn call center conversation audio files of customers and agents into valueable data in a single workflow orchastrated by MLRun.

MLRun will be automating the entire workflow, auto-scale resources as needed and automatically log and parse values between the workflow different steps.

By the end of this demo you will see the potential power of LLMs for feature extraction, and how easy it is being done using MLRun!

We will use:
* [**OpenAI's Whisper**](https://openai.com/research/whisper) - To transcribe the audio calls into text.
* [**Flair**](https://flairnlp.github.io/) and [**Microsoft's Presidio**](https://microsoft.github.io/presidio/) - To recognize PII for filtering it out.
* [**HuggingFace**](https://huggingface.co/) - as the main machine learning framework to get the model and tokenizer for the features extraction. The demo uses [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) as the LLM to asnwer questions.
* and [**MLRun**](https://www.mlrun.org/) - as the orchastraitor to operationalize the workflow.

The demo contains a single [notebook](./notebook.ipynb) that covers the entire demo.

Most of the functions are being imported from [MLRun's hub](https://docs.mlrun.org/en/stable/runtimes/load-from-hub.html) - a wide range of functions that can be used for a variety of use cases. You can find all the python source code under [/src](./src) and links to the used functions from the hub in the notebook.

Enjoy!

___
<a id="installation"></a>
## Installation

This project can run in different development environments:
* Local computer (using PyCharm, VSCode, Jupyter, etc.)
* Inside GitHub Codespaces 
* Other managed Jupyter environments

### Install the code and mlrun client 

To get started, fork this repo into your GitHub account and clone it into your development environment.

To install the package dependencies (not required in GitHub codespaces) use:
 
    make install-requirements
    
If you prefer to use Conda use this instead (to create and configure a conda env):

    make conda-env

> Make sure you open the notebooks and select the `mlrun` conda environment 
 
### Install or connect to MLRun service/cluster

The MLRun service and computation can run locally (minimal setup) or over a remote Kubernetes environment.

If your development environment support docker and have enough CPU resources run:

    make mlrun-docker
    
> MLRun UI can be viewed in: http://localhost:8060
    
If your environment is minimal, run mlrun as a process (no UI):

    [conda activate mlrun &&] make mlrun-api
 
For MLRun to run properly you should set your client environment, this is not required when using **codespaces**, the mlrun **conda** environment, or **iguazio** managed notebooks.

Your environment should include `MLRUN_ENV_FILE=<absolute path to the ./mlrun.env file> ` (point to the mlrun .env file 
in this repo), see [mlrun client setup](https://docs.mlrun.org/en/latest/install/remote.html) instructions for details.  
     
> Note: You can also use a remote MLRun service (over Kubernetes), instead of starting a local mlrun, 
> edit the [mlrun.env](./mlrun.env) and specify its address and credentials  
