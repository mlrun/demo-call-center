# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import mlrun

from src.calls_analysis.db_management import create_tables
from src.common import ProjectSecrets


def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project: The project to set up.

    :returns: A fully prepared project for this demo.
    """
    # Unpack secrets from environment variables:
    openai_key = os.environ[ProjectSecrets.OPENAI_API_KEY]
    openai_base = os.environ[ProjectSecrets.OPENAI_API_BASE]
    mysql_url = os.environ[ProjectSecrets.MYSQL_URL]

    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image", default=None)
    build_image = project.get_param(key="build_image", default=False)
    gpus = project.get_param(key="gpus", default=0)
    node_name = project.get_param(key="node_name", default=None)
    node_selector = project.get_param(key="node_selector", default={"alpha.eksctl.io/nodegroup-name": "added-t4"})

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set default image:
    if default_image:
        project.set_default_image(default_image)

    # Build the image:
    if build_image:
        print("Building default image for the demo:")
        _build_image(project=project)

    # Set the secrets:
    _set_secrets(
        project=project,
        openai_key=openai_key,
        openai_base=openai_base,
        mysql_url=mysql_url,
    )

    # Refresh MLRun hub to the most up-to-date version:
    mlrun.get_run_db().get_hub_catalog(source_name="default", force_refresh=True)

    # Set the functions:
    _set_calls_generation_functions(project=project, gpus=gpus, node_name=node_name, node_selector=node_selector)
    _set_calls_analysis_functions(project=project, gpus=gpus, node_name=node_name, node_selector=node_selector)

    # Set the workflows:
    _set_workflows(project=project)

    # Create the DB tables:
    create_tables()

    # Save and return the project:
    project.save()
    return project


def _build_image(project: mlrun.projects.MlrunProject):
    assert project.build_image(
        base_image="mlrun/mlrun-gpu",
        commands=[
            # Update apt-get to install ffmpeg (support audio file formats):
            "apt-get update -y && apt-get install ffmpeg -y",
            # Install demo requirements:
            "pip install transformers==4.44.1",
            "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118",
            "pip install bitsandbytes==0.41.1 accelerate==0.24.1 datasets==2.14.6 peft==0.5.0 optimum==1.13.2",
            "pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/",
            "pip install langchain==0.0.327 openai==0.28.1",
            "pip install git+https://github.com/suno-ai/bark.git",  # suno-bark
            "pip install streamlit==1.28.0 st-annotated-text==4.0.1 spacy==3.7.2 librosa==0.10.1 presidio-anonymizer==2.2.34 presidio-analyzer==2.2.34 nltk==3.8.1 flair==0.13.0",
            "python -m spacy download en_core_web_lg",
            "pip install -U SQLAlchemy",
            "pip uninstall -y onnxruntime-gpu onnxruntime",
            "pip install onnxruntime-gpu",
        ],
        set_as_default=True,
    )


def _set_secrets(
    project: mlrun.projects.MlrunProject,
    openai_key: str,
    openai_base: str,
    mysql_url: str,
):
    # Must have secrets:
    project.set_secrets(
        secrets={
            ProjectSecrets.OPENAI_API_KEY: openai_key,
            ProjectSecrets.OPENAI_API_BASE: openai_base,
            ProjectSecrets.MYSQL_URL: mysql_url,
        }
    )


def _set_function(
        project: mlrun.projects.MlrunProject,
        func: str,
        name: str,
        kind: str,
        gpus: int = 0,
        node_name: str = None,
        with_repo: bool = None,
        image: str = None,
        node_selector: dict = None,
):
    # Set the given function:
    if with_repo is None:
        with_repo =  not func.startswith("hub://")
    mlrun_function = project.set_function(
        func=func, name=name, kind=kind, with_repo=with_repo, image=image,
    )

    # Configure GPUs according to the given kind:
    if gpus >= 1:
        mlrun_function.with_node_selection(node_selector=node_selector)
        if kind == "mpijob":
            # 1 GPU for each rank:
            mlrun_function.with_limits(gpus=1)
            mlrun_function.spec.replicas = gpus
        else:
            # All GPUs for the single job:
            mlrun_function.with_limits(gpus=gpus)
    # Set the node selection:
    elif node_name:
        mlrun_function.with_node_selection(node_name=node_name)
    # Save:
    mlrun_function.save()


def _set_calls_generation_functions(
    project: mlrun.projects.MlrunProject,
    gpus: int,
    node_name: str = None,
    node_selector: dict = None,
):
    # Client and agent data generator
    _set_function(
        project=project,
        func="hub://structured_data_generator",
        name="structured-data-generator",
        kind="job",
        node_name=node_name,
    )

    # Conversation generator:
    _set_function(
        project=project,
        func="./src/calls_generation/conversations_generator.py",
        name="conversations-generator",
        kind="job",
        node_name=node_name,
    )

    # Text to audio generator:
    _set_function(
        project=project,
        func="hub://text_to_audio_generator",
        name="text-to-audio-generator",
        kind="job",  # TODO: MPI once MLRun supports it out of the box
        gpus=gpus,
        node_selector=node_selector,
    )


def _set_calls_analysis_functions(
    project: mlrun.projects.MlrunProject,
    gpus: int,
    node_name: str = None,
    node_selector: dict = None,
):
    # DB management:
    _set_function(
        project=project,
        func="./src/calls_analysis/db_management.py",
        name="db-management",
        kind="job",
        node_name=node_name,
    )

    # Speech diarization:
    _set_function(
        project=project,
        func="hub://silero_vad",
        name="silero-vad",
        kind="job",
        node_name=node_name,
    )

    # Transcription:
    _set_function(
        project=project,
        func="hub://transcribe",
        name="transcription",
        kind="mpijob" if gpus > 1 else "job",
        gpus=gpus,
        node_name=node_name,
        node_selector=node_selector,
    )

    # PII recognition:
    _set_function(
        project=project,
        func="hub://pii_recognizer",
        name="pii-recognition",
        kind="job",
        node_name=node_name,
        image="guyliguazio/call-center-11.8:1.4.1.6",
    )

    # Question answering:
    _set_function(
        project=project,
        func="hub://question_answering",
        name="question-answering",
        kind="job",
        gpus=gpus,
        node_name=node_name,
        node_selector=node_selector,
    )

    # Postprocessing:
    _set_function(
        project=project,
        func="./src/calls_analysis/postprocessing.py",
        name="postprocessing",
        with_repo=False,
        kind="job",
        node_name=node_name,
    )


def _set_workflows(project: mlrun.projects.MlrunProject):
    project.set_workflow(
        name="calls-generation", workflow_path="./src/workflows/calls_generation.py"
    )
    project.set_workflow(
        name="calls-analysis", workflow_path="./src/workflows/calls_analysis.py"
    )
