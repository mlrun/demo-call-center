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
    default_image = project.get_param(key="default_image")
    gpus = project.get_param(key="gpus", default=0)
    node_name = project.get_param(key="node_name", default=None)

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set or build the default image:
    if default_image is None:
        print("Building default image for the demo:")
        _build_image(project=project)
    else:
        project.set_default_image(default_image)

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
    _set_calls_generation_functions(project=project, gpus=gpus, node_name=node_name)
    _set_calls_analysis_functions(project=project, gpus=gpus, node_name=node_name)

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
            "apt-get update -y",
            "apt-get install ffmpeg -y",
            # Install demo requirements:
            "pip install tqdm mpi4py",
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install pyannote.audio faster-whisper bitsandbytes transformers accelerate datasets peft optimum",
            "pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/",
            "pip install langchain openai",
            "pip install git+https://github.com/suno-ai/bark.git",  # suno-bark
            "pip install streamlit st-annotated-text spacy librosa presidio-anonymizer presidio-analyzer nltk flair",
            "pip install -U SQLAlchemy",
            "pip uninstall -y onnxruntime-gpu",
            "pip uninstall -y onnxruntime",
            "pip install onnxruntime-gpu",
            "python -m spacy download en_core_web_lg",
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
        image: str = None,
):
    # Set the given function:
    with_repo = not func.startswith("hub://")
    mlrun_function = project.set_function(
        func=func, name=name, kind=kind, with_repo=with_repo, image=image,
    )

    # Configure GPUs according to the given kind:
    if gpus >= 1:
        mlrun_function.with_node_selection(node_selector={"alpha.eksctl.io/nodegroup-name": "added-t4"})
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
    node_name: str = None
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
    )


def _set_calls_analysis_functions(
    project: mlrun.projects.MlrunProject,
    gpus: int,
    node_name: str = None
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
    )

    # Postprocessing:
    _set_function(
        project=project,
        func="./src/calls_analysis/postprocessing.py",
        name="postprocessing",
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
