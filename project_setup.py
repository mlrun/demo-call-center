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
from pathlib import Path
import boto3
import mlrun

from src.calls_analysis.db_management import create_tables
from src.common import ProjectSecrets

CE_MODE = mlrun.mlconf.is_ce_mode()

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
    openai_key = os.getenv(ProjectSecrets.OPENAI_API_KEY)
    openai_base = os.getenv(ProjectSecrets.OPENAI_API_BASE)
    mysql_url = os.getenv(ProjectSecrets.MYSQL_URL, "")

    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image", default=None)
    build_image = project.get_param(key="build_image", default=False)
    gpus = project.get_param(key="gpus", default=0)
    node_name = project.get_param(key="node_name", default=None)
    node_selector = project.get_param(key="node_selector", default=None)
    use_sqlite = project.get_param(key="use_sqlite", default=False)

    # Update sqlite data:
    if use_sqlite:
        # uploading db file to s3:
        if CE_MODE:
            s3 = boto3.client("s3") if not os.getenv("S3_ENDPOINT_URL") else boto3.client('s3', endpoint_url=os.getenv("S3_ENDPOINT_URL"))
            bucket_name = Path(mlrun.mlconf.artifact_path).parts[1]
            # Upload the file
            s3.upload_file(
                Filename="data/sqlite.db",
                Bucket=bucket_name,
                Key="sqlite.db",
            )
            os.environ["S3_BUCKET_NAME"] = bucket_name
        else:
            os.environ["MYSQL_URL"] = f"sqlite:///{os.path.abspath('.')}/data/sqlite.db"
            mysql_url = os.environ["MYSQL_URL"]

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=False)
        

    # Set default image:
    if default_image:
        project.set_default_image(default_image)

    # Build the image:
    if build_image:
        print("Building default image for the demo:")
        _build_image(project=project, with_gpu=gpus, default_image=default_image)

    print(f"use_sqlite ========>>>>>>> {use_sqlite}")
    print(f"Before set secrets mysql_url ========>>>>>>> {mysql_url}")    
    # Set the secrets:
    _set_secrets(
        project=project,
        openai_key=openai_key,
        openai_base=openai_base,
        mysql_url=mysql_url,
        bucket_name=os.getenv(ProjectSecrets.S3_BUCKET_NAME),
    )

    # Refresh MLRun hub to the most up-to-date version:
    mlrun.get_run_db().get_hub_catalog(source_name="default", force_refresh=True)

    # Set the functions:
    _set_calls_generation_functions(project=project, node_name=node_name, image=default_image)
    _set_calls_analysis_functions(project=project, gpus=gpus, node_name=node_name, node_selector=node_selector, image=default_image)

    # Set the workflows:
    _set_workflows(project=project)

    # Set UI application:
    app = project.set_function(
        name="call-center-ui",
        kind="application",
        requirements=["vizro==0.1.38", "gunicorn"]
    )
    # Set the internal application port to Vizro's default port
    app.set_internal_application_port(8050)

    # Set the command to run the Vizro application
    app.spec.command = "gunicorn"
    app.spec.args = [
        "app:app",
        "--bind",
        "0.0.0.0:8050",
        "--chdir",
        f"home/mlrun_code/vizro"
    ]
    app.save()

    # Create the DB tables:
    create_tables()

    # Save and return the project:
    project.save()
    return project

def _build_image(project: mlrun.projects.MlrunProject, with_gpu: bool, default_image):
    config = {
        "base_image": "mlrun/mlrun-gpu" if with_gpu else "mlrun/mlrun-kfp",
        "torch_index": "https://download.pytorch.org/whl/cu118" if with_gpu else "https://download.pytorch.org/whl/cpu",
        "onnx_package": "onnxruntime-gpu" if with_gpu else "onnxruntime"
    }
    # Define commands in logical groups while maintaining order
    system_commands = [
        # Update apt-get to install ffmpeg (support audio file formats):
        "apt-get update -y && apt-get install ffmpeg -y"
    ]

    infrastructure_requirements = [
        "pip install transformers==4.44.1",
        f"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url {config['torch_index']}"
    ]

    huggingface_requirements = [
        "pip install bitsandbytes==0.41.1 accelerate==0.24.1 datasets==2.14.6 peft==0.5.0 optimum==1.13.2"
    ]

    gpu_specific_requirements = [
        "pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/"
    ] if with_gpu else []

    other_requirements = [
        "pip install mlrun langchain==0.2.17 openai==1.58.1 langchain_community==0.2.19 pydub==0.25.1 streamlit==1.28.0 st-annotated-text==4.0.1 spacy==3.7.2 librosa==0.10.1 presidio-anonymizer==2.2.34 presidio-analyzer==2.2.34 nltk==3.8.1 flair==0.13.0 htbuilder==0.6.2",
        "python -m spacy download en_core_web_lg",
        "pip install SQLAlchemy==2.0.31 pymysql requests_toolbelt==0.10.1",
        "pip uninstall -y onnxruntime-gpu onnxruntime",
        f"pip install {config['onnx_package']}",
    ]
    
    # if python 
    # other_requirements += ['pip install protobuf==3.20.30']
    

    # Combine commands in the required order
    commands = (
            system_commands +
            infrastructure_requirements +
            huggingface_requirements +
            gpu_specific_requirements +
            other_requirements
    )

    # Build the image
    assert project.build_image(
        image = default_image,
        base_image=config["base_image"],
        commands=commands,
        set_as_default=True,
        overwrite_build_params=True
    )
    
    # builld the workflow inmage, but set_as_default=False
    
#     workflow_commands=['pip install SQLAlchemy==2.0.31 pymysql && \
#           echo "" > /empty/requirements.txt && \
#           ls -l /empty/ && \
#           cat /empty/Dockerfile && \
#           ls -l /home/ && \
#           rm -rf /home/mlrun-code/project_setup.py'
#          ]
    
#     assert project.build_image(
#                         set_as_default=False,
#                         base_image='mlrun/mlrun-kfp',
#                         image ='.demo-call-center-kfp',
#                         overwrite_build_params=True,
#                         commands=workflow_commands)
    
    
def _set_secrets(
    project: mlrun.projects.MlrunProject,
    openai_key: str,
    openai_base: str,
    mysql_url: str,
    bucket_name: str = None,
):
    print(f"Inside _set_secrets mysql_url ========>>>>>>> {mysql_url}")    
    # Must have secrets:
    project.set_secrets(
        secrets={
            ProjectSecrets.OPENAI_API_KEY: openai_key,
            ProjectSecrets.OPENAI_API_BASE: openai_base,
            ProjectSecrets.MYSQL_URL: mysql_url,
        }
    )
    if bucket_name:
        project.set_secrets(
            secrets={
                ProjectSecrets.S3_BUCKET_NAME: bucket_name,
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
        apply_auto_mount: bool = True,
):
    # Set the given function:
    if with_repo is None:
        with_repo =  not func.startswith("hub://")
    mlrun_function = project.set_function(
        func=func, name=name, kind=kind, with_repo=with_repo, image=image,
    )

    # Configure GPUs according to the given kind:
    if gpus >= 1:
        if node_selector:
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

    if not CE_MODE and apply_auto_mount:
        # Apply auto mount:
        mlrun_function.apply(mlrun.auto_mount())
    # Save:
    print(f"function name ========>>>>>>>>> {name}")
    print(f"function image ========>>>>>>>>> {image}")
    mlrun_function.save()


def _set_calls_generation_functions(
    project: mlrun.projects.MlrunProject,
    node_name: str = None,
    image: str = ".mlrun-project-image-zzz"
):
    # Client and agent data generator
    _set_function(
        project=project,
        func="hub://structured_data_generator",
        name="structured-data-generator",
        kind="job",
        node_name=node_name,
        apply_auto_mount=True,
    )

    # Conversation generator:
    _set_function(
        project=project,
        func="./src/calls_generation/conversations_generator.py",
        name="conversations-generator",
        kind="job",
        image=image,
        node_name=node_name,
        apply_auto_mount=True,
        with_repo=False,
    )

    # Text to audio generator:
    _set_function(
        project=project,
        func="hub://text_to_audio_generator",
        name="text-to-audio-generator",
        kind="job",
        with_repo=False,
        apply_auto_mount=True,
    )


def _set_calls_analysis_functions(
    project: mlrun.projects.MlrunProject,
    gpus: int,
    node_name: str = None,
    node_selector: dict = None,
    image: str = ".mlrun-project-image-zzz"
):
    # DB management:
    _set_function(
        project=project,
        func="./src/calls_analysis/db_management.py",
        name="db-management",
        kind="job",
        image = image,
        node_name=node_name,
        apply_auto_mount=True,
        with_repo=False,
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
        image=image,
        node_name=node_name,
    )


def _set_workflows(project: mlrun.projects.MlrunProject):

    project.set_workflow(
        name="calls-generation", workflow_path="./src/workflows/calls_generation.py", image='.mlrun-project-image-zzz'
    )
    project.set_workflow(
        name="calls-analysis", workflow_path="./src/workflows/calls_analysis.py", image='.mlrun-project-image-zzz'
    )