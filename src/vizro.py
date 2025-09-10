import os
import shutil
import tarfile
from pathlib import Path

import boto3
import mlrun
import mlrun.common.schemas
import pandas as pd

from src.calls_analysis.db_management import get_calls, get_clients

COLUMNS_MAPPING = {
    "active_listening": "Active Listening",
    "agent_id": "Agent ID",
    "agent_tone": "Agent Tone",
    "date": "Call Date",
    "client_id": "Caller ID",
    "client_tone": "Client Tone",
    "concern_addressed": "Concern Addressed",
    "customization": "Customization",
    "effective_communication": "Effective Communication",
    "empathy": "Empathy",
    "kindness": "Kindness",
    "professionalism": "Professionalism",
    "summary": "Summary",
    "time": "Time",
    "topic": "Topic",
    "upsale_attempted": "Upsale Attempted",
    "upsale_success": "Upsale Success",
    "client_city": "Caller City",
    "anonymized_file": "text_file",
}


def deploy_vizro_application():
    dir_name = "vizro"

    # Prepare the dataframe for vizro:
    _prepare_vizro_source(dir_name)
    print("Application source code ready for deployment.")

    # Archive
    bucket_name = os.getenv("S3_BUCKET_NAME")
    if bucket_name:
        _upload_to_s3(dir_name)
        # Add the source code to the application
        src_path = f"s3://{bucket_name}/{dir_name}.tar.gz"
        print(f"Uploading {src_path} to {bucket_name}")
    else:
        # Set the source path to V3IO
        src_path = f'v3io:///users/{os.environ["V3IO_USERNAME"]}/{os.getcwd().replace("/User/", "")}/{dir_name}.tar.gz'
        print(f"Configuring V3IO {src_path} to UI")
    project = mlrun.get_current_project()
    app = project.get_function("call-center-ui")
    app.with_source_archive(src_path, pull_at_runtime=False)

    # Deploy the application
    app.deploy(force_build=True, create_default_api_gateway=False, with_mlrun=False)
    app.create_api_gateway(
        name="call-center-ui",
        direct_port_access=True,
        set_as_default=True,
        authentication_mode=mlrun.common.schemas.api_gateway.APIGatewayAuthenticationMode.none,
    )
    print("Application deployed successfully!")


def _prepare_vizro_source(dir_name: str):
    clients_df = get_clients()
    calls_df = get_calls()
    vizro_df = pd.merge(
        calls_df,
        clients_df[["client_id", "client_city", "latitude", "longitude"]],
        on="client_id",
    )
    vizro_df = vizro_df.rename(columns=COLUMNS_MAPPING)
    vizro_df.to_csv("vizro/data.csv")

    # add text and audio files to vizro:
    shutil.copytree("outputs", "vizro/outputs", dirs_exist_ok=True)

    # Write the application code to a file
    app_dir = "vizro"

    # Create an archive of the application code
    archive_name = f"{dir_name}.tar.gz"
    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(app_dir)


def _upload_to_s3(dir_name: str):
    # uploading db file to s3:
    s3 = boto3.client("s3")
    bucket_name = Path(mlrun.mlconf.artifact_path).parts[1]

    # Upload the file
    s3.upload_file(
        Filename=f"{dir_name}.tar.gz",
        Bucket=bucket_name,
        Key=f"{dir_name}.tar.gz",
    )
