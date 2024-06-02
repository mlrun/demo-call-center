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
import yaml
from pathlib import Path

from sqlalchemy import create_engine, insert
from sqlalchemy.orm import sessionmaker
from src.calls_analysis.db_management import Agent, Client, Call, create_tables
from src.common import ProjectSecrets

import mlrun


def skip_and_import_local_data():
    """
    This function logs example data to the database and to the project.
    Call this function from the notebook in order to skip the calls generation workflow.
    """
    # Get the example data directory:
    example_data_dir = Path("example_data")
    # Get the project:
    project = mlrun.get_current_project()

    # clean and recreate database tables:
    engine = create_engine(url=os.environ[ProjectSecrets.MYSQL_URL])
    Call.__table__.drop(engine)
    Client.__table__.drop(engine)
    Agent.__table__.drop(engine)
    create_tables()
    print("- Initialized tables")

    # load agent and client data:
    agents = project.import_artifact(
        item_path=str(example_data_dir / "agents.zip"),
    ).to_dataitem()
    clients = project.import_artifact(
        item_path=str(example_data_dir / "clients.zip"),
    ).to_dataitem()
    agents = yaml.load(agents.get(), Loader=yaml.FullLoader)
    clients = yaml.load(clients.get(), Loader=yaml.FullLoader)

    # insert agent and client data to database:
    _insert_agents_and_clients_to_db(agents, clients)
    print("- agents and clients inserted")

    # import artifacts for each step:
    for (step_name, artifact_directory) in [
        ("conversation-generation", example_data_dir / "conversation_generation"),
        ("text-to-audio", example_data_dir / "text_to_audio"),
        ("batch-creation", example_data_dir / "batch_creation"),
    ]:
        print(f"- logging step {step_name}")
        _import_artifacts(
            project=project,
            artifact_directory=artifact_directory,
        )

    print("*** first workflow skipped successfully ***")


def _insert_agents_and_clients_to_db(agents: list, clients: list):
    # Create an engine:
    engine = create_engine(url=os.environ[ProjectSecrets.MYSQL_URL])

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Insert the new calls into the table and commit:
    with session.begin() as sess:
        sess.execute(insert(Agent), agents)
        sess.execute(insert(Client), clients)


def _import_artifacts(project, artifact_directory: Path):
    # iterate over artifacts and log them:
    for artifact_file in artifact_directory.iterdir():
        if artifact_file.is_file():
            artifact = project.import_artifact(
                item_path=str(artifact_file),
            )
            print(f"    - artifact {artifact.key} imported")


def save_current_example_data():
    project = mlrun.get_current_project()
    export_dir = Path("example_data")
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name, target_path in [
        ("client-data-generator_clients", "clients.yaml"),
        ("agent-data-generator_agents", "agents.yaml"),
        ("conversation-generation_conversations", "conversation_generation/conversations.zip"),
        ("conversation-generation_metadata", "conversation_generation/metadata.zip"),
        ("conversation-generation_ground_truths", "conversation_generation/ground_truths.zip"),
        ("text-to-audio_audio_files", "text_to_audio/audio_files.zip"),
        ("text-to-audio_audio_files_dataframe", "text_to_audio/dataframe.zip"),
        ("batch-creation_calls_batch", "batch_creation/calls_batch.zip"),
    ]:
        export_path = export_dir / target_path
        if not export_path.exists():
            export_path.parent.mkdir(parents=True, exist_ok=True)
        project.get_artifact(artifact_name).export(f"example_data/{target_path}")
        print(f"- exported {artifact_name} to {target_path}")
    print("*** all artifacts exported successfully ***")
