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
from mlrun.artifacts import ArtifactSpec, DatasetArtifact
import mlrun
import pandas as pd

def skip_and_import_local_data(language:str ):
    """
    This function logs example data to the database and to the project.
    Call this function from the notebook in order to skip the calls generation workflow.
    """
    # Get the example data directory:
    example_data_dir = Path("data")
    # Get the project:
    project = mlrun.get_current_project()

    # clean and recreate database tables:
    engine = create_engine(url=os.environ[ProjectSecrets.MYSQL_URL])
    Call.__table__.drop(engine)
    Client.__table__.drop(engine)
    Agent.__table__.drop(engine)
    create_tables()
    print("- Initialized tables")
    
    #log agents and clients data 
    
    json_spec = ArtifactSpec(unpackaging_instructions={"packager_name": "ListPackager",
                                                       "object_type": "builtins.list","artifact_type": "file","instructions":{"file_format": "json"}})
    zip_spec = ArtifactSpec(unpackaging_instructions={"packager_name": "StrPackager",
                                                      "object_type": "builtins.str","artifact_type": "path","instructions":{"archive_format": "zip","is_directory": "true"}})
    parquet_spec = ArtifactSpec(unpackaging_instructions={"packager_name": "PandasDataFramePackager","object_type": 
                                                          "pandas.core.frame.DataFrame","artifact_type": "dataset","instructions": {}})
    # load agent and client data:
    agents = project.log_artifact(item="agent-data-generator_agents",spec=json_spec,
                                  local_path=str(example_data_dir / f"{language}_agents.json"),db_key="agent-data-generator_agents")
    agents = agents.to_dataitem()
    agents = yaml.load(agents.get(), Loader=yaml.FullLoader)
    clients = project.log_artifact(item="client-data-generator_clients",spec=json_spec,
                                   local_path=str(example_data_dir / f"{language}_clients.json"),db_key="client-data-generator_clients")
    clients = clients.to_dataitem()
    clients = yaml.load(clients.get(), Loader=yaml.FullLoader)

    # insert agent and client data to database:
    _insert_agents_and_clients_to_db(agents, clients)
    print("- agents and clients inserted")

    #log zip files
    remote_zip_path = mlrun.get_sample_path(f'call-demo/{language}_audio_files.zip')
    conversations_art = project.log_artifact(item="conversation-generation_conversations",
                                             spec=zip_spec,local_path=str(example_data_dir / f"{language}_conversations.zip"),db_key="conversation-generation_conversations")
    audio_files_art = project.log_artifact(item="text-to-audio_audio_files",
                                           spec=zip_spec,target_path=remote_zip_path,db_key="text-to-audio_audio_files") 
    #log parquet files
    calls_batch_df = pd.read_parquet(str(example_data_dir / f"{language}_calls_batch.parquet"))
    dataframe_df = pd.read_parquet(str(example_data_dir / f"{language}_dataframe.parquet"))
    ground_truths_df = pd.read_parquet(str(example_data_dir / f"{language}_ground_truths.parquet"))
    metadata_df = pd.read_parquet(str(example_data_dir / f"{language}_metadata.parquet"))
                                
    project.log_artifact(item=DatasetArtifact(key="batch-creation_calls_batch",df=calls_batch_df),
                         spec=parquet_spec,local_path=str(example_data_dir / f"{language}_calls_batch.parquet"))
    project.log_artifact(item=DatasetArtifact(key="text-to-audio_dataframe",df=dataframe_df),spec=parquet_spec)
    project.log_artifact(item=DatasetArtifact(key="conversation-generation_ground_truths",df=ground_truths_df),spec=parquet_spec)
    project.log_artifact(item=DatasetArtifact(key="conversation-generation_metadata",df=metadata_df),spec=parquet_spec)
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




#TODO: change to export the actual data and not the artifacts
def save_current_example_data():
    project = mlrun.get_current_project()
    export_dir = Path("example_data")
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name, target_path in [
        ("client-data-generator_clients", "clients.zip"),
        ("agent-data-generator_agents", "agents.zip"),
        ("conversation-generation_conversations", "conversation_generation/conversations.zip"),
        ("conversation-generation_metadata", "conversation_generation/metadata.zip"),
        ("conversation-generation_ground_truths", "conversation_generation/ground_truths.zip"),
        ("text-to-audio_audio_files", "text_to_audio/audio_files.zip"),
        ("text-to-audio_dataframe", "text_to_audio/dataframe.zip"),
        ("batch-creation_calls_batch", "batch_creation/calls_batch.zip"),
    ]:
        export_path = export_dir / target_path
        if not export_path.exists():
            export_path.parent.mkdir(parents=True, exist_ok=True)
        project.get_artifact(artifact_name).export(f"example_data/{target_path}")
        print(f"- exported {artifact_name} to {target_path}")
    print("*** all artifacts exported successfully ***")
