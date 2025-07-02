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
from typing import List

import kfp
import mlrun
from kfp import dsl

###########################
import os
import sys
print(os.getcwd())
print(kfp.__version__)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.insert(0, parent_dir)
parent_dir = os.path.join(parent_dir, os.pardir)
sys.path.insert(0, parent_dir)
print(f'parent dir = {parent_dir}')
###########################

@kfp.dsl.pipeline()
def pipeline(
    amount: int,
    generation_model: str,
    tts_model: str,
    language: str,
    available_voices: List[str],
    min_time: int,
    max_time: int,
    from_date: str,
    to_date: str,
    from_time: str,
    to_time: str,
    num_clients: int,
    num_agents: int,
    generate_clients_and_agents: bool = True,
):
    # Get the project:
    project = mlrun.get_current_project()

    with dsl.Condition(generate_clients_and_agents == True) as generate_data_condition:
        # Generate client data:
        client_data_generator_function = project.get_function(
            "structured_data_generator"
        )
        client_data_run = project.run_function(
            client_data_generator_function,
            handler="generate_data",
            name="client-data-generator",
            params={
                "amount": num_clients,
                "model_name": generation_model,
                "language": language,
                "fields": [
                    f"first_name: in {language}, no special characters",
                    f"last_name: in {language}, no special characters",
                    "phone_number",
                    "email",
                    "client_id: no leading zeros",
                    "client_city: Enter city, state in the US (e.g., Austin, TX), Not only Texas",
                    "latitude: That correspond to the city",
                    "longitude: That correspond to the city",
                ],
            },
            returns=["clients: file"],
        )

        # Insert client data to database
        db_management_function = project.get_function("db-management")
        project.run_function(
            db_management_function,
            handler="insert_clients",
            name="insert-clients",
            inputs={
                "clients": client_data_run.outputs["clients"],
            },
        )

        # Generate agent data:
        agent_data_generator_function = project.get_function(
            "structured_data_generator"
        )
        agent_data_run = project.run_function(
            agent_data_generator_function,
            handler="generate_data",
            name="agent-data-generator",
            params={
                "amount": num_agents,
                "model_name": generation_model,
                "language": language,
                "fields": [
                    f"first_name: in {language}, no special characters",
                    f"last_name: in {language}, no special characters",
                    "agent_id: no leading zeros",
                ],
            },
            returns=["agents: file"],
        )

        # Insert agent data to database
        db_management_function = project.get_function("db-management")
        project.run_function(
            db_management_function,
            handler="insert_agents",
            name="insert-agents",
            inputs={
                "agents": agent_data_run.outputs["agents"],
            },
        )

    # Get agents from database
    db_management_function = project.get_function("db-management")
    get_agents_run = project.run_function(
        db_management_function,
        handler="get_agents",
        name="get-agents",
        returns=["agents: file"],
    ).after(generate_data_condition)

    # Get clients from database
    db_management_function = project.get_function("db-management")
    get_clients_run = project.run_function(
        db_management_function,
        handler="get_clients",
        name="get-clients",
        returns=["clients: file"],
    ).after(generate_data_condition)

    # Generate conversations texts:
    conversations_generator_function = project.get_function("conversations-generator")
    generate_conversations_run = project.run_function(
        conversations_generator_function,
        handler="generate_conversations",
        name="conversation-generation",
        params={
            "amount": amount,
            "model_name": generation_model,
            "language": language,
            "min_time": min_time,
            "max_time": max_time,
            "from_date": from_date,
            "to_date": to_date,
            "from_time": from_time,
            "to_time": to_time,
        },
        inputs={
            "agent_data": get_agents_run.outputs["agents"],
            "client_data": get_clients_run.outputs["clients"],
        },
        returns=[
            "conversations: path",
            "metadata: dataset",
            "ground_truths: dataset",
        ],
    )

    # Text to audio:
    text_to_audio_generator_function = project.get_function("text-to-audio-generator")
    generate_multi_speakers_audio_run = project.run_function(
        text_to_audio_generator_function,
        handler="generate_multi_speakers_audio",
        name="text-to-audio",
        inputs={"data_path": generate_conversations_run.outputs["conversations"]},
        params={
            "speakers": {"Agent": 0, "Client": 1},
            "available_voices": available_voices,
            "model": tts_model,
            "speed": 1,
        },
        returns=[
            "audio_files: path",
            "dataframe: dataset",
            "errors: file",
        ],
    )

    # Create the input batch:
    create_batch_for_analysis_run = project.run_function(
        conversations_generator_function,
        handler="create_batch_for_analysis",
        name="batch-creation",
        inputs={
            "conversations_data": generate_conversations_run.outputs["metadata"],
            "audio_files": generate_multi_speakers_audio_run.outputs["dataframe"],
        },
        returns=["calls_batch: dataset"],
    )
