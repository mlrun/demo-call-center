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
import datetime
import hashlib
import os
import pathlib
import random
import tempfile
from typing import Tuple

import mlrun
import pandas as pd
import tqdm
from langchain.chat_models import ChatOpenAI

from src.common import TONES, TOPICS, ProjectSecrets

#: The approximate amount of words in one minute.
WORDS_IN_1_MINUTE = 240


def generate_conversations(
        context: mlrun.MLClientCtx,
        amount: int,
        agent_data: pd.DataFrame,
        client_data: pd.DataFrame,
        output_directory: str = None,
        model_name: str = "gpt-3.5-turbo",
        language: str = "en",
        min_time: int = 2,
        max_time: int = 5,
        from_date: str = "01.01.2023",
        to_date: str = "01.03.2023",
        from_time: str = "09:00",
        to_time: str = "17:00",

) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Generates a list of conversations between an internet provider call center and a customer.

    :param context:             The MLRun context.
    :param amount:              The number of conversations to generate.
    :param agent_data:          The agent data to use for the conversations.
    :param client_data:         The client data to use for the conversations.
    :param output_directory:    The directory to save the conversations to.
    :param model_name:          The name of the model to use for conversation generation.
                                You should choose one of GPT-4 or GPT-3.5 from the list here:
                                https://platform.openai.com/docs/models. Default: 'gpt-3.5-turbo'.
    :param language:            The language to use for the generated conversation text.
    :param min_time:            Minimum time of conversation in minutes.
                                Will be used approximately to generate the minimum words with the following assessment:
                                240 words are equal to one minute. Default: 2.
    :param max_time:            Maximum time of conversation in minutes.
                                Will be used approximately to generate the maximum words  with the following assessment:
                                240 words are equal to one minute. Default: 5.
    :param from_date:           The minimum date of the conversation.
    :param to_date:             The maximum date of the conversation.
    :param from_time:           The minimum time (HH:MM) of the conversation.
    :param to_time:             The maximum time (HH:MM) of the conversation.
    """
    # Get the minimum and maximum amount of words:
    min_words = WORDS_IN_1_MINUTE * min_time
    max_words = WORDS_IN_1_MINUTE * max_time

    # Get the minimum and maximum dates and times:
    min_time = datetime.datetime.strptime(from_time, "%H:%M")
    max_time = datetime.datetime.strptime(to_time, "%H:%M")
    min_date = datetime.datetime.strptime(from_date, "%m.%d.%Y").date()
    max_date = datetime.datetime.strptime(to_date, "%m.%d.%Y").date()

    # Create the concern addressed options:
    concern_addressed_options = {
        True: "",
        False: "Don't",
    }

    # Create the agent upsales options:
    agent_upsales_options = {
        "Doesn't try": "Doesn't try to upsale the customer on more services.",
        "Tries and doesn't succeed": "Tries to upsale the customer on more services, and doesn't succeed",
        "Tries and succeeds": "Tries to upsale the customer on more services, and succeeds",
    }

    # Create the upsale mapping:
    upsale_mapping = {
        "Doesn't try": [False, False],
        "Tries and doesn't succeed": [True, False],
        "Tries and succeeds": [True, True],
    }

    # Create the prompt structure:
    prompt_structure = (
        "Generate a conversation between an internet provider call center agent named {agent_name} from (“Iguazio Internet”) and "
        "a client named {client_name} with email: {client_email} and phone number: {client_phone} in {language} except 'Agent' and 'Client' prefixes which are constants.\n"
        "Format the conversation as follow:\n"
        "Agent: <text here>\n"
        "Client: <text here>\n"
        "The conversations has to include at least {min_words} words and no more than {max_words} words.\n"
        "The call must include the following steps: \n"
        "1. Opening (greeting and customer details validation and confirmation)\n"
        "2. Presenting the problem by the customer"
        "3. The agent {concern_addressed} address the client's concern.\n"
        "4. The Agent {agent_upsales}"
        "5. Summerizing and closing the call"
        "It has to be about a client who is calling to discuss about {topic}.\n"
        "Do not add any descriptive tag and do not mark the end of the conversation with [End of conversation].\n"
        "Use ... for hesitation.\n"
        "The client needs to have a {client_tone} tone.\n"
        "The agent needs to have a {agent_tone}.\n"
        "Remove from the final output any word inside parentheses of all types. \n"
        "use the following levels of these attributes while describing the agent's role: \n"
        "Empathy {empathy}, Professionalism {professionalism}, Kindness {kindness}, \n"
        "Effective Communication {effective_communication}, Active listening {active_listening}, Customization {customization}."
    )

    # Load the OpenAI model using langchain:
    os.environ["OPENAI_API_KEY"] = context.get_secret(key=ProjectSecrets.OPENAI_API_KEY)
    os.environ["OPENAI_API_BASE"] = context.get_secret(
        key=ProjectSecrets.OPENAI_API_BASE
    )
    llm = ChatOpenAI(model=model_name)

    # Create the output directory:
    if output_directory is None:
        output_directory = tempfile.mkdtemp()
    output_directory = pathlib.Path(output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # Start generating conversations:
    conversations = []
    ground_truths = []
    for _ in tqdm.tqdm(range(amount), desc="Generating"):
        # Randomize the conversation metadata:
        conversation_id = _generate_id()
        date = _get_random_date(min_date=min_date, max_date=max_date)
        time = _get_random_time(min_time=min_time, max_time=max_time)

        # Randomly select the conversation parameters:
        concern_addressed = random.choice(list(concern_addressed_options.keys()))
        agent_upsales = random.choice(list(agent_upsales_options.keys()))
        client_tone = random.choice(TONES)
        agent_tone = random.choice(TONES)
        topic = random.choice(TOPICS)
        agent = agent_data.sample().to_dict(orient="records")[0]
        client = client_data.sample().to_dict(orient="records")[0]

        # Generate levels os different agent attributes:
        empathy = random.randint(1, 5)
        professionalism = random.randint(1, 5)
        kindness = random.randint(1, 5)
        effective_communication = random.randint(1, 5)
        active_listening = random.randint(1, 5)
        customization = random.randint(1, 5)

        # Create the prompt:
        prompt = prompt_structure.format(
            language=language,
            min_words=min_words,
            max_words=max_words,
            topic=topic,
            concern_addressed=concern_addressed_options[concern_addressed],
            agent_upsales=agent_upsales_options[agent_upsales],
            client_tone=client_tone,
            agent_tone=agent_tone,
            agent_name=f"{agent['first_name']} {agent['last_name']}",
            client_name=f"{client['first_name']} {client['last_name']}",
            client_email=client["email"],
            client_phone=client["phone_number"],
            empathy=empathy,
            professionalism=professionalism,
            kindness=kindness,
            effective_communication=effective_communication,
            active_listening=active_listening,
            customization=customization,
        )

        # Generate the conversation:
        conversation = llm.predict(text=prompt)
        # Remove redundant newlines and spaces:
        conversation = "".join(
            [
                line
                for line in conversation.strip().splitlines(keepends=True)
                if line.strip("\n").strip()
            ]
        )
        # Save to file:
        conversation_text_path = output_directory / f"{conversation_id}.txt"
        with open(conversation_text_path, "w") as fp:
            fp.write(conversation)

        # Collect to the conversations and ground truths lists:
        conversations.append(
            [
                conversation_id,
                conversation_text_path.name,
                client['client_id'],
                agent['agent_id'],
                date,
                time,

            ]
        )
        ground_truths.append(
            [
                conversation_id,
                language,
                topic,
                concern_addressed,
                upsale_mapping[agent_upsales][0],
                upsale_mapping[agent_upsales][1],
                client_tone,
                agent_tone,
                client['client_id'],
                agent['agent_id'],
                empathy,
                professionalism,
                kindness,
                effective_communication,
                active_listening,
                customization,
            ]
        )

    # Cast the conversations and ground truths into a dataframe:
    conversations = pd.DataFrame(
        conversations,
        columns=["call_id", "text_file", "client_id", "agent_id", "date", "time"],
    )
    ground_truths = pd.DataFrame(
        ground_truths,
        columns=[
            "call_id",
            "language",
            "topic",
            "concern_addressed",
            "agent_tries_upsale",
            "agent_succeeds_upsale",
            "client_tone",
            "agent_tone",
            "agent_id",
            "client_id",
            "empathy",
            "professionalism",
            "kindness",
            "effective_communication",
            "active_listening",
            "customization",
        ],
    )

    return str(output_directory), conversations, ground_truths


def _get_random_time(
    min_time: datetime.datetime, max_time: datetime.datetime
) -> datetime.time:
    if max_time.hour <= min_time.hour:
        max_time += datetime.timedelta(days=1)
    return (
        min_time
        + datetime.timedelta(
            seconds=random.randint(0, int((max_time - min_time).total_seconds())),
        )
    ).time()


def _get_random_date(min_date, max_date) -> datetime.date:
    return min_date + datetime.timedelta(
        days=random.randint(0, int((max_date - min_date).days)),
    )


def create_batch_for_analysis(
    conversations_data: pd.DataFrame, audio_files: pd.DataFrame
) -> pd.DataFrame:
    conversations_data = conversations_data.join(
        other=audio_files.set_index(keys="text_file"), on="text_file"
    )
    conversations_data.drop(columns="text_file", inplace=True)
    conversations_data.dropna(inplace=True)
    return conversations_data


def _generate_id() -> str:
    return hashlib.md5(str(datetime.datetime.now()).encode("utf-8")).hexdigest()
