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
import os
import tempfile
from typing import List, Optional, Tuple

import boto3
import mlrun
import pandas as pd
from sqlalchemy import (
    Boolean,
    Date,
    Enum,
    ForeignKey,
    Integer,
    String,
    Time,
    bindparam,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.orm import (
    Mapped,
    declarative_base,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.common import CallStatus, ProjectSecrets

ID_LENGTH = 32
FILE_PATH_LENGTH = 500

Base = declarative_base()


class Client(Base):
    __tablename__ = "client"

    # Columns:
    client_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(length=30))
    last_name: Mapped[str] = mapped_column(String(length=30))
    phone_number: Mapped[str] = mapped_column(String(length=20))
    email: Mapped[str] = mapped_column(String(length=50))

    # Many-to-one relationship:
    calls: Mapped[List["Call"]] = relationship(back_populates="client", lazy=True)


class Agent(Base):
    __tablename__ = "agent"

    # Columns:
    agent_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(length=30))
    last_name: Mapped[str] = mapped_column(String(length=30))
    # phone: Mapped[str] = mapped_column(String(length=20))
    # email: Mapped[str] = mapped_column(String(length=50))

    # Many-to-one relationship:
    calls: Mapped[List["Call"]] = relationship(back_populates="agent", lazy=True)


class Call(Base):
    __tablename__ = "call"

    # Metadata:
    call_id: Mapped[str] = mapped_column(String(length=ID_LENGTH), primary_key=True)
    client_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("client.client_id")
    )
    agent_id: Mapped[str] = mapped_column(
        String(length=ID_LENGTH), ForeignKey("agent.agent_id")
    )
    date: Mapped[datetime.date] = mapped_column(Date())
    time: Mapped[datetime.time] = mapped_column(Time())
    status: Mapped[CallStatus] = mapped_column(Enum(CallStatus))
    # Files:
    audio_file: Mapped[str] = mapped_column(String(length=FILE_PATH_LENGTH))
    # TODO: processed_audio_file: Mapped[Optional[str]] = mapped_column(String(length=FILE_PATH_LENGTH))
    transcription_file: Mapped[Optional[str]] = mapped_column(
        String(length=FILE_PATH_LENGTH),
        nullable=True,
        default=None,
    )
    anonymized_file: Mapped[Optional[str]] = mapped_column(
        String(length=FILE_PATH_LENGTH),
        nullable=True,
        default=None,
    )
    # Analysis:
    topic: Mapped[Optional[str]] = mapped_column(
        String(length=50),
        nullable=True,
        default=None,
    )
    summary: Mapped[Optional[str]] = mapped_column(
        String(length=1000),
        nullable=True,
        default=None,
    )
    concern_addressed: Mapped[Optional[bool]] = mapped_column(
        Boolean(),
        nullable=True,
        default=None,
    )
    client_tone: Mapped[Optional[str]] = mapped_column(
        String(length=20),
        nullable=True,
        default=None,
    )
    agent_tone: Mapped[Optional[str]] = mapped_column(
        String(length=20),
        nullable=True,
        default=None,
    )
    upsale_attempted: Mapped[Optional[bool]] = mapped_column(
        Boolean(),
        nullable=True,
        default=None,
    )
    upsale_success: Mapped[Optional[bool]] = mapped_column(
        Boolean(),
        nullable=True,
        default=None,
    )
    empathy: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    professionalism: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    kindness: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    effective_communication: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    active_listening: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )
    customization: Mapped[Optional[int]] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )

    # One-to-many relationships:
    client: Mapped["Client"] = relationship(back_populates="calls", lazy=True)
    agent: Mapped["Agent"] = relationship(back_populates="calls", lazy=True)


def _create_engine(bucket_name: str = None):
    if bucket_name:
        with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
            s3 = boto3.client('s3')
            s3.download_file(bucket_name, "sqlite.db", tmp.name)
            return create_engine(f"sqlite:///{tmp.name}")
    else:
        return create_engine(url=os.environ[ProjectSecrets.MYSQL_URL])

def _update_db(engine):
    bucket_name = os.environ[ProjectSecrets.S3_BUCKET_NAME]
    if bucket_name:
        s3 = boto3.client('s3')
        s3.upload_file(engine.url.database, bucket_name, "sqlite.db")

def create_tables():
    """
    Create the call center schema tables for when creating or loading the MLRun project.
    """
    # Create an engine:
    engine = _create_engine()

    # Create the schema's tables:
    Base.metadata.create_all(engine)

    _update_db(engine)

def insert_clients(context: mlrun.MLClientCtx, clients: list):
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Insert the new calls into the table and commit:
    with session.begin() as sess:
        sess.execute(insert(Client), clients)

    _update_db(engine)

def insert_agents(context: mlrun.MLClientCtx, agents: list):
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Insert the new calls into the table and commit:
    with session.begin() as sess:
        sess.execute(insert(Agent), agents)

    _update_db(engine)

def insert_calls(
        context: mlrun.MLClientCtx, calls: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Cast data from dataframe to a list of dictionaries:
    records = calls.to_dict(orient="records")

    # Insert the new calls into the table and commit:
    with session.begin() as sess:
        sess.execute(insert(Call), records)

    _update_db(engine)

    # Return the metadata and audio files:
    audio_files = list(calls["audio_file"])
    return calls, audio_files


def update_calls(
        context: mlrun.MLClientCtx,
        status: str,
        table_key: str,
        data_key: str,
        data: pd.DataFrame,
):
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Add the status to the dataframe:
    data["status"] = [CallStatus(status)] * len(data)

    # Make sure keys are not duplicates (so we can update by the key with `bindparam`):
    if data_key == table_key:
        data_key += "_2"
        data.rename(columns={table_key: data_key}, inplace=True)

    # Cast data from dataframe to a list of dictionaries:
    data = data.to_dict(orient="records")

    # Insert the new calls into the table and commit:
    with session.begin() as sess:
        sess.connection().execute(
            update(Call).where(getattr(Call, table_key) == bindparam(data_key)), data
        )

    _update_db(engine)

def get_calls() -> pd.DataFrame:
    context = mlrun.get_or_create_ctx("get_calls")
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Select all calls:
    with session.begin() as sess:
        calls = pd.read_sql(select(Call), sess.connection())

    return calls


def get_agents(context: mlrun.MLClientCtx) -> list:
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Select all calls:
    with session.begin() as sess:
        agents = pd.read_sql(select(Agent), sess.connection())
    return agents


def get_clients(context: mlrun.MLClientCtx) -> list:
    # Create an engine:
    engine = _create_engine(context.get_secret(key=ProjectSecrets.S3_BUCKET_NAME))

    # Initialize a session maker:
    session = sessionmaker(engine)

    # Select all calls:
    with session.begin() as sess:
        clients = pd.read_sql(select(Client), sess.connection())
    return clients
