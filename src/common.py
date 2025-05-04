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
import enum


class ProjectSecrets:
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    MYSQL_URL = "MYSQL_URL"
    MYSQL_CONNECT_ARGS = "MYSQL_CONNECT_ARGS"
    S3_BUCKET_NAME = "S3_BUCKET_NAME"


class CallStatus(enum.Enum):
    CREATED = "Created"
    AUDIO_PROCESSED = "Audio processed"
    SPEECH_DIARIZED = "Speech diarized"
    TRANSCRIBED = "Transcribed"
    TRANSLATED = "Translated"
    ANONYMIZED = "Anonymized"
    ANALYZED = "Analyzed"


TOPICS = [
    "slow internet speed",
    "billing discrepancies",
    "account login problems",
    "setting up a new device",
    "phishing or malware concerns",
    "scheduled maintenance notifications",
    "service upgrades",
    "negotiating pricing",
    "canceling service",
    "customer service feedback",
]

TONES = [
    "Positive",
    "Neutral",
    "Negative",
]
