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

from src.common import TONES, TOPICS, CallStatus


QUESTIONS = [
    [
        f"1. Classify the topic of the text from the following list (choose one): {TOPICS}",
        "2. Write a long summary of the text, focus on the topic (max 50 words).",
        "3. Was the Client's concern addressed, (choose only one) [Yes, No]?",
        f"4. Was the Client tone (choose only one, if not sure choose Neutral) {TONES}? ",
        f"5. Was the Call Center Agent tone (choose only one, if not sure choose Neutral) {TONES}?",
    ],
    [
        "1. Did the agent try to upsale the customer (choose only one) [Yes, No]? (sell any additional product or service)",
        "2. If the agent indeed try to upsale the client, did he succeed (choose only one) [Yes, No]? if he didn't try' answer No",
        "3. Rate the agent's level of empathy (The ability to understand and share the feelings of others) on a scale of 1-5.",
        "4. Rate the agent's level of professionalism (Conducting oneself in a way that is appropriate for the workplace) on a scale of 1-5.",
        "5. Rate the agent's level of kindness (The quality of being friendly, generous, and considerate) on a scale of 1-5.",
        "6. Rate the agent's level of effective communication (The ability to convey information clearly and concisely) on a scale of 1-5.",
        "7. Rate the agent's level of active listening (The process of paying attention to and understanding what someone is saying) on a scale of 1-5.",
        "8. Rate the agent's level of customization (The process of tailoring something to the specific needs or preferences of an individual) on a scale of 1-5.",
    ]
]
DEMO_CALL = (
    "Agent: Good afternoon, you've reached [Internet Service Provider] customer support. I'm Megan. How can I assist "
    "you today?\n"
    "Customer: Hello, Megan. This is Lisa. I've noticed some billing discrepancies on my last statement.\n"
    "Agent: I'm sorry to hear that, Lisa. I'd be happy to help you with that. Could you please provide me with your "
    "account number or phone number associated with your account?\n"
    "Customer: Of course, my account number is 123456789.\n"
    "Agent: Thank you, Lisa. Let me pull up your account. I see the billing discrepancies you mentioned. It appears "
    "there was an error in the charges. I apologize for the inconvenience.\n"
    "Customer: Thank you for acknowledging the issue, Megan. Can you please help me get it resolved?\n"
    "Agent: Absolutely, Lisa. I've made note of the discrepancies, and I'll escalate this to our billing department "
    "for investigation and correction. You should see the adjustments on your next statement.\n"
    "Customer: That sounds good, Megan. I appreciate your help.\n"
    "Agent: You're welcome, Lisa. If you have any more questions or concerns in the future, please don't hesitate to "
    "reach out. Is there anything else I can assist you with today?\n"
    "Customer: No, that's all. Thank you for your assistance, Megan.\n"
    "Agent: Not a problem, Lisa. Have a wonderful day, and we'll get this sorted out for you.\n"
    "Customer: You too! Goodbye, Megan.\n"
    "Agent: Goodbye, Lisa!"
)
DEMO_ANSWERS = [(
    "1. billing discrepancies\n"
    "2. The customer, contacted the call center regarding billing discrepancies on her statement. The agent, "
    "acknowledged the issue, assured The customer it would be resolved, and escalated it to the billing department for "
    "correction.\n"
    "3. Yes.\n"
    "4. Natural.\n"
    "5. positive.\n"),

    (
    "1. No\n"
    "2. No\n"
    "3. 4\n"
    "4. 5\n"
    "5. 4\n"
    "6. 5\n"
    "7. 4\n"
    "8. 3"
)]
TEXT_WRAPPER = [(
    f"<|im_start|>system: You are an AI assistant that answers questions accurately and shortly<|im_end|>\n"
    f"<|im_start|>user: Given the following text:\n"
    f"{DEMO_CALL}\n"
    f"answer the questions as accurately as you can:\n"
    f"{QUESTIONS[i]}<|im_end|>\n"
    f"<|im_start|>assistant:\n"
    f"{DEMO_ANSWERS[i]}<|im_end|>\n"
    f"<|im_start|>user: Given the following text:\n"
    "{}"
) for i in range(len(QUESTIONS))]
QUESTIONS_WRAPPER = (
    " answer the given questions as accurately as you can, do not write more answers the questions:\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant:\n"
)


@kfp.dsl.pipeline()
def pipeline(
    batch: str,
    calls_audio_files: str,
    transcribe_model: str,
    translate_to_english: bool,
    pii_recognition_model: str,
    pii_recognition_entities: List[str],
    pii_recognition_entity_operator_map: List[str],
    question_answering_model: str,
    batch_size: int = 2,
):
    # Get the project:
    project = mlrun.get_current_project()

    # Insert new calls:
    db_management_function = project.get_function("db-management")
    insert_calls_run = project.run_function(
        db_management_function,
        handler="insert_calls",
        name="insert-calls",
        inputs={"calls": batch},
        returns=[
            "calls_batch: dataset",
            "audio_files: file",
        ],
    )

    # Speech diarize:
    speech_diarization_function = project.get_function("silero-vad")
    diarize_run = project.run_function(
        speech_diarization_function,
        handler="diarize",
        name="diarization",
        inputs={"data_path": calls_audio_files},
        params={
            "speaker_labels": ["Agent", "Client"],
            "verbose": True,
        },
        returns=["speech_diarization: file", "diarize_errors: file"],
    ).after(insert_calls_run)

    # Update diarization state:
    update_calls_post_speech_diarization_run = project.run_function(
        db_management_function,
        handler="update_calls",
        name="update-calls",
        inputs={"data": insert_calls_run.outputs["calls_batch"]},
        params={
            "status": CallStatus.SPEECH_DIARIZED.value,
            "table_key": "call_id",
            "data_key": "call_id",
        },
    ).after(diarize_run)

    # Transcribe:
    transcription_function = project.get_function("transcription")
    transcribe_run = project.run_function(
        transcription_function,
        handler="transcribe",
        name="transcription",
        inputs={
            "data_path": calls_audio_files,
            "speech_diarization": diarize_run.outputs["speech_diarization"],
        },
        params={
            "model_name": transcribe_model,
            "device": "cuda",
            "use_better_transformers": True,
            "batch_size": batch_size,
            "translate_to_english": translate_to_english,

        },
        returns=[
            "transcriptions: path",
            "transcriptions_dataframe: dataset",
            "transcriptions_errors: file",
        ],
    )

    # Update transcription state:
    update_calls_post_transcription_run = project.run_function(
        db_management_function,
        handler="update_calls",
        name="update-calls-2",
        inputs={"data": transcribe_run.outputs["transcriptions_dataframe"]},
        params={
            "status": CallStatus.TRANSCRIBED.value,
            "table_key": "audio_file",
            "data_key": "audio_file",
        },
    )

    # Recognize PII:
    pii_recognition_function = project.get_function("pii-recognition")
    recognize_pii_run = project.run_function(
        pii_recognition_function,
        handler="recognize_pii",
        name="pii-recognition",
        inputs={"input_path": transcribe_run.outputs["transcriptions"]},
        params={
            "model": pii_recognition_model,
            "html_key": "highlighted",
            "entities": pii_recognition_entities,
            "entity_operator_map": pii_recognition_entity_operator_map,
            "score_threshold": 0.8,
            "is_full_report": False,
        },
        returns=[
            "anonymized_files: path",
            "anonymized_files_dataframe: dataset",
            "anonymized_files_errors: file",
            "anonymized_files_report: file",
        ],
    )

    # Update PII state:
    update_calls_post_pii_recognition_run = project.run_function(
        db_management_function,
        handler="update_calls",
        name="update-calls-3",
        inputs={"data": recognize_pii_run.outputs["anonymized_files_dataframe"]},
        params={
            "status": CallStatus.ANONYMIZED.value,
            "table_key": "transcription_file",
            "data_key": "original_file",
        },
    )

    # Question-answering:
    question_answering_function = project.get_function("question-answering")
    answer_questions_run = project.run_function(
        question_answering_function,
        handler="answer_questions",
        name="analysis",
        inputs={"data_path": recognize_pii_run.outputs["anonymized_files"]},
        params={
            "verbose": True,
            "model_name": question_answering_model,
            "auto_gptq_exllama_max_input_length": 8192,
            "device_map": "auto",
            "text_wrapper": TEXT_WRAPPER,
            "questions": QUESTIONS,
            "questions_wrapper": QUESTIONS_WRAPPER,
            "questions_columns": [
                "topic",
                "summary",
                "concern_addressed",
                "client_tone",
                "agent_tone",
                "upsale_attempted",
                "upsale_success",
                "empathy",
                "professionalism",
                "kindness",
                "effective_communication",
                "active_listening",
                "customization",
            ],
            "questions_config": [
                {},
                {
                    "type": "poll",
                    "poll_count": 3,
                    "poll_strategy": "most_common"
                }
            ],
            "generation_config": {
                "max_new_tokens": 250,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
            "batch_size": 1,
            "model_kwargs": {},
        },
        returns=[
            "question_answering_dataframe: dataset",
            "question_answering_errors: file",
        ],
    )

    # Postprocess answers:
    postprocessing_function = project.get_function("postprocessing")
    postprocess_answers_run = project.run_function(
        postprocessing_function,
        handler="postprocess_answers",
        name="answers-postprocessing",
        inputs={
            "answers": answer_questions_run.outputs["question_answering_dataframe"]
        },
        returns=["processed_answers: dataset"],
    )

    # Update question answering state:
    update_calls_post_question_answering_run = project.run_function(
        db_management_function,
        handler="update_calls",
        name="update-calls-4",
        inputs={"data": postprocess_answers_run.outputs["processed_answers"]},
        params={
            "status": CallStatus.ANALYZED.value,
            "table_key": "anonymized_file",
            "data_key": "text_file",
        },
    )
