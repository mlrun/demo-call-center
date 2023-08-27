from typing import List

import mlrun


def pipeline(
    input_path: str,
    transcribe_model: str,
    pii_model: str,
    pii_entities: List[str],
    pii_entity_operator_map: List[str],
    qa_model: str,
    qa_questions: List[str],
    qa_questions_columns: List[str],
):
    """
    The main workflow of the project that gets a directory of conversation audio files and extract features from 
    them into a dataframe dataset.
    
    :param input_path:           Path to the conversation audio files directory.
    :param transcribe_model:     The model to use for the transcribe function. 
                                 Must be one of the official model names listed here: 
                                 https://github.com/openai/whisper#available-models-and-languages
    :param pii_model:            The model to use. Can be "spacy", "flair", "pattern" or "whole".
    :param pii_entities:         The list of entities to recognize.
    :param qa_model:             The model to use for asnwering the given questions.
    :param qa_questions:         A list of questions to ask the LLM chosen about the conversations.
    :param qa_questions_columns: A list of columns to store the LLM asnwers in.
    """
    # Get the project:
    project = mlrun.get_current_project()
    
    # Transcribe:
    transcribe_func = project.get_function("transcribe")
    transcribe_func.apply(mlrun.auto_mount())
    transcription_run = project.run_function(
        function="transcribe",
        handler="transcribe",
        params={
            "input_path": input_path,
            "decoding_options": {"fp16": False},
            "model_name": transcribe_model,
            "output_directory": "./transcripted_data",
        },
        returns=[
            "transcriptions: path",
            "transcriptions_df: dataset",
            {"key": "transcriptions_errors", "artifact_type": "file"},
        ],
    )

    # Recognize and filter PII:
    pii_recognizing_run = project.run_function(
        function="pii-recognizer",
        handler="recognize_pii",
        inputs={"input_path": transcription_run.outputs["transcriptions"]},
        params={
            "model": pii_model,
            "output_path": "./cleaned_data",
            "output_suffix": "output",
            "html_key": "highlighted",
            "entities": pii_entities,
            "entity_operator_map": pii_entity_operator_map,
            "score_threshold": 0.8,
        },
        returns=["output_path: path", "rpt_json: file", "errors: file"],
    )

    # Question answering:
    question_answering_run = project.run_function(
        function="question-answering",
        handler="answer_questions",
        inputs={"input_path": pii_recognizing_run.outputs["output_path"]},
        params={
            "model": qa_model,
            "model_kwargs": {
                "device_map": "auto",
                "load_in_8bit": True,
            },
            "text_wrapper": (
                "Given the following conversation between a Customer and a Call Center Agent:\n"
                "-----\n"
                "{}\n"
                "-----"
            ),
            "questions": qa_questions,
            "questions_columns": qa_questions_columns,
            "generation_config": {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.9,
                "early_stopping": True,
                "max_new_tokens": 150,
            },
        },
        returns=[
            "question_answering_df: dataset",
            "question_answering_errors: result",
        ],
    )
    
    # Postprocess:
    postprocess_run = project.run_function(
        function="postprocess",
        handler="postprocess",
        inputs={
            "transcript_dataset": transcription_run.outputs["transcriptions_df"],
            "qa_dataset": question_answering_run.outputs["question_answering_df"],
        },
        returns=["final_df: dataset"],
    )
