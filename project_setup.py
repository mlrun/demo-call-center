import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    """
    Creating the project for this demo.
    
    :param project: The project to setup.
    
    :returns: A fully prepared project for this demo.
    """
    # Set the project git source
    source = project.get_param("source")
    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)

    # Set or build the default image:
    if project.get_param("default_image") is None:
        print("Building image for the demo:")
        assert project.build_image(
            base_image='mlrun/mlrun-gpu',
            commands=[
                "apt-get update -y",
                "apt-get install ffmpeg -y",
                "pip install tqdm torch", 
                "pip install bitsandbytes transformers accelerate",
                "pip install openai-whisper",
                "pip install streamlit spacy librosa presidio-anonymizer presidio-analyzer nltk flair",
                "python -m spacy download en_core_web_lg",
            ],
            set_as_default=True,
        )
    else:
        project.set_default_image(project.get_param("default_image"))

    # Set the transcribe function:
    transcribe_func = project.set_function("hub://transcribe", name="transcribe")
    transcribe_func.apply(mlrun.auto_mount())
    transcribe_func.save()

    # Set the PII recognition function:
    pii_recognizer_func = project.set_function("hub://pii_recognizer", name="pii-recognizer")

    # Set the question asnwering function:
    question_answering_func = project.set_function("hub://question_answering", name="question-answering")
    if project.get_param("gpus") > 0:
        print("Using GPUs for question asnwering.")
        question_answering_func.with_limits(gpus=project.get_param("gpus"))
        question_answering_func.save()
    
    # Set the postprocessing function:
    postprocess_function = project.set_function("./src/postprocess.py", kind="job", name="postprocess")
    
    # Set the workflow:
    project.set_workflow("workflow", "./src/workflow.py")

    # Save and return the project:
    project.save()
    return project
