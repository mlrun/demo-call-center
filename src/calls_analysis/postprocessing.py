import pandas as pd


def postprocess_answers(answers: pd.DataFrame):
    for column in ["concern_addressed", "upsale_attempted", "upsale_success"]:
        answers[column] = answers[column].apply(lambda x: "yes" in x.casefold())
    for column in ["client_tone", "agent_tone"]:
        answers[column] = answers[column].apply(
            lambda x: "Positive" if "Positive" in x else x
        )
        answers[column] = answers[column].apply(
            lambda x: "Negative" if "Negative" in x else x
        )
        answers[column] = answers[column].apply(
            lambda x: "Neutral" if "Neutral" in x else x
        )
    return answers
