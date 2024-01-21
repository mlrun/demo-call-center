import pandas as pd


def postprocess_answers(answers: pd.DataFrame):
    for column in ["concern_addressed", "upsale_attempted", "upsale_success"]:
        answers[column] = answers[column].apply(
            lambda x: "yes" in x.casefold()
        )

    return answers
