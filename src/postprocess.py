import pandas as pd


def _clean_issue(s: str) -> str:
    """
    Clean issue column from enumerate prefix and remove {'(', ')', ':', '"'}
    
    :param s: The string to clean.
    
    :returns: The cleaned string.
    """
    if len(s) > 2 and s[1] == ".":
        s = s[2:]
    s = s.translate({ord(c): None for c in '():"'})
    return s


def _extract_is_fixed(s: str) -> str:
    """
    Extract a single word answer from the LLM response (Yes / No).
    
    :param s: The content to extract the single word asnwer from.
    
    :returns: The extracted answer.
    """
    s = s.casefold()
    if "not explicitly" in s:
        return "Unknown"
    if any(sub in s for sub in ["yes", "was fixed"]):
        return "Yes"
    if any(sub in s for sub in ["no", "was not fixed"]):
        return "No"
    return "Unknown"


def _extract_tone(s: str) -> str:
    """
    Extract a single word answer from the LLM response (Positive / Neutral / Negative)
    
    :param s: The content to extract the single word asnwer from.
    
    :returns: The extracted answer.
    """
    s = s.casefold()
    if "positive" in s:
        return "Positive"
    if "negative" in s:
        return "Negative"
    return "Neutral"


def postprocess(
    transcript_dataset: pd.DataFrame,
    qa_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Some custom post processing to apply for getting the complete features dataset.
    
    :param transcript_dataset: The transcript features collected.
    :param qa_dataset:         The questions and answers features collected.
    
    :returns: The processed and joined dataframe.
    """
    # Left join:
    qa_dataset.rename(columns={"text_file": "transcription_file"}, inplace=True)
    df = pd.merge(transcript_dataset, qa_dataset, how="left", on="transcription_file")
    df.dropna(inplace=True)
    
    # Clean content and extract short answers:
    for column, apply_function in [
        ("Issue", _clean_issue),
        ("is_fixed", _extract_is_fixed),
        ("customer_tone", _extract_tone),
        ("agent_tone", _extract_tone),
    ]:
        df[column] = df[column].apply(lambda s: apply_function(s))

    return df
