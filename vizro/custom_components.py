"""Custom components for Vizro dashboard extensions.
"""

import base64
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from dash import html
from dash.exceptions import PreventUpdate

import vizro.models as vm
from vizro.models.types import capture


@capture("action")
def update_from_selected_row(
    selected_rows: Sequence[dict[str, Any]]
) -> tuple[str, str]:
    """Update transcript and audio from the selected row in the grid.

    Args:
        selected_rows (Sequence[dict[str, Any]]):
            List of selected row dictionaries from the grid, each containing 'text_file' and 'audio_file' keys.

    Returns:
        tuple[str, str]:
            A tuple containing:
                - The transcript as markdown-formatted string.
                - The audio source as a base64-encoded string suitable for HTML audio playback.

    Raises:
        PreventUpdate: If the required files are not found or cannot be read.
    """
    selected_row = selected_rows[0]
    text_file_path = Path(f"outputs/anonymized_files/{selected_row['text_file']}")
    audio_file_path = Path(f"outputs/audio_files/{selected_row['audio_file']}")
    if (
        text_file_path not in Path("outputs/anonymized_files").iterdir()
        or audio_file_path not in Path("outputs/audio_files").iterdir()
    ):
        raise PreventUpdate
    try:
        call_transcript = text_file_path.read_text()
    except Exception as e:
        raise PreventUpdate from e
    call_transcript = call_transcript.replace("\n", "  \n")
    call_transcript = re.sub(r"^(\w+)", r"**\1**", call_transcript, flags=re.MULTILINE)
    try:
        call_audio_src = base64.b64encode(audio_file_path.read_bytes())
    except Exception as e:
        raise PreventUpdate from e
    call_audio_src = f"data:audio/wav;base64,{call_audio_src.decode('utf-8')}"
    return call_transcript, call_audio_src


class Audio(vm.VizroBaseModel):
    """Audio component for Vizro dashboard.

    This component renders an audio player for playback of call recordings or other audio content.
    """

    type: Literal["audio"] = "audio"

    def build(self) -> html.Audio:
        """Build the Dash Audio component for playback.

        Returns:
            html.Audio: Dash HTML audio component with controls enabled.
        """
        return html.Audio(id=self.id, controls=True)


vm.Container.add_type("components", Audio)


def make_tabs_with_title(title: str, tabs: list[vm.Container]) -> vm.Container:
    """Create a container with a title and tabbed content for the Vizro dashboard.

    Args:
        title (str):
            The title to display above the tabbed content.
        tabs (list[vm.Container]):
            List of vm.Container objects, each representing a tab.

    Returns:
        vm.Container: A container with a title and tabbed content, styled for the dashboard.
    """
    return vm.Container(
        title=title, components=[vm.Tabs(tabs=tabs)], variant="filled", collapsed=False
    )
