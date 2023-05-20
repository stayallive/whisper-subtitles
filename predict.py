import os

from cog import BasePredictor, Input, Path, BaseModel
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from typing import Iterable
from whisper.tokenizer import LANGUAGES

SUPPORTED_MODEL_NAMES = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
]


class ModelOutput(BaseModel):
    preview: str
    srt_file: Path
    vtt_file: Path


class Predictor(BasePredictor):
    def predict(
            self,
            audio_path: Path = Input(
                description="Audio file to generate subtitles for.",
            ),
            model_name: str = Input(
                default="small",
                choices=SUPPORTED_MODEL_NAMES,
                description="Name of the Whisper model to use.",
            ),
            language: str = Input(
                default="en",
                choices=LANGUAGES.keys(),
                description="Language of the audio.",
            ),
            vad_filter: bool = Input(
                default=True,
                description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech.",
            ),
    ) -> ModelOutput:
        if model_name.endswith(".en") and language != "en":
            print("English only model detected, forcing language to 'en'!")
            language = "en"

        print(f"Transcribe with {model_name} model for the {LANGUAGES[language]} language...")

        model = WhisperModel(
            model_name,
            device="cuda",
            compute_type="float16",
            download_root="whisper-cache",
            local_files_only=True,
        )

        transcription, _ = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=True,
        )

        segments = []

        for segment in transcription:
            print(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)} {segment.text}")

            segments.append(segment)

        audio_basename = os.path.basename(str(audio_path)).rsplit(".", 1)[0]

        out_path_vtt = f"/tmp/{audio_basename}.{language}.vtt"
        with open(out_path_vtt, "w", encoding="utf-8") as vtt:
            vtt.write(generate_vtt(segments))

        out_path_srt = f"/tmp/{audio_basename}.{language}.srt"
        with open(out_path_srt, "w", encoding="utf-8") as srt:
            srt.write(generate_srt(segments))

        preview = " ".join([segment.text.strip() for segment in segments[:5]])
        if len(preview) > 5:
            preview += f"... (only the first 5 segments are shown, {len(segments) - 5} more segments in subtitles)"

        return ModelOutput(
            preview=preview,
            srt_file=Path(out_path_srt),
            vtt_file=Path(out_path_vtt),
        )


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def generate_vtt(result: Iterable[Segment]):
    vtt = "WEBVTT\n"
    for segment in result:
        vtt += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        vtt += f"{segment.text.replace('-->', '->')}\n"
    return vtt


def generate_srt(result: Iterable[Segment]):
    srt = ""
    for i, segment in enumerate(result, start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        srt += f"{segment.text.strip().replace('-->', '->')}\n"
    return srt
