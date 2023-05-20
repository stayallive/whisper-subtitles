import os
import whisper
from whisper.tokenizer import LANGUAGES
from cog import BasePredictor, Input, Path, BaseModel

SUPPORTED_MODEL_NAMES = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
]


class ModelOutput(BaseModel):
    text: str
    srt_file: Path
    vtt_file: Path


class Predictor(BasePredictor):
    def predict(
            self,
            audio_path: Path = Input(description="Audio file to generate subtitles for."),
            model_name: str = Input(
                default="small.en",
                choices=SUPPORTED_MODEL_NAMES,
                description="Name of the Whisper model to use.",
            ),
            language: str = Input(
                default="en",
                choices=LANGUAGES.keys(),
                description="Language of the audio.",
            ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        if model_name.endswith(".en"):
            print("English model detected, forcing language to 'en'!")
            language = "en"

        print(f"Transcribe with {model_name} model and {LANGUAGES[language]} language.")

        model = whisper.load_model(
            model_name, download_root="whisper-cache", device="cuda"
        )

        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=True,
        )

        audio_basename = os.path.basename(str(audio_path))

        out_path_vtt = f"/tmp/{audio_basename}.{language}.vtt"
        with open(out_path_vtt, "w", encoding="utf-8") as vtt:
            vtt.write(generate_vtt(result))

        out_path_srt = f"/tmp/{audio_basename}.{language}.srt"
        with open(out_path_srt, "w", encoding="utf-8") as srt:
            srt.write(generate_srt(result))

        return ModelOutput(
            text=result["text"],
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


def generate_vtt(result: dict):
    vtt = "WEBVTT\n"
    for segment in result['segments']:
        vtt += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        vtt += f"{segment['text'].replace('-->', '->')}\n"
    return vtt


def generate_srt(result: dict):
    srt = ""
    for i, segment in enumerate(result['segments'], start=1):
        srt += f"{i}\n"
        srt += f"{format_timestamp(segment['start'], always_include_hours=True)} --> {format_timestamp(segment['end'], always_include_hours=True)}\n"
        srt += f"{segment['text'].strip().replace('-->', '->')}\n"
    return srt
