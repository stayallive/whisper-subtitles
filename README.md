# Whisper Subtitles

Generate subtitles (`.srt` and `.vtt`) from audio files using [OpenAI's Whisper](https://github.com/openai/whisper) models.

Using [faster-whisper](https://github.com/guillaumekln/faster-whisper), a reimplementation of [OpenAI's Whisper](https://github.com/openai/whisper) model using CTranslate2, which is a fast inference engine for Transformer models.

This is a fork of [m1guelpf/whisper-subtitles](https://replicate.com/m1guelpf/whisper-subtitles) with added support for VAD, selecting a language, use the language specific models and download the `.vtt`/`.srt` files directly from the result.

## Usage

You can run the model on [Replicate](https://replicate.com/stayallive/whisper-subtitles).

## Development

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i audio_path=@/path/to/audio.mp3

Or, build a Docker image:

    cog build

Or, [push it to Replicate](https://replicate.com/docs/guides/push-a-model):

    cog push r8.im/...
