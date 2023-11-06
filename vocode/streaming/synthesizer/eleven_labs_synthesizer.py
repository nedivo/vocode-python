import asyncio
import logging
from typing import AsyncGenerator, List, Optional, Tuple, Union

import aiohttp
from opentelemetry.trace import Span

from vocode import getenv
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.synthesizer.base_synthesizer import (
    FILLER_AUDIO_PATH,
    BaseSynthesizer,
    FillerAudio,
    SynthesisResult,
    encode_as_wav,
    tracer,
)
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker
from vocode.streaming.utils.mp3_helper import decode_mp3

ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"


class ElevenLabsFillerAudioPhrase(BaseMessage):
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None


# Set the stability higher to get more even, unexcited feel.
ELEVENLABS_FILLER_PHRASES = [
    ElevenLabsFillerAudioPhrase(
        text="Um—",
        stability=0.8,
        similarity_boost=0.1,  # Em-dash
    ),
    ElevenLabsFillerAudioPhrase(
        text="Uh—",
        stability=0.8,
        similarity_boost=0.1,  # Em-dash
    ),
    ElevenLabsFillerAudioPhrase(
        text="Uh-huh—",
        stability=0.8,
        similarity_boost=0.1,  # Em-dash
    ),
    ElevenLabsFillerAudioPhrase(
        text="Mm-hmm-",
        stability=0.8,
        similarity_boost=0.1,  # En-dash
    ),
    ElevenLabsFillerAudioPhrase(
        text="Hmm—",
        stability=0.8,
        similarity_boost=0.1,  # Em-dash
    ),
    ElevenLabsFillerAudioPhrase(
        text="Hmm-",
        stability=0.8,
        similarity_boost=0.1,  # En-dash
    ),
    ElevenLabsFillerAudioPhrase(text="Okay...", stability=0.8, similarity_boost=0.1),
    ElevenLabsFillerAudioPhrase(text="Right...", stability=0.8, similarity_boost=0.1),
    ElevenLabsFillerAudioPhrase(
        text="Let me see...", stability=0.8, similarity_boost=0.1
    ),
]


class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.logger = logger or logging.getLogger(__name__)

    async def experimental_streaming_output_generator(
        self,
        response: aiohttp.ClientResponse,
        chunk_size: int,
        create_speech_span: Optional[Span],
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        miniaudio_worker_input_queue: asyncio.Queue[
            Union[bytes, None]
        ] = asyncio.Queue()
        miniaudio_worker_output_queue: asyncio.Queue[
            Tuple[bytes, bool]
        ] = asyncio.Queue()
        miniaudio_worker = MiniaudioWorker(
            self.synthesizer_config,
            chunk_size,
            miniaudio_worker_input_queue,
            miniaudio_worker_output_queue,
        )
        miniaudio_worker.start()
        stream_reader = response.content

        # Create a task to send the mp3 chunks to the MiniaudioWorker's input
        # queue in a separate loop
        async def send_chunks():
            async for chunk in stream_reader.iter_any():
                miniaudio_worker.consume_nonblocking(chunk)
            miniaudio_worker.consume_nonblocking(None)  # sentinel

        try:
            asyncio.create_task(send_chunks())

            # Await the output queue of the MiniaudioWorker and yield the wav
            # chunks in another loop
            while True:
                # Get the wav chunk and the flag from the output queue of the
                # MiniaudioWorker
                wav_chunk, is_last = await miniaudio_worker.output_queue.get()
                if self.synthesizer_config.should_encode_as_wav:
                    wav_chunk = encode_as_wav(wav_chunk, self.synthesizer_config)

                yield SynthesisResult.ChunkResult(wav_chunk, is_last)
                # If this is the last chunk, break the loop
                if is_last and create_speech_span is not None:
                    create_speech_span.end()
                    break
        except asyncio.CancelledError:
            pass
        finally:
            miniaudio_worker.terminate()

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        orig_experimental_streaming = self.experimental_streaming
        self.experimental_streaming = False
        filler_phrase_audios = []
        for filler_phrase in ELEVENLABS_FILLER_PHRASES:
            cache_key = "-".join(
                (
                    str(filler_phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.voice_id),
                    str(self.model_id),
                    str(filler_phrase.stability),
                    str(filler_phrase.similarity_boost),
                    str(self.words_per_minute),
                )
            )
            filler_audio_path = FILLER_AUDIO_PATH / f"{cache_key}.bytes"
            if filler_audio_path.exists():
                audio_data = filler_audio_path.open("rb").read()
            else:
                self.logger.debug("Generating filler audio for %s", filler_phrase.text)

                response = await self.elevenlabs_request(
                    filler_phrase.text,
                    filler_phrase.stability,
                    filler_phrase.similarity_boost,
                )

                audio_data = await response.read()
            wav_bytes = decode_mp3(audio_data).getvalue()

            if not filler_audio_path.exists():
                filler_audio_path.write_bytes(wav_bytes)

            filler_synthesizer_config = self.synthesizer_config.copy(
                update={
                    "stability": filler_phrase.stability,
                    "similarity_boost": filler_phrase.similarity_boost,
                },
            )
            filler_phrase_audios.append(
                FillerAudio(
                    filler_phrase,
                    wav_bytes,
                    filler_synthesizer_config,
                )
            )
        self.experimental_streaming = orig_experimental_streaming
        return filler_phrase_audios

    async def elevenlabs_request(
        self,
        text: str,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
    ):
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)

        # Default to the values in the synthesizer config given to __init__
        if stability is None and self.stability is not None:
            stability = self.stability
        if similarity_boost is None and self.similarity_boost is not None:
            similarity_boost = self.similarity_boost
        if stability is not None and similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=stability, similarity_boost=similarity_boost
            )

        url = ELEVEN_LABS_BASE_URL + f"text-to-speech/{self.voice_id}"

        if self.experimental_streaming:
            url += "/stream"

        if self.optimize_streaming_latency:
            url += f"?optimize_streaming_latency={self.optimize_streaming_latency}"
        headers = {"xi-api-key": self.api_key}
        body = {
            "text": text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        session = self.aiohttp_session

        response = await session.request(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        if not response.ok:
            raise Exception(f"ElevenLabs API returned {response.status} status code")
        return response

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        type_str = SynthesizerType.ELEVEN_LABS.value.split("_", 1)[-1]
        create_speech_span = tracer.start_span(
            f"synthesizer.{type_str}.create_total",
        )
        response = await self.elevenlabs_request(
            message.text, self.stability, self.similarity_boost
        )
        if self.experimental_streaming:
            return SynthesisResult(
                self.experimental_streaming_output_generator(
                    response, chunk_size, create_speech_span
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
            )
        else:
            audio_data = await response.read()
            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{type_str}.convert",
            )
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            return result
