"""interaction with audio"""

import whisperx
from whisperx import types as whisperx_types
from machine import get_optimal_device,get_optimal_compute_type,clear_gpu,T_Device,T_Compute_Type

def transcribe(
        audio_filepath:str,
        hf_access_token:str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 16,
        device: T_Device | None = None,
        compute_type: T_Compute_Type | None = None,
        debug_mode: bool = False,
    ) -> whisperx_types.AlignedTranscriptionResult :
        """transcribe the target audio"""

        ## assign default arguments

        if device == None:
            device = get_optimal_device()

        if compute_type == None:
            compute_type = get_optimal_compute_type()


        #load target audio
        audio = whisperx.load_audio(audio_filepath)

        #transcribe
        model_transcribe = whisperx.load_model("base",device,compute_type=compute_type)
        result = model_transcribe.transcribe(audio=audio,batch_size=batch_size)

        if debug_mode:
            print("TRANSCRIBED")
        clear_gpu()


        #align whisper output
        model_alignment, alignment_metadata = whisperx.load_align_model(language_code=result["language"],device=device)
        result = whisperx.align(result["segments"], model=model_alignment, align_model_metadata=alignment_metadata, audio=audio, device=device, return_char_alignments=False)

        if debug_mode:
            print("ALIGNED")
        clear_gpu()

        #diarize
        model_diarize = whisperx.DiarizationPipeline(use_auth_token=hf_access_token, device=device)
        diarized_segments = model_diarize(audio=audio,min_speakers=min_speakers,max_speakers=max_speakers)

        if debug_mode:
            print("DIARIZED")

        whisperx.assign_word_speakers(diarized_segments,result)

        return result


def diarize(
        audio_filepath:str,
        transcription_result:whisperx_types.AlignedTranscriptionResult | whisperx_types.TranscriptionResult,
        hf_access_token:str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: T_Device | None = None,
        debug_mode: bool = False,
    ):
        """diarize an already transcribed audio-file"""
        
        if device == None:
            device = get_optimal_device()

        #load target audio
        audio = whisperx.load_audio(audio_filepath)

        #diarize
        model_diarize = whisperx.DiarizationPipeline(use_auth_token=hf_access_token, device=device)
        diarized_segments = model_diarize(audio=audio,min_speakers=min_speakers,max_speakers=max_speakers)

        if debug_mode:
            print("DIARIZED")

        whisperx.assign_word_speakers(diarized_segments,transcription_result)

        return transcription_result


def transcribe_diarized(
        audio_filepath:str,
        hf_access_token:str,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        batch_size: int = 16,
        device: T_Device | None = None,
        compute_type: T_Compute_Type | None = None,
        debug_mode: bool = False,
    ) -> whisperx_types.AlignedTranscriptionResult:
        """transcribe & diarize the specified audio-file"""
        
        ## default args
        if device == None:
            device = get_optimal_device()

        if compute_type == None:
            compute_type = get_optimal_compute_type()

        #transcribe
        aligned_transcription = transcribe(
             audio_filepath=audio_filepath,
             hf_access_token=hf_access_token,
             min_speakers=min_speakers,
             max_speakers=max_speakers,
             batch_size=batch_size,
             device=device,
             compute_type=compute_type,
             debug_mode=debug_mode,
        )

        
        #diarize
        diarized_transcription = diarize(
             audio_filepath=audio_filepath,
             transcription_result=aligned_transcription,
             hf_access_token=hf_access_token,
             min_speakers=min_speakers,
             max_speakers=max_speakers,
             device=device,
             debug_mode=debug_mode,
        )


        return diarized_transcription


     
     