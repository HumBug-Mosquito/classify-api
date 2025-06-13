import argparse
import asyncio
import io
import json
import logging
import os
from datetime import datetime

import soundfile as sf
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_client")

async def stream_audio(uri: str, filepath: str, chunk_duration: float = 1.0, output_dir: str = '.'):
    """
    Connects to the WebSocket server, streams audio file in WAV chunks of given duration,
    collects incoming JSON results, and writes them to a JSON file per session.
    """
    # Read entire file
    data, sample_rate = sf.read(filepath, dtype='int16')  # read as PCM int16
    total_frames = data.shape[0]
    frames_per_chunk = int(chunk_duration * sample_rate)

    logger.info(f"File: {filepath}, sample rate: {sample_rate}, total_frames: {total_frames}")

    results = []
    session_id = None

    async with websockets.connect(uri) as ws:
        # Receive initial session_id
        greeting = await ws.recv()
        try:
            info = json.loads(greeting)
            session_id = info.get('session_id')
            logger.info(f"Session ID: {session_id}")
        except json.JSONDecodeError:
            logger.warning(f"Unexpected greeting: {greeting}")
            session_id = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"results_{session_id}.json")

        # Tasks: send and receive concurrently
        async def send_chunks():
            seq = 0
            for start in range(0, total_frames, frames_per_chunk):
                end = min(start + frames_per_chunk, total_frames)
                chunk = data[start:end]
                # Write chunk to WAV in-memory
                buf = io.BytesIO()
                sf.write(buf, chunk, sample_rate, format='WAV')
                buf.seek(0)
                frame_bytes = buf.read()
                await ws.send(frame_bytes)
                logger.info(f"Sent chunk #{seq}, frames {start}-{end}")
                seq += 1
                await asyncio.sleep(chunk_duration)
            # After sending all frames, close the send side
            await ws.close()
            logger.info("All chunks sent, WebSocket closed for sending.")

        async def receive_results():
            try:
                async for message in ws:
                    try:
                        payload = json.loads(message)
                        results.append(payload)
                        logger.info(f"Received result seq={payload.get('sequence')}")
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON message: {message}")
            except websockets.ConnectionClosed:
                logger.info("WebSocket connection closed by server or client.")

        await asyncio.gather(send_chunks(), receive_results())

    # After connection closes, write results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results written to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results file: {e}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket audio streaming client for mosquito detection API")
    parser.add_argument("--url", type=str, required=True, help="WebSocket URL, e.g. ws://localhost:8000/ws/stream?token=...")
    parser.add_argument("--file", type=str, required=True, help="Path to WAV file to stream")
    parser.add_argument("--chunk-duration", type=float, default=2.0, help="Chunk duration in seconds")
    parser.add_argument("--output-dir", type=str, default='.', help="Directory to save results JSON file")
    args = parser.parse_args()

    asyncio.run(stream_audio(args.url, args.file, args.chunk_duration, args.output_dir))


if __name__ == '__main__':
    main()
