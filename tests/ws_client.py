import argparse
import asyncio
import json
import logging
import os

import soundfile as sf
import websockets

# --- Logging Setup ---
logger = logging.getLogger("websockets_client")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


async def stream_audio(uri: str, filepath: str, chunk_duration: float, output_dir: str):
    """
    Connects to the WebSocket server and streams an audio file in raw chunks.

    Args:
        uri: The WebSocket server URI.
        filepath: The path to the audio file.
        chunk_duration: The duration of each audio chunk in seconds.
        output_dir: Directory to save the collected JSON results.
    """
    session_id = f"ws-test-{os.path.basename(filepath)}"
    logger.info(f"Starting audio stream for session: {session_id}")

    try:
        # Read the entire audio file as 32-bit floating point samples.
        # This is the format the server expects.
        audio_data, sample_rate = sf.read(filepath, dtype='float32')
        total_frames = audio_data.shape[0]
        frames_per_chunk = int(chunk_duration * sample_rate)
        logger.info(f"File '{filepath}' loaded. Sample Rate: {sample_rate}, Total Frames: {total_frames}")

    except FileNotFoundError:
        logger.error(f"Error: Audio file not found at '{filepath}'")
        return
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return

    async with websockets.connect(uri, timeout=30) as websocket:
        # 1. Send Configuration
        config_message = {
            "config": {
                "sample_rate": sample_rate,
                "remove_voices": True,
                "number_of_samples": 1,
            }
        }
        await websocket.send(json.dumps(config_message))
        logger.info("Configuration sent to the server.")

        # 2. Concurrently send audio chunks and receive results
        results = []

        async def send_chunks():
            """Sends audio data in chunks."""
            for i, start_frame in enumerate(range(0, total_frames, frames_per_chunk)):
                end_frame = min(start_frame + frames_per_chunk, total_frames)
                chunk = audio_data[start_frame:end_frame]

                # Send the raw bytes of the float32 numpy array
                await websocket.send(chunk.tobytes())
                logger.info(f"Sent chunk #{i} ({len(chunk)} frames)")

                # We wait a bit to simulate a real-time stream and avoid overwhelming the server
                await asyncio.sleep(chunk_duration * 0.8)
            logger.info("All audio chunks sent.")

        async def receive_results():
            """Receives and stores prediction results."""
            try:
                async for message in websocket:
                    try:
                        payload = json.loads(message)
                        results.append(payload)
                        chunk_index = payload.get("payload", {}).get("chunk_index", "N/A")
                        logger.info(f"Received result for chunk #{chunk_index}")
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message}")
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Connection closed by server (Code: {e.code}, Reason: '{e.reason}'). This is expected.")

        # Start sending and receiving concurrently.
        # Wait for the sender to finish first.
        send_task = asyncio.create_task(send_chunks())
        receive_task = asyncio.create_task(receive_results())

        await send_task
        # Once sending is done, we can signal the server we are done
        await websocket.close(code=1000, reason="Client finished sending.")
        # Wait for the receiver to finish processing any remaining messages
        await receive_task

    # 3. Save results after the session is complete
    _save_results(results, session_id, output_dir)


def _save_results(results: list, session_id: str, output_dir: str):
    """Saves the collected results to a JSON file."""
    if not results:
        logger.warning("No results were received from the server.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"results_{session_id}.json")

    try:
        with open(output_filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results successfully saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to write results to file: {e}")


def main():
    parser = argparse.ArgumentParser(description="WebSocket audio streaming client for MED API")
    parser.add_argument("--uri", type=str, required=True, help="WebSocket URI, e.g. ws://localhost:3000/predict-stream")
    parser.add_argument("--file", type=str, required=True, help="Path to WAV audio file to stream")
    parser.add_argument("--chunk-duration", type=float, default=2.0, help="Chunk duration in seconds")
    parser.add_argument("--output-dir", type=str, default='results', help="Directory to save results JSON file")
    args = parser.parse_args()

    asyncio.run(stream_audio(args.uri, args.file, args.chunk_duration, args.output_dir))


if __name__ == '__main__':
    main()