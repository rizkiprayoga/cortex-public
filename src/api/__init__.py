"""
src.api — FastAPI dashboard for the HMM + LSTM trading bot.

Serves REST endpoints + SSE live stream on the same asyncio loop
as the trading tick. No IPC — handlers read live state directly
from shared Python objects.
"""
