"""
Streaming support for LangChain agent responses.

Provides Server-Sent Events (SSE) generators for streaming output.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


logger = logging.getLogger(__name__)


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler for streaming LLM tokens to a queue.

    Used to capture streaming output from LangChain and forward
    to SSE response generators.
    """

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue
        self._done = asyncio.Event()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle new token from LLM."""
        try:
            self.queue.put_nowait({"type": "token", "content": token})
        except asyncio.QueueFull:
            logger.warning("Streaming queue full, dropping token")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM completion."""
        self.queue.put_nowait({"type": "end", "content": ""})
        self._done.set()

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle LLM error."""
        self.queue.put_nowait({"type": "error", "content": str(error)})
        self._done.set()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Handle chain completion."""
        pass

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Handle tool invocation start."""
        tool_name = serialized.get("name", "unknown")
        self.queue.put_nowait({
            "type": "tool_start",
            "content": f"Using tool: {tool_name}"
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Handle tool completion."""
        self.queue.put_nowait({"type": "tool_end", "content": output[:200]})

    @property
    def is_done(self) -> bool:
        return self._done.is_set()


async def create_sse_generator(
    run_fn: Callable[[], Any],
    timeout: float = 60.0,
) -> AsyncGenerator[str, None]:
    """
    Create an SSE generator from a synchronous run function.

    Args:
        run_fn: Function that runs the agent (should not be async).
        timeout: Maximum time to wait for completion.

    Yields:
        SSE-formatted event strings.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    handler = StreamingCallbackHandler(queue)

    async def run_in_executor() -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_fn)

    # Start the agent run in background
    task = asyncio.create_task(run_in_executor())

    try:
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                yield format_sse_event("error", {"message": "Timeout exceeded"})
                break

            # Check if task completed
            if task.done():
                # Drain remaining queue items
                while not queue.empty():
                    try:
                        event = queue.get_nowait()
                        yield format_sse_event(event["type"], {"content": event["content"]})
                    except asyncio.QueueEmpty:
                        break

                # Get result or error
                try:
                    result = task.result()
                    yield format_sse_event("complete", {"result": str(result)})
                except Exception as e:
                    yield format_sse_event("error", {"message": str(e)})
                break

            # Get next event from queue
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield format_sse_event(event["type"], {"content": event["content"]})

                if event["type"] in ("end", "error"):
                    break
            except asyncio.TimeoutError:
                continue

    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """
    Format data as an SSE event.

    Args:
        event_type: Event type identifier.
        data: Event data dictionary.

    Returns:
        SSE-formatted string.
    """
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def stream_agent_response(
    session_data: Any,
    query: str,
    on_token: Optional[Callable[[str], None]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent response as SSE events.

    Args:
        session_data: SessionData containing the ChatSession.
        query: User query to process.
        on_token: Optional callback for each token.

    Yields:
        SSE-formatted event strings.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    result_holder: list = []
    error_holder: list = []

    def run_agent() -> str:
        try:
            result = session_data.chat_session.send(query)
            result_holder.append(result)
            return result
        except Exception as e:
            error_holder.append(e)
            raise

    # Send initial event
    yield format_sse_event("start", {"query": query})

    # Run agent in thread pool
    loop = asyncio.get_event_loop()

    try:
        # Execute synchronously since ChatSession.send is blocking
        result = await loop.run_in_executor(None, run_agent)

        # Stream the result character by character for demo
        # (In production, this would be integrated with LLM streaming)
        for i, char in enumerate(result):
            yield format_sse_event("token", {"content": char})
            if on_token:
                on_token(char)
            # Small delay for demonstration
            if i % 10 == 0:
                await asyncio.sleep(0.01)

        yield format_sse_event("complete", {"result": result})

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield format_sse_event("error", {"message": str(e)})
