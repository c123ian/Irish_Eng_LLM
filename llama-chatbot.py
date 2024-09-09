from components.assets import arrow_circle_icon, github_icon
from components.chat import chat, chat_form, chat_message, chat_messages
import asyncio
import modal
from fasthtml.common import *
import fastapi
import requests

MODELS_DIR = "/llamas"
MODEL_NAME = "ReliableAI/UCCIX-Llama2-13B"

# Download the model weights
try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.4.3",
    "requests"
)

# Define the Modal app
app = modal.App("llama-chatbot")

# vLLM server implementation
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=5 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve_vllm():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger
    import fastapi
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import asyncio
    from typing import Optional, AsyncGenerator

    # Create a FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # Create an `AsyncLLMEngine`, the core of the vLLM server.
    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Get model config using the robust event loop handling
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    # Initialize OpenAIServingChat
    request_logger = RequestLogger(max_log_len=256)  # Adjust max_log_len as needed
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],  # served_model_names
        "assistant",   # response_role
        lora_modules=None,  # Adjust if you're using LoRA
        prompt_adapters=None,  # Adjust if you're using prompt adapters
        request_logger=request_logger,
        chat_template=None,  # Adjust if you have a specific chat template
    )

    @web_app.get("/health")
    async def health():
        """Health check."""
        try:
            await openai_serving_chat.engine.check_health()
            return fastapi.Response(status_code=200)
        except Exception as e:
            return fastapi.Response(content=str(e), status_code=500)

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 100, stream: bool = False):
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(max_tokens=max_tokens)
        
        async def completion_generator() -> AsyncGenerator[str, None]:
            try:
                async for result in engine.generate(prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        yield result.outputs[0].text
            except Exception as e:
                yield str(e)

        if stream:
            return StreamingResponse(completion_generator(), media_type="text/event-stream")
        else:
            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            return JSONResponse(content={"choices": [{"text": completion}]})

    return web_app

# FastHTML web interface implementation
@app.function(
    image=image,
)
@modal.asgi_app()
def serve_fasthtml():
    fasthtml_app, rt = fast_app(ws_hdr=True)

    @rt("/")
    async def get():
        return Div(
            H1("Chat with LLaMA Model"),
            chat(),
            cls="flex flex-col items-center min-h-screen bg-gray-100",
        )

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, send):
        chat_messages.append({"role": "user", "content": msg})
        await send(chat_form(disabled=True))
        await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))

        # Correctly pass the prompt in the GET request
        vllm_url = f"https://c123ian--llama-chatbot-serve-vllm.modal.run/v1/completions"
        response = requests.get(vllm_url, params={"prompt": msg, "max_tokens": 100})

        if response.status_code == 200:
            message = response.json()["choices"][0]["text"]
        else:
            message = "Error: Unable to get response from LLM."

        chunks = [message[i:i+10] for i in range(0, len(message), 10)]
        chat_messages.append({"role": "assistant", "content": ""})
        await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))
        for chunk in chunks:
            chat_messages[-1]["content"] += chunk
            await send(Span(chunk, id=f"msg-content-{len(chat_messages)-1}", hx_swap_oob="beforeend"))
            await asyncio.sleep(0.2)
        await send(chat_form(disabled=False))

    return fasthtml_app


if __name__ == "__main__":
    serve_vllm()  # Serve the vLLM server
    serve_fasthtml()  # Serve the FastHTML web interface



