import asyncio
import modal
from fasthtml.common import *
import fastapi
import aiohttp
import uuid
from typing import Optional, AsyncGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model via Huggingface Hub, will need to run download script if not already stored on Modal /volume
MODELS_DIR = "/llamas"
MODEL_NAME = "ReliableAI/UCCIX-Llama2-13B"

# Modal setup
try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.4.3",
    "aiohttp",
    "fastapi",
    "uvicorn"
)

app = modal.App("llama-chatbot")

# FastHTML setup
fasthtml_app, rt = fast_app(
    hdrs=(
        Script(src="https://cdn.tailwindcss.com"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
    ),
    ws_hdr=True
)

chat_messages = []

def chat_input(disabled=False):
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        disabled=disabled,
        cls="input input-bordered w-full max-w-xs"
    )

def chat_button(disabled=False):
    return Button(
        "Send",
        id="send-button",
        disabled=disabled,
        cls="btn btn-primary"
    )

def chat_form(disabled=False):
    return Form(
        chat_input(disabled),
        chat_button(disabled),
        id="form",
        ws_send=True,
        cls="flex gap-2 items-center border-t border-base-300 p-2"
    )

def chat_message(msg_idx):
    msg = chat_messages[msg_idx]
    is_user = msg['role'] == 'user'
    return Div(
        Div(msg["role"], cls="chat-header opacity-50"),
        Div(msg["content"], 
            cls=f"chat-bubble chat-bubble-{'primary' if is_user else 'secondary'}",
            id=f"msg-content-{msg_idx}"),
        id=f"msg-{msg_idx}",
        cls=f"chat chat-{'end' if is_user else 'start'}"
    )

def chat_window():
    return Div(
        id="messages",
        *[chat_message(i) for i in range(len(chat_messages))],
        cls="flex flex-col gap-2 p-4 h-[45vh] overflow-y-auto w-full",
        style="scroll-behavior: smooth;"
    )

def chat():
    return Div(
        Div("Ask me to translate", cls="text-xs font-mono absolute top-0 left-0 w-fit p-1 bg-red border-b border-r border-red-500 rounded-tl-md rounded-br-md font-celtic"),
        chat_window(),
        chat_form(),
        hx_ext="ws", ws_connect="/ws",
        cls="flex flex-col w-full max-w-2xl border border-base-300 h-full rounded-box shadow-lg relative bg-base-100"
    )

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
    # Import vLLM-related modules here
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    request_logger = RequestLogger(max_log_len=256)
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],
        "assistant",
        lora_modules=None,
        prompt_adapters=None,
        request_logger=request_logger,
        chat_template=None,
    )

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 100, stream: bool = False):
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stop=["Human:", "\n\n"]
        )
        
        system_prompt = "You are a helpful Irish English translator and tutor. Respond concisely and stay on topic."
        full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        
        async def completion_generator() -> AsyncGenerator[str, None]:
            try:
                full_response = ""
                async for result in engine.generate(full_prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        new_text = result.outputs[0].text
                        if not full_response:
                            new_text = new_text.split("Assistant:")[-1].lstrip()
                        
                        new_part = new_text[len(full_response):]
                        full_response = new_text
                        
                        if new_part:
                            yield new_part
                        
                        if full_response.strip().endswith((".", "!", "?")):
                            break
            except Exception as e:
                logger.error(f"Error in completion_generator: {e}")
                yield str(e)

        if stream:
            return fastapi.responses.StreamingResponse(completion_generator(), media_type="text/event-stream")
        else:
            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            return fastapi.responses.JSONResponse(content={"choices": [{"text": completion.strip()}]})

    return web_app

# FastHTML web interface implementation
@app.function(image=image)
@modal.asgi_app()
def serve_fasthtml():
    @rt("/")
    async def get():
        return Div(
            H1("Chat with Irish English translator and tutor bot"),
            chat(),
            cls="flex flex-col items-center min-h-screen bg-red-100"
        )

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, send):
        chat_messages.append({"role": "user", "content": msg})
        await send(chat_form(disabled=True))
        await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))
        await send(Script("document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;"))

        vllm_url = f"https://c123ian--llama-chatbot-serve-vllm.modal.run/v1/completions"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(vllm_url, params={
                    "prompt": msg, 
                    "max_tokens": 100, 
                    "stream": "true"
                }, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        chat_messages.append({"role": "assistant", "content": ""})
                        message_index = len(chat_messages) - 1
                        await send(Div(chat_message(message_index), id="messages", hx_swap_oob="beforeend"))
                        
                        async for chunk in response.content.iter_any():
                            if chunk:
                                text = chunk.decode('utf-8')
                                chat_messages[message_index]["content"] += text
                                await send(Span(text, id=f"msg-content-{message_index}", hx_swap_oob="beforeend"))
                        
                        await send(Script("document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;"))
                    else:
                        logger.error(f"vLLM server responded with status {response.status}")
                        logger.error(f"Response content: {await response.text()}")
                        error_message = f"Error: Unable to get response from LLM. Status: {response.status}"
                        chat_messages.append({"role": "assistant", "content": error_message})
                        await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))
                        await send(Script("document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;"))
            except aiohttp.ClientError as e:
                logger.error(f"Error connecting to vLLM server: {e}")
                error_message = "Error: Unable to connect to the LLM server."
                chat_messages.append({"role": "assistant", "content": error_message})
                await send(Div(chat_message(len(chat_messages) - 1), id="messages", hx_swap_oob="beforeend"))

        await send(chat_form(disabled=False))

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
