import asyncio
import modal
from fasthtml.common import *
import fastapi
import aiohttp
from typing import AsyncGenerator

MODELS_DIR = "/llamas"
MODEL_NAME = "ReliableAI/UCCIX-Llama2-13B"

try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.6.2",
    "aiohttp",
    "fastapi",
    "uvicorn"
)

app = modal.App("llama-chatbot")

fasthtml_app, rt = fast_app(
    hdrs=(
        Script(src="https://cdn.tailwindcss.com"),
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css")
    )
)

chat_messages = []
message_count = 0

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
        hx_post="/chat",
        hx_target="#messages",
        hx_swap="beforeend",
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
        Div("Ask me to translate", cls="text-xs font-mono absolute top-0 left-0 w-fit p-1 bg-red border-b border-red-500 rounded-tl-md rounded-br-md font-celtic"),
        Div("The first message may take a while to process as the model loads.", 
            id="initial-message", 
            cls="text-sm font-mono w-full p-2 bg-yellow-100 border-b border-yellow-300 hidden"),
        chat_window(),
        chat_form(),
        cls="flex flex-col w-full max-w-2xl border border-base-300 h-full rounded-box shadow-lg relative bg-base-100"
    )

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
    import uuid
    import traceback

    web_app = fastapi.FastAPI()

    print(f"Initializing AsyncLLMEngine with model: {MODELS_DIR}/{MODEL_NAME}")
    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("AsyncLLMEngine initialized successfully")

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 100):
        print(f"Received prompt: {prompt}")
        try:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["Human:", "\n\n"]
            )
            
            system_prompt = "You are a helpful Irish English translator and tutor. Respond concisely and stay on topic."
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            request_id = str(uuid.uuid4())
            print(f"Generated request_id: {request_id}")
            
            async def completion_generator():
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
                    print(f"Error in completion_generator: {e}")
                    print(traceback.format_exc())
                    yield f"Error: {str(e)}"

            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            print(f"Generated completion: {completion}")
            return fastapi.responses.JSONResponse(content={"choices": [{"text": completion.strip()}]})
        except Exception as e:
            print(f"Error in get_completions: {e}")
            print(traceback.format_exc())
            return fastapi.responses.JSONResponse(
                status_code=500,
                content={"error": f"An error occurred while processing your request: {str(e)}"}
            )

    return web_app

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

    def message_preview(msg_idx):
        if msg_idx < len(chat_messages) and chat_messages[msg_idx]['role'] == 'assistant':
            return chat_message(msg_idx)
        else:
            return Div(
                Div("assistant", cls="chat-header opacity-50"),
                Div(Span(cls="loading loading-dots loading-sm"), cls="chat-bubble chat-bubble-secondary"),
                id=f"msg-{msg_idx}",
                cls="chat chat-start",
                hx_get=f"/chat/{msg_idx}",
                hx_trigger="every 1s",
                hx_swap="outerHTML"
            )

    @rt("/chat")
    async def post(msg: str):
        global message_count
        message_count += 1
        
        chat_messages.append({"role": "user", "content": msg})
        user_message = chat_message(len(chat_messages) - 1)
        
        assistant_preview = message_preview(len(chat_messages))
        
        clear_input = Input(id="msg-input", name="msg", placeholder="Type a message", hx_swap_oob="true")
        
        asyncio.create_task(generate_response(msg, len(chat_messages)))
        
        if message_count == 1:
            show_initial_message = Script("document.getElementById('initial-message').classList.remove('hidden'); setTimeout(() => document.getElementById('initial-message').classList.add('hidden'), 5000);")
            return user_message, assistant_preview, clear_input, show_initial_message
        else:
            return user_message, assistant_preview, clear_input

    @rt("/chat/{msg_idx}")
    async def get(msg_idx: int):
        return message_preview(msg_idx)

    async def generate_response(msg: str, msg_idx: int):
        vllm_url = f"https://c123ian--llama-chatbot-serve-vllm.modal.run/v1/completions"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(vllm_url, params={
                    "prompt": msg, 
                    "max_tokens": 100
                }, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        assistant_response = data['choices'][0]['text']
                        chat_messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        error_message = f"Error: Unable to get response from LLM. Status: {response.status}"
                        chat_messages.append({"role": "assistant", "content": error_message})
            except aiohttp.ClientError as e:
                error_message = f"Error: Unable to connect to the LLM server. {str(e)}"
                chat_messages.append({"role": "assistant", "content": error_message})

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
