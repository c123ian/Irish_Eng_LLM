### Base Model

Using UCCIX-Llama2-13B, an Irish-English bilingual model based on Llama 2-13B
Capable of understanding both languages and outperforms larger models on Irish language tasks

- Available at: https://huggingface.co/ReliableAI/UCCIX-Llama2-13B

Key aspects of the final code:

1. Modal app configuration: We define a Modal app named "llama-chatbot" and set up the necessary image with dependencies.

2. Volume setup: We use a Modal volume to store and access the model weights efficiently.

3. vLLM server: Implemented as a Modal function with GPU resources allocated. It handles the core language model inference.

4. FastHTML interface: Another Modal function that serves the web interface for user interaction.

5. Deployment: Both the vLLM server and FastHTML interface are deployed as ASGI apps, allowing them to run concurrently.

To deploy this application, you would run:

```
modal deploy llama-chatbot.py
```

This command tells Modal to deploy both the vLLM server and the FastHTML interface as defined in the script. Modal handles the complexities of serverless deployment, including GPU allocation, scaling, and networking.

![image](https://github.com/user-attachments/assets/3117196d-8ed4-412a-9e77-3f929ea6f843)


### note:
- This is essentially a combination of this template (which just echos back user input) https://github.com/arihanv/fasthtml-modal and Modal Labs [Run an OpenAI-Compatible vLLM Server](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_inference.py) tutorial. 
- Using OpenAI API's  "/v1/completions" rather then the more apporpriate "/v1/chat/completions", see where code was sourced [here]( https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1)
- This uses FastHTML's websockest `/ws` rather then [FastHTML SSE](https://github.com/AnswerDotAI/fasthtml-example/blob/main/04_sse/sse_rand_scroll.py)
- This uses UCCIX-Llama2-13B, you may need to request permission via Huggingface Hub and run the `download_llama.py` script in order to download weights onto a Modla labs Volume (which we call here `/llamas`).
- The code generate two URLs, one is the backend running on a Modal labs GPU, the second is the front-end (the FastHTML GUI running on a Modal Labs CPUT), I have yet to add some user affordance to alert user they have to wait for initial cold-boot response. 
