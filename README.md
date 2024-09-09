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
