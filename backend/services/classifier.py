import aiohttp

INSTRUCTION_SUFFIX = """
Return ONLY a valid JSON object. No markdown. No backticks. No explanation.
The JSON must be parseable by Python json.loads().
"""


async def call_anthropic(b64: str, mime: str, prompt: str, api_key: str, model: str) -> str:
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ValueError(f"Anthropic {resp.status}: {data.get('error', {}).get('message', str(data))}")
            return data["content"][0]["text"]


async def call_google(b64: str, mime: str, prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {"text": prompt},
                ]
            }
        ],
        "safetySettings": [
            {"category": cat, "threshold": "BLOCK_ONLY_HIGH"}
            for cat in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ValueError(f"Google {resp.status}: {data.get('error', {}).get('message', str(data))}")
            return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_openai(b64: str, mime: str, prompt: str, api_key: str, model: str) -> str:
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ValueError(f"OpenAI {resp.status}: {data.get('error', {}).get('message', str(data))}")
            return data["choices"][0]["message"]["content"]


async def call_ollama(b64: str, mime: str, prompt: str, endpoint: str, model: str) -> str:
    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "images": [b64], "stream": False}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ValueError(f"Ollama {resp.status}: {data}")
            return data["response"]
