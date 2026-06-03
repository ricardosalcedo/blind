"""LLM provider integrations."""

import asyncio
import json
import os
import random
import sys
import time

import httpx

from .config import MODEL_COSTS


def estimate_cost(model_name, in_tokens, out_tokens):
    """Estimate API cost in USD."""
    if model_name in MODEL_COSTS:
        i, o = MODEL_COSTS[model_name]
        return (in_tokens * i + out_tokens * o) / 1_000_000
    return 0.0


async def call_model(client, model_cfg, prompt, timeout=60, retries=2):
    """Call a single model with retries. Returns (content, latency_ms, in_tokens, out_tokens)."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    api_key = os.environ.get(model_cfg.get("api_key_env", ""), "")

    if not api_key and provider not in ("bedrock",):
        return None, 0, 0, 0

    for attempt in range(retries + 1):
        start = time.time()
        try:
            content, in_t, out_t = await _dispatch(client, provider, model, model_cfg, api_key, prompt, timeout)
            return content, int((time.time() - start) * 1000), in_t, out_t
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
                continue
            print(f"  ⚠ {model_cfg['id']} timed out", file=sys.stderr)
            return None, 0, 0, 0
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(1)
                continue
            print(f"  ⚠ {model_cfg['id']}: {e}", file=sys.stderr)
            return None, 0, 0, 0
    return None, 0, 0, 0


async def _dispatch(client, provider, model, cfg, api_key, prompt, timeout):
    """Dispatch to the appropriate provider. Returns (content, in_tokens, out_tokens)."""
    if provider == "openai":
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
            timeout=timeout,
        )
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    elif provider == "anthropic":
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        u = data.get("usage", {})
        return data["content"][0]["text"], u.get("input_tokens", 0), u.get("output_tokens", 0)

    elif provider == "google":
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        u = data.get("usageMetadata", {})
        return (
            data["candidates"][0]["content"]["parts"][0]["text"],
            u.get("promptTokenCount", 0),
            u.get("candidatesTokenCount", 0),
        )

    elif provider == "openai-compatible":
        base_url = cfg.get("base_url", "http://localhost:11434/v1")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
            timeout=timeout,
        )
        data = resp.json()
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    elif provider == "bedrock":
        import boto3

        region = cfg.get("region", os.environ.get("AWS_REGION", "us-west-2"))
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        br_resp = bedrock.invoke_model(modelId=model, body=body, contentType="application/json")
        data = json.loads(br_resp["body"].read())
        u = data.get("usage", {})
        return data["content"][0]["text"], u.get("input_tokens", 0), u.get("output_tokens", 0)

    raise ValueError(f"Unknown provider: {provider}")


async def call_models_parallel(models, prompt, config):
    """Call all models in parallel."""
    timeout = config.get("timeout_seconds", 60)
    retries = config.get("max_retries", 2)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[call_model(client, m, prompt, timeout, retries) for m in models])


def demo_response(model_id, prompt):
    """Generate mock response for demo mode. Returns (content, latency_ms, in_tokens, out_tokens)."""
    latency = random.randint(400, 1200)
    responses = {
        "model-alpha": (
            f"Here's a direct answer:\n\n{prompt.split()[-1].title()} involves three key principles:\n"
            "1. Simplicity in design\n2. Composability of components\n3. Clear separation of concerns\n\n"
            "Less complexity leads to more maintainable systems."
        ),
        "model-beta": (
            "Let me break this down with an example.\n\n"
            "Think of it like LEGO — each piece has a purpose, but you combine them freely.\n\n"
            "- Start with the basics\n- Layer complexity gradually\n- Test each addition independently\n\n"
            "```\nresult = compose(step1, step2, step3)\n```\n\n"
            "Good abstractions compound over time. The key is to identify the right level of "
            "abstraction for your problem domain and stick with it consistently."
        ),
        "model-gamma": (
            "Oh, this is a fun one! 🎯\n\n"
            "Most people overthink this. The secret: find patterns, make them repeatable.\n\n"
            "Think of it as a conversation between current-you and future-you. "
            "What would future-you want to know?\n\n"
            "Keep it simple, keep it human, keep iterating."
        ),
    }
    return responses.get(model_id, f"Response: {prompt}"), latency, 50, random.randint(80, 200)
