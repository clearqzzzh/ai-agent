"""
千问 VL 视觉识别服务
=====================
- 调用 DashScope 原生多模态 API (qwen-vl-plus)
- 支持 ws 密钥
- 医疗类图片自动附加免责声明
"""

import httpx
from loguru import logger
from config import QWEN_API_KEY, QWEN_MODEL

MEDICAL_DISCLAIMER = "\n\n⚠️ 我不是兽医，以上识别仅供参考，涉及宠物健康问题建议及时就医。"

# DashScope 原生多模态 API
QWEN_NATIVE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


async def analyze_image(image_url: str, question: str = "") -> str:
    if not QWEN_API_KEY:
        return "图片识别服务未配置（缺少 QWEN_API_KEY）"

    # 提取 base64 数据（去掉 data:xxx;base64, 前缀如果有）
    img_data = image_url
    if image_url.startswith("data:"):
        img_data = image_url.split(",", 1)[-1] if "," in image_url else image_url

    # 判断是纯 base64 还是 URL
    if img_data.startswith("http://") or img_data.startswith("https://"):
        img_payload = {"image": img_data}
    else:
        img_payload = {"image": f"data:image/jpeg;base64,{img_data}"}

    payload = {
        "model": QWEN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        img_payload,
                        {"text": question or "请详细描述这张图片的内容"},
                    ],
                }
            ]
        },
        "parameters": {"result_format": "message"},
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                QWEN_NATIVE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # DashScope 原生响应格式
            output = data.get("output", {})
            choices = output.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                # content 可能是 list 或 string
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
            else:
                content = ""

            if not content:
                content = "未能从图片中识别出有意义的内容，请换一张清晰的图片重试。"

            # 医疗相关加免责
            kw = ["症状", "皮肤", "红肿", "脱毛", "拉稀", "呕吐", "生病", "伤口", "流血", "异常", "疾病", "过敏"]
            if any(w in str(content) or w in question for w in kw):
                content += MEDICAL_DISCLAIMER

            return content

    except httpx.TimeoutException:
        return "图片识别超时，请换一张小一点的图片重试"
    except Exception as e:
        logger.warning(f"图片识别失败: {e}")
        return f"抱歉，无法识别这张图片。请换一张清晰的图片重试，或直接描述您的问题。"
