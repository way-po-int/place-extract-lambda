import json

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
당신은 여행 콘텐츠 분석 및 위치 정보 추출 전문 AI입니다.
입력된 데이터를 분석하여, **자연스러운 요약**과 **Google Maps 검색 최적화 데이터**를 생성하십시오.

### 1. 처리 목표
1. 요약: 콘텐츠의 핵심 내용을 1~2문장의 자연스러운 한국어로 요약하십시오. (장소 추출 여부와 관계없이 필수 작성)
2. 장소 추출: 본문에 방문 경험이나 목적지로 명확히 언급된 **구체적 장소(POI)**만 추출하십시오.

### 2. 검색 쿼리 생성 규칙
추출된 각 장소에 대해 아래 우선순위로 `search_query`를 생성하십시오.
* **1순위 [주소 기반]:** 본문에 '도로명' 또는 '지번'이 명시된 경우 -> `장소명 + 주소`
* **2순위 [지역/지점 기반]:** 주소가 없는 경우 -> `지점명`이 있으면 포함, 없으면 `행정구역(시/군/구)` 결합.
* **Note:** '동네', '근처' 등의 모호한 표현 대신 상위 행정구역명을 우선 사용하십시오.

### 3. 절대 금지 및 제약 사항
* **No Hallucination:** 본문에 명시되지 않은 지점명이나 상세 주소를 절대 임의로 생성하지 마십시오.
* **Specific POI Only:** 광범위한 지명(서울, 강원도, 제주도 등)은 추출하지 마십시오.
* **명칭 보정:** 오타나 약칭은 문맥을 파악하여 공식 명칭으로 수정하십시오.
* **구체적인 장소(POI)가 하나도 없다면 `places` 리스트는 빈 배열 `[]`로 반환되어야 합니다.**
"""


class PlaceItem(BaseModel):
    place_name: str = Field(description="본문에서 추출한 공식 장소명")
    search_query: str = Field(description="규칙에 따라 생성된 검색 쿼리")


class AnalysisResult(BaseModel):
    summary: str = Field(description="요약문")
    places: list[PlaceItem] = Field(description="장소 목록")


class PlaceExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _construct_user_content(data: dict) -> str:
        video_info = data.get('video_info', {})
        content_data = {
            "title": video_info.get('title', ''),
            "description": video_info.get('description', ''),
            "pinned_comment": data.get('pinned_comment') or "",
            "transcript": data.get('processed_transcript') or "(자막 없음)"
        }
        return json.dumps(content_data, ensure_ascii=False, indent=2)

    def extract(self, source_data: dict) -> dict:
        user_content = self._construct_user_content(source_data)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  # 위험한 콘텐츠
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,  # 증오심 표현 및 콘텐츠
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,  # 괴롭힘 콘텐츠
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  # 음란물
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    )
                ]
            )
        )

        # 정상적으로 응답은 왔는데, 파싱된 데이터가 있는 경우
        if response.parsed:
            result_data = response.parsed.model_dump()

            input_tokens = 0
            output_tokens = 0

            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            return {
                "result": result_data,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            }

        # 에러는 안 났지만 내용이 비어서 온 경우
        return {"summary": "분석 결과 없음 (안전 필터 또는 내용 없음)", "places": []}
