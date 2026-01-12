import json
import logging
import os
import time

import boto3

from src.place_extractor import PlaceExtractor

# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 클라이언트 초기화
sts = boto3.client('sts')
s3 = boto3.client('s3')
ssm = boto3.client('ssm')
CACHED_API_KEY = None

# 계정 ID 조회
CURRENT_ACCOUNT_ID = sts.get_caller_identity()["Account"]


def get_api_key():
    """SSM Parameter Store에서 API 키 가져오기"""
    global CACHED_API_KEY

    # 1. 이미 메모리에 있으면 리턴
    if CACHED_API_KEY:
        return CACHED_API_KEY

    # 2. 없으면 SSM 호출해서 가져오기
    try:
        logger.info("SSM에서 API 키 가져오는 중...")
        response = ssm.get_parameter(
            Name='/place-extractor-lambda/gemini-api-key',
            WithDecryption=True
        )
        CACHED_API_KEY = response['Parameter']['Value']
        return CACHED_API_KEY
    except Exception as e:
        logger.error(f"API 키 가져오기 실패: {e}")
        raise e


def get_source_data_from_s3(bucket_name, video_id):
    """S3에서 소스 데이터 읽기"""
    key = f"{video_id}/source.json"
    response = s3.get_object(
        Bucket=bucket_name,
        Key=key,
        ExpectedBucketOwner=CURRENT_ACCOUNT_ID
    )
    return json.loads(response['Body'].read().decode('utf-8'))


def save_result_to_s3(bucket_name, video_id, data):
    """분석 결과를 S3에 저장"""
    key = f"{video_id}/extracted.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2),
        ContentType='application/json',
        ExpectedBucketOwner=CURRENT_ACCOUNT_ID
    )
    return key


def lambda_handler(event, context):
    logger.info(f"Lambda invoked. Video ID: {event.get('video_id')}")

    video_id = event.get('video_id')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    api_key = get_api_key()
    model_name = os.environ.get('MODEL_NAME')

    if not video_id or not bucket_name:
        raise ValueError(f"설정 누락: video_id={video_id}, bucket={bucket_name}")

    try:
        # 1. 데이터 가져오기
        logger.info(f"S3 소스 데이터 다운로드 중... (Bucket: {bucket_name})")
        source_data = get_source_data_from_s3(bucket_name, video_id)

        # 2. 분석 실행
        extractor = PlaceExtractor(api_key=api_key, model_name=model_name)
        logger.info(f"Gemini 분석 시작 (Model: {extractor.model_name})")

        start_time = time.time()
        output = extractor.extract(source_data)
        elapsed = time.time() - start_time

        # 3. 로그용 지표
        metrics = {
            "event": "LLM_METRICS",
            "video_id": video_id,
            "latency": round(elapsed, 2),
            "input_tokens": output['usage']['input_tokens'],
            "output_tokens": output['usage']['output_tokens'],
            "places_count": len(output['result'].get('places', []))
        }
        logger.info(json.dumps(metrics))

        # 4. 결과를 S3에 저장
        logger.info("분석 결과 S3 업로드 중...")
        saved_key = save_result_to_s3(bucket_name, video_id, output)
        logger.info(f"처리 완료. (Key: {saved_key})")

        return {
            'statusCode': 200,
            'video_id': video_id,
            's3_key': saved_key
        }

    except Exception as e:
        logger.error("처리 중 오류 발생", exc_info=True)
        raise e
