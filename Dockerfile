# AWS Lambda 공식 Python 3.12 이미지 사용
FROM public.ecr.aws/lambda/python:3.12

# requirements.txt 복사 및 라이브러리 설치
# ${LAMBDA_TASK_ROOT}는 람다 코드가 위치해야 할 기본 경로(/var/task)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 실행할 핸들러 지정 (파일명.함수명)
CMD [ "lambda_function.lambda_handler" ]
