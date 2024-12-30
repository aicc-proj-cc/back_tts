from vits_service import TTSService
import pika
import json
import os
from scipy.io import wavfile
import threading
import base64

# RabbitMQ 설정
RABBITMQ_HOST = "222.112.27.120"  # RabbitMQ 서버의 실제 IP 또는 도메인
USERNAME = "guest"  # RabbitMQ 사용자 이름
PASSWORD = "guest"  # RabbitMQ 사용자 비밀번호
REQUEST_QUEUE = "tts_generation_requests"
RESPONSE_QUEUE = "tts_generation_responses"

# TTS 서비스 초기화
config_path = "./configs/modified_finetune_speaker.json"
model_checkpoint = "./GENSHIN_VITS_MODEL/G_latest.pth"
tts_service = TTSService(config_path, model_checkpoint)

# 오디오 파일 저장 폴더 설정
OUTPUT_FOLDER = "output_audio"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # 폴더가 없으면 생성


def generate_tts(data):
    """
    TTS를 생성하고 결과 오디오 파일 경로를 반환.
    """
    text = data.get("text", "default text")
    speaker = data.get("speaker", 0)
    language = data.get("language", "en")
    speed = data.get("speed", 1.0)

    print("text:", text, "speaker:", speaker, "language:",language, "speed:", speed)
    # TTS 생성
    message, result = tts_service.generate_tts(text, speaker, language, speed)
    print(f"TTS 서비스 호출 결과: {message}, {result}")

    if message == "Success":
        sampling_rate, audio_data = result
        output_path = os.path.join(OUTPUT_FOLDER, f"{data['id']}.wav")
        wavfile.write(output_path, sampling_rate, audio_data)
        return "success", output_path
    else:
        return "error", result


def async_save_tts(audio_path):
    """
    비동기적으로 TTS 파일 저장 (현재 동작은 이미 동기적이므로 필요 시 수정 가능).
    """
    print(f"TTS 파일 저장 완료: {audio_path}")



def callback(ch, method, properties, body):
    data = json.loads(body)
    print(f"TTS 생성 요청 수신: {data['id']}")

    try:
        # TTS 생성
        status, result = generate_tts(data)

        if status == "success":
            file_path = result
            with open(file_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

            response = {"id": data["id"], "status": "success", "audio_base64": audio_base64}
        else:
            response = {"id": data["id"], "status": "error", "error": result}

        ch.basic_publish(
            exchange="",
            routing_key=RESPONSE_QUEUE,
            body=json.dumps(response),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"TTS 생성 중 오류 발생: {e}")
        response = {"id": data.get("id", "unknown"), "status": "error", "error": str(e)}
        ch.basic_publish(
            exchange="",
            routing_key=RESPONSE_QUEUE,
            body=json.dumps(response),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)



# RabbitMQ 연결 및 소비자 실행
try:
    credentials = pika.PlainCredentials(USERNAME, PASSWORD)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=5675, credentials=credentials, heartbeat=6000)
    )
    print("RabbitMQ에 연결 성공")
except Exception as e:
    print(f"RabbitMQ 연결 실패: {e}")
    raise

channel = connection.channel()

# 큐 선언
channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)

# prefetch_count 설정
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=REQUEST_QUEUE, on_message_callback=callback)

print("TTS 생성 서버가 요청을 대기 중입니다...")
channel.start_consuming()


# uvicorn main:app --reload --log-level debug --port 8004