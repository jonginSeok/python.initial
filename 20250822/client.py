# client.py - 비동기 동시 요청 클라이언트
import socket
import threading
import time
import asyncio


class AsyncClient:
    def __init__(self, server_host="localhost", server_port=8888):
        self.server_host = server_host
        self.server_port = server_port

    def send_request(self, message, client_id):
        """단일 요청을 보내는 함수"""
        try:
            # 소켓 연결
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(
                (self.server_host, self.server_port)
            )  # 클라이언트가 서버측에 접속 요청을 보낸다

            # 요청 전송
            print(f"[클라이언트 {client_id}] 요청 전송: {message}")
            client_socket.send(
                message.encode("utf-8")
            )  # 클라이언트가 서버에 메시지 전송

            # 응답 수신
            response = client_socket.recv(1024).decode(
                "utf-8"
            )  # 클라이언트가 서버측에서 온 데이터(응답) 수신
            print(f"[클라이언트 {client_id}] 응답 받음: {response}")

            client_socket.close()  # 소켓 종료

        except Exception as e:
            print(f"[클라이언트 {client_id}] 오류 발생: {e}")

    def send_concurrent_requests_threading(
        self, messages
    ):  # 클라이언트가 서버에 대해서 쓰레드를 사용한 동시 요청
        """스레딩을 사용한 동시 요청"""
        print("\n=== 스레딩을 사용한 동시 요청 ===")
        start_time = time.time()

        threads = []
        for i, message in enumerate(
            messages
        ):  # 전달할 메시지의 수만큼 반복하여 쓰레드 생성 및 실행(메시지가 동시에 병행처리됨)
            thread = threading.Thread(target=self.send_request, args=(message, i + 1))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()  # 데몬 쓰레드는 메인 쓰레드가 종료되면 따라 종료되므로 메인 쓰레드를 대기 상태로 설정

        end_time = time.time()  # 모든 쓰레드가 종료된 후에 실행됨
        print(f"총 소요 시간: {end_time - start_time:.2f}초\n")

    # coroutine 선언 : io 작업 등으로 인해 함수가 지연되는 경우에
    async def async_send_request(
        self, message, client_id
    ):  # await 키워드를 가진 호출을 포함한 경우에는 반드시 async 사용
        """asyncio를 사용한 비동기 요청"""
        try:
            # 비동기 소켓 연결
            reader, writer = (
                await asyncio.open_connection(  # 비동기 소켓을 이용한 접속 요청
                    self.server_host, self.server_port
                )
            )

            # 요청 전송
            print(f"[비동기 클라이언트 {client_id}] 요청 전송: {message}")
            writer.write(message.encode("utf-8"))
            await writer.drain()  # 전송 버퍼의 내용 전송 및 전송 완료시까지 기다림. 리턴될 때까지 기다리려면 반드시 await 키워드를 사용

            # 응답 수신
            response = await reader.read(1024)
            print(
                f"[비동기 클라이언트 {client_id}] 응답 받음: {response.decode('utf-8')}"
            )

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            print(f"[비동기 클라이언트 {client_id}] 오류 발생: {e}")

    async def send_concurrent_requests_async(self, messages):
        """asyncio를 사용한 동시 요청"""
        print("=== asyncio를 사용한 동시 요청 ===")
        start_time = time.time()

        # 모든 요청을 동시에 실행
        tasks = [
            self.async_send_request(
                message, i + 1
            )  # coroutine객체만 생성되므로 로직 실행을 기다리지 않음
            for i, message in enumerate(messages)
        ]

        await asyncio.gather(
            *tasks  # *는 unpack - list를 받는게 아니라, 그 요소들을 받는다
        )  # tasks에는 코루틴 객체가 포함되며 각각의 코루틴이 실행완료 될 때까지 기다림

        end_time = time.time()
        print(f"총 소요 시간: {end_time - start_time:.2f}초\n")


# 클라이언트 실행 예제
def main():
    client = AsyncClient()

    # 테스트 메시지들
    messages = [
        "요청 1: 데이터 조회",
        "요청 2: 파일 업로드",
        "요청 3: 계산 처리",
        "요청 4: 이메일 발송",
        "요청 5: 백업 실행",
    ]

    print("서버가 실행 중인지 확인하고 Enter를 누르세요...")
    input()

    # 1. 스레딩을 사용한 동시 요청
    client.send_concurrent_requests_threading(messages)

    time.sleep(2)  # 잠깐 대기

    # 2. asyncio를 사용한 동시 요청
    asyncio.run(client.send_concurrent_requests_async(messages))


#
if __name__ == "__main__":
    main()
