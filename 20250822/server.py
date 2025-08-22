# server.py - 간단한 TCP 서버
import socket
import threading
import time


def handle_client(client_socket, address):  # 클라인트의 접속요청이 왔을 때 호출
    """클라이언트 요청을 처리하는 함수"""
    try:
        while True:
            data = client_socket.recv(1024).decode(
                "utf-8"
            )  # 클라이언트의 메시지가 수신되었을 때
            if not data:
                break

            print(f"[서버] {address}에서 받은 메시지: {data}")

            # 처리 시간 시뮬레이션 (1-3초)
            import random

            time.sleep(random.uniform(1, 3))  # 무작위 시간동안(1~3초) 쉰다

            # 클라이언트에게 응답 전송
            response = f"서버 응답: {data}를 처리했습니다"
            client_socket.send(response.encode("utf-8"))  # 클라이언트에게 메시지 전송

    except Exception as e:
        print(f"[서버] 오류 발생: {e}")
    finally:
        client_socket.close()
        print(f"[서버] {address} 연결 종료")


def start_server():
    """서버 시작"""
    # socket.AF_INET: 주소 체계로 IPv4를 사용, socket.SOCK_STREAM: 소켓 타입으로 TCP를 사용
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # socket.SOL_SOCKET: 소켓 레벨의 옵션을 설정하겠다는 의미
    # socket.SO_REUSEADDR: 소켓이 이전에 사용했던 주소와 포트를 재사용할 수 있도록 설정
    # 이는 서버를 재시작할 때 "Address already in use" 오류를 방지하는 데 유용
    server.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
    )  # 1은 이 옵션을 활성화하겠다는 설정
    server.bind(("localhost", 8888))  # ('localhost', 8888): 바인딩할 주소와 포트
    server.listen(5)  # 5: 대기 큐의 크기를 지정. 그 이상은 연결 거부될 수 있음

    print("[서버] 포트 8888에서 서버 시작...")

    try:
        while (
            True
        ):  # 접속 요청이 오면 쓰레드를 생성하여 실행(접속 클라이언트 당 1개의 쓰레드가 생성됨)
            client_socket, address = server.accept()  # 클라이언트 접속 요청이 오면...
            print(f"[서버] {address}에서 연결됨")

            # 각 클라이언트를 별도 스레드에서 처리
            client_thread = threading.Thread(  # 쓰레드 생성
                target=handle_client,  # 쓰레드.start() 호출시 실행될 함수
                args=(client_socket, address),  # 쓰레드 함수로 전달될 아규먼트
            )
            client_thread.daemon = (
                True  # 메인 쓰레드가 죽으면 함께 죽는 데몬 쓰레드로 설정
            )
            client_thread.start()  # 쓰레드 실행 -> 쓰레드 함수 실행됨

    except KeyboardInterrupt:
        print("\n[서버] 서버 종료")
    finally:
        server.close()


if __name__ == "__main__":
    start_server()