from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import base64
import json
import traceback
from sockets.services.socket_services import FaceServices

face_service = FaceServices(threshold=1.3)
print("Initialize Socket")
websocket_connections = []
socket_router = APIRouter()


@socket_router.websocket("/ws-face")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        websocket_connections.append(websocket)
        print("Connections: ", websocket_connections[0].client.host)
        while True:
            print("websocket")
            try:
                data = await websocket.receive_text()
            except Exception as e:
                print("Client disconnected 1: ", str(e))
                traceback.print_exc()
                websocket_connections.remove(websocket)
                break

            json_data = json.loads(data)
            print(json_data['Image'][:10])
            # img = base64.b64decode(json_data['Image'])
            #
            # img = face_service.convertbase64(img)
            # data = face_service(img)
            # json_data = json.dumps(data)
            # print(json_data)
            await websocket.send_text('')

    except Exception as e:
        print('Client disconnected 2: ', str(e))
        return websocket_connections.remove(websocket)
        # return await websocket.close()
