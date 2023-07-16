from datetime import datetime
import io

import socket

class FaceCheckInService:
    def __init__(self, connection):
        self.connection = connection

    def insert_record(self, data):
        ip = socket.gethostbyname(socket.gethostname())

        # Convert the PIL image to bytes
        image = data['faceImage']
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()

        # Prepare the SQL query
        query = "INSERT INTO tblUserHistory(UserId, ApprovalType, ApproveReject, LoginTime, LoginIP, TypeCode, FaceCheckIn) VALUES (?, ?, ?, ?, ?, ?, ?)"
        parameters = (
            data['userId'],
            data['approvalType'],
            1,
            f'{datetime.now():%Y-%m-%d %H:%M:%S%z}',
            ip,
            "CHECKIN",
            image_data
        )

        print(query)
        self.connection.execute_query_with_parameters_no_result(query, parameters)
