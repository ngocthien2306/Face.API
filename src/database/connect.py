import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

server = os.getenv("SERVER")
database = os.getenv("DATABASE")
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")


class DatabaseConnection:
    def __init__(self, server, database, username, password, driver='ODBC Driver 17 for SQL Server'):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.driver = driver
        self.connection = None
        self.cursor = None

    def connect(self):
        connection_string = f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID=parkingai;PWD={self.password}"
        self.connection = pyodbc.connect(connection_string)
        self.cursor = self.connection.cursor()

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query):
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows

    def execute_query_with_parameters_no_result(self, query, parameters):
        self.cursor.execute(query, parameters)
        self.connection.commit()

    def execute_query_with_parameters(self, query, parameters):
        self.cursor.execute(query, parameters)
        rows = self.cursor.fetchall()
        return rows


class DeviceService:
    def __init__(self, connection):
        self.connection = connection

    def get_all_records(self):
        query = "SELECT * FROM tblStoreDevice"
        rows = self.connection.execute_query(query)
        return rows
    def get_by_mac_address(self, mac: str):
        query = "SELECT * FROM tblStoreDevice WHERE DeviceKeyNo = '{0}' AND DeviceType = 'DVC003'".format(mac)
        print(query)
        rows = self.connection.execute_query(query)
        return rows
    def get_records_by_condition(self, condition):
        query = "SELECT * FROM YourTable WHERE " + condition
        rows = self.connection.execute_query(query)
        return rows

    def insert_record(self, data):
        query = "INSERT INTO YourTable (Column1, Column2) VALUES (?, ?)"
        parameters = (data['column1'], data['column2'])
        self.connection.execute_query_with_parameters(query, parameters)

    def update_record(self, record_id, data):
        query = "UPDATE YourTable SET Column1 = ?, Column2 = ? WHERE Id = ?"
        parameters = (data['column1'], data['column2'], record_id)
        self.connection.execute_query_with_parameters(query, parameters)

    def delete_record(self, record_id):
        query = "DELETE FROM YourTable WHERE Id = ?"
        parameters = (record_id,)
        self.connection.execute_query_with_parameters(query, parameters)


# Get the connection details from environment variables
connection = DatabaseConnection(server=server, database=database, username=username, password=password)
