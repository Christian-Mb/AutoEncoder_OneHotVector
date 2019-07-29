import mysql.connector
class Connect:


    db = mysql.connector.connect(
        host = "localhost",
        user = "root",
        passwd = "root",
        port = 8889,
        database = "agenda"
    )
    