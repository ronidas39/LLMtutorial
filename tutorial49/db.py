import pyodbc

server="localhost"
database="city_info"

def genCarInfo(carno):
    carno_value=carno
    try:
        conn=pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            f'SERVER={server};'
                            f'DATABASE={database};'
                            'Trusted_Connection=yes;')

        cursor=conn.cursor()
        query=f"select * from cars where CARNO = ?;"
        cursor.execute(query,carno_value)
        rows=cursor.fetchall()
        if rows:
            for row in rows:
                print(row)
        else:
            print(f"no car info found for carno: {carno_value}")
    except pyodbc.Error as error:
        print(error)
    finally:
        if conn:
            conn.close()