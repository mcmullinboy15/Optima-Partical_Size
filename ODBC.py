import pyodbc


def read(conn):
    cursor = conn.cursor()
    cursor.execute('Select Value from tags where Tagname = ?;', ('RPM_Mill1_SQL'))

    for row in cursor:
        var = row
    return ( var )


conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=WIN-I0Q93QA0NCN\SQLEXPRESS;"
                      "Database=optima;"
                      "Trusted_Connection=yes;")

RPM = read(conn)[0]
conn.close()


def update(conn):
    cursor = conn.cursor()
    cursor.execute('update tags set Value = ? where Tagname = ?;', (peak_range[least_different_cycle, 0], 'Peak_Start'))
    cursor.execute("update tags set Value = ? where Tagname = ?;", (peak_range[least_different_cycle, 1], 'Peak_End'))
    conn.commit()
    conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                          "Server=WIN-I0Q93QA0NCN\SQLEXPRESS;"
                          "Database=optima;"
                          "Trusted_Connection=yes;")


update(conn)
conn.close()