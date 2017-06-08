#!/usr/bin/python3

import pymysql

# Open database connection
db = pymysql.connect("localhost","root","newpass","odp_3" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()

print ("Database version : %s " % data)

sql = "SELECT * FROM SITE WHERE cid = '%d'" % (3)
try:
   # Execute the SQL command
   cursor.execute(sql)
   # Fetch all the rows in a list of lists.
   results = cursor.fetchall()
   for row in results:
      sid = row[0]
      cid = row[1]
      title = row[2]
      desc = row[3]
      url = row[4]
      # Now print fetched result
      print ("sid = %d,cid = %d,title = %s,desc = %s,url = %s" % \
             (sid, cid, title, desc, url ))
except:
   print ("Error: unable to fetch data")

# disconnect from server
db.close()
