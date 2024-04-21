from pymongo import MongoClient
import urllib

username="ronidas" 
pwd="q3uBTgCcfxicDkBX"
client = MongoClient("mongodb+srv://"+urllib.parse.quote_plus(username)+":"+urllib.parse.quote_plus(pwd)+"@cluster0.lymvb.mongodb.net/?retryWrites=true&w=majority")


db_name=client["sample_airbnb"]
collection=db_name["listingsAndReviews"]
result=collection.aggregate([
  { "$group": { "_id": "$host.host_id", "total": { "$sum": "$host.host_listings_count" } } },
  { "$sort": { "total": -1 } },
  { "$limit": 1 },
  { "$lookup": { "from": "listings", "localField": "_id", "foreignField": "host.host_id", "as": "host_info" } },
  { "$project": { "host_id": "$_id", "host_name": "$host_info.host.host_name", "_id": 0 } }
])
for doc in result:
    print(doc)