from pymongo import MongoClient
import urllib

username="ronidas"
pwd="okZAaW0eTqKqfCwh"
client=MongoClient("mongodb+srv://"+urllib.parse.quote(username)+":"+urllib.parse.quote(pwd)+"@cluster0.lymvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db=client["invoice"]
collection=db["invoice"]
result = collection.aggregate([{'$unwind': '$product'}, {'$group': {'_id': '$product.item', 'price': {'$first': '$product.single_unit_price.$numberInt'}}}, {'$project': {'_id': 0, 'product_name': '$_id', 'price': 1}}])
for doc in result:
    print(doc)