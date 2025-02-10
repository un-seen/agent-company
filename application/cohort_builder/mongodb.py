import os
from pymongo import MongoClient

MONGO_URI = "mongodb://mongo:nGsLscIRmxgkGxFfKDdlKdopkrCzipis@junction.proxy.rlwy.net:45148" # os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_SYSTEM_URI = "mongodb://mongo:yBOctqkWozxRqVgNBNqXupbDiSkNOhFB@junction.proxy.rlwy.net:30525" # os.getenv("MONGO_SYSTEM_URI")

def mongodb_connect() -> MongoClient:
    client = MongoClient(MONGO_URI)
    return client

def mongodb_system_connect() -> MongoClient:
    client = MongoClient(MONGO_SYSTEM_URI)
    return client