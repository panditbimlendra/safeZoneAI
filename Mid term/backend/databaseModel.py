from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["accident_detection"]

class AlertDB:
    @staticmethod
    def save_alert(alert_data):
        db.alerts.insert_one(alert_data)
    
    @staticmethod
    def get_alerts(start_time=None, end_time=None):
        query = {}
        if start_time and end_time:
            query = {
                "timestamp": {
                    "$gte": datetime.fromisoformat(start_time),
                    "$lte": datetime.fromisoformat(end_time)
                }
            }
        return list(db.alerts.find(query).sort("timestamp", -1))