import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import requests

class NotificationService:
    def __init__(self):
        self.smtp_server = "smtp.example.com"
        self.smtp_port = 587
        self.email_from = "alerts@trafficmonitoring.com"
        self.emergency_contacts = [
            "police@local.gov",
            "ambulance@health.gov",
            "traffic_control@city.gov"
        ]
        
    def send_email_alert(self, alert_data):
        msg = MIMEMultipart()
        msg['Subject'] = f"Traffic Accident Alert - {alert_data['location']}"
        msg['From'] = self.email_from
        msg['To'] = ", ".join(self.emergency_contacts)
        
        # Add text
        text = MIMEText(
            f"Accident detected at {alert_data['timestamp']}\n"
            f"Location: {alert_data['location']}\n"
            f"Severity: {alert_data['severity_score']}"
        )
        msg.attach(text)
        
        # Add image
        with open(alert_data['frame_path'], 'rb') as f:
            img = MIMEImage(f.read())
            msg.attach(img)
            
        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login("username", "password")
            server.send_message(msg)
            
    def call_emergency_api(self, alert_data):
        # Example integration with emergency service API
        payload = {
            "incident_type": "TRAFFIC_ACCIDENT",
            "location": alert_data["location"],
            "timestamp": alert_data["timestamp"],
            "severity": alert_data["severity_score"]
        }
        response = requests.post(
            "https://emergency-api.local.gov/v1/incidents",
            json=payload,
            headers={"Authorization": "Bearer YOUR_API_KEY"}
        )
        return response.status_code == 200