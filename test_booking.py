import requests
import json

DEVICE_ID = "c5485e8f-2130-4844-b993-2591a2e24599"

session = requests.Session()
session.headers.update({
    "accept": "*/*",
    "accept-language": "en",
    "clientid": "web",
    "content-type": "application/json;charset=UTF-8",
    "operid": "2",
    "origin": "https://www.sportybet.com",
    "platform": "web",
    "referer": "https://www.sportybet.com/ng/sport/football/",
    "sporty-referer": "utm_source=https://www.google.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
})
session.cookies.set("locale",     "en",       domain="www.sportybet.com")
session.cookies.set("device-id",  DEVICE_ID,  domain="www.sportybet.com")
session.cookies.set("sb_country", "ng",        domain="www.sportybet.com")

# Test with a single 1X2 Home selection — use a real event_id from today's fixtures
payload = {
    "selections": [
        {
            "eventId":   "sr:match:67126642",  # replace with a real today fixture
            "marketId":  "1",
            "specifier": None,
            "outcomeId": "1"
        }
    ]
}

r = session.post("https://www.sportybet.com/api/ng/orders/share",
                 json=payload, timeout=20)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")