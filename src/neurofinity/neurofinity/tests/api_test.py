import requests
import json

from neurofinity.config import INTERIM_DATA_DIR

response = requests.post(
    "http://localhost:8000/mindmap/generate",
    json={
        "text": "John: Good morning everyone. Let's start with the Q3 roadmap. "
            "Sarah: I think we should prioritize the mobile app redesign. "
            "Our user research shows 60% of users prefer mobile. "
            "John: Good point. What about the API performance issues? "
            "Mike: I've been looking into that. We need to optimize the "
            "database queries. I estimate it'll take two sprints. "
            "Sarah: Can we run both in parallel? "
            "John: Yes, let's do that. Mike, you own the API optimization. "
            "Sarah, you lead the mobile redesign. "
            "Mike: Sounds good. I'll have a plan by Friday. "
            "John: Great. Let's reconvene next week.",
        "title": "My Meeting",
        "backend": "claude",
    },
)
data = response.json()
json.dump(data, open(INTERIM_DATA_DIR / "mindmap_test.json", "w"))