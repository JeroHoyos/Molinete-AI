# Echo App — FastAPI + React

## Structure
```
├── main.py
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── img/1.png        ← place your background image here
└── templates/
    └── index.html
```

## Setup & Run
```bash
pip install fastapi uvicorn python-multipart jinja2
```

Add your background image to `static/img/1.png`, then:

```bash
python main.py
```

Open http://localhost:8000

## How it works
1. User types a message in the frontend input
2. On submit, a POST request is sent to `/echo`
3. The backend **prints** the message to the terminal
4. The backend returns `"<your message> message"` as JSON
5. The frontend displays the server response