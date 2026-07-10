"""Vercel Python serverless entrypoint.

This project is built as a Streamlit app, which is best hosted on
Streamlit Community Cloud. This WSGI app keeps Vercel deployment valid
and returns basic usage information.
"""


def app(environ, start_response):
    body = (
        "ClimbAssist AI is a Streamlit application.\n\n"
        "Recommended hosting: Streamlit Community Cloud\n"
        "Main file: app_v2.py\n"
        "Run locally: streamlit run app_v2.py\n"
    ).encode("utf-8")

    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]
