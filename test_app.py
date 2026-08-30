from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=60)
at.session_state["live_detector"] = None
at.run()
print("exceptions:", len(at.exception))
for e in at.exception:
    print("EXC:", e.value)
    if hasattr(e, "stack_trace"):
        print(e.stack_trace)
print("error count:", len(at.error))
print("OK")
