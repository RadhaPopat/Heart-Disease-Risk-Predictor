from fastapi import FastAPI

app = FastAPI(
    title="CardioVascular Risk API",
    description="API for cardiovascular risk prediction",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "CardioVascular Risk API is running"
    }